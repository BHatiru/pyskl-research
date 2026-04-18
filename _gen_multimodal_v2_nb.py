"""Generate train_colab_multimodal_v2.ipynb — follow-up multimodal experiments.

Experiments:
  A. Extended Late Fusion (4 sensors, 12 channels)
  B. Fusion-GCN (IMU as virtual nodes in skeleton graph)
  C. Comparison chart: skeleton-only vs late-6ch vs late-12ch vs Fusion-GCN
"""
import json, math

TARGET = r'd:\Research\pyskl-research\demo1_fed_skeleton\train_colab_multimodal_v2.ipynb'

def src(text):
    text = text.strip('\n')
    if not text:
        return ['']
    lines = text.split('\n')
    if len(lines) == 1:
        return [lines[0]]
    return [l + '\n' for l in lines[:-1]] + [lines[-1]]

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "source": src(text),
            "execution_count": None, "outputs": []}

cells = []

# ═══════════════════════════════════════════════════════════
# CELL 1: Title
# ═══════════════════════════════════════════════════════════
cells.append(md("""\
# Multimodal Exploration v2: Fusion-GCN & Extended Sensors

**Previous result** (v1): Late Fusion (2 sensors, 6ch) scored **77.3%** vs Skeleton-only **79.3%** → synthetic IMU *hurt* by -2.0 pp.

**Hypothesis**: Late fusion treats modalities independently — the IMU encoder can't leverage spatial relationships. Graph-level fusion (Fusion-GCN) should extract complementary information by embedding IMU sensors as nodes in the skeleton graph.

## Experiments
| # | Method | Key Change | Expected |
|---|--------|-----------|----------|
| A | Skeleton-only (baseline repeat) | Same as v1, same data subset | ~79% |
| B | Late Fusion 12ch | 4 sensors (wrist×2, waist, ankle) → 12ch | Marginal ↑ |
| C | **Fusion-GCN** | IMU as virtual graph nodes connected to anatomy | Best |

## Fusion-GCN Architecture
```
COCO-17 skeleton graph + K virtual IMU nodes
  Virtual node 17 ←→ Joint 11 (right wrist)
  Virtual node 18 ←→ Joint  0 (waist)
  Virtual node 19 ←→ Joint  7 (left wrist)
  Virtual node 20 ←→ Joint 15 (right ankle)

Input: (N, 2, 100, 21, 3)  ← 17 skeleton + 4 IMU nodes
Graph: 21-node adjacency with inter-modal edges
```
Reference: Duhme et al. "Fusion-GCN" (GCPR 2021) reported +12.4% F1 with skeleton+IMU graph fusion."""))

# ═══════════════════════════════════════════════════════════
# CELL 2: Setup
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
# @title Setup {display-mode: "form"}
!pip install -q scikit-learn

import os, time, copy, json, math, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter, defaultdict, OrderedDict
from scipy.signal import butter, filtfilt

print(f"PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
assert torch.cuda.is_available(), "No GPU! Runtime → Change runtime type → T4 GPU"
DEVICE = 'cuda'"""))

# ═══════════════════════════════════════════════════════════
# CELL 3: Config
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
# ═══ Config ═══
SEED = 42
NUM_CLASSES = 10
BATCH_SIZE = 64
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
QUICK_EPOCHS = 10

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
os.makedirs('outputs', exist_ok=True)

MEDICAL_CLASS_MAP = {42: 0, 41: 1, 43: 2, 44: 3, 45: 4, 46: 5, 47: 6, 8: 7, 7: 8, 58: 9}
MEDICAL_LABELS = [
    'falling', 'staggering', 'touch head', 'touch chest', 'touch back',
    'touch neck', 'nausea', 'standing up', 'sitting down', 'walking towards',
]

# v1 results for comparison
V1_SKEL_BEST = 0.793
V1_LATE_BEST = 0.773

print('Config set ✓')"""))

# ═══════════════════════════════════════════════════════════
# CELL 4: Data heading
# ═══════════════════════════════════════════════════════════
cells.append(md("## 1. Data Download & Preprocessing"))

# ═══════════════════════════════════════════════════════════
# CELL 5: Download
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
data_dir = 'data/nturgbd'
pkl_path = f'{data_dir}/ntu60_hrnet.pkl'
pkl_3d = f'{data_dir}/ntu60_3danno.pkl'

for url, path in [
    ('https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl', pkl_path),
    ('https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl', pkl_3d),
]:
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        !wget -q --show-progress -O {path} {url}
        print(f'Downloaded: {path}')
    else:
        print(f'Already exists: {path}')

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
with open(pkl_3d, 'rb') as f:
    data_3d = pickle.load(f)

split_info = data['split']
annotations = data['annotations']
ann_3d = data_3d['annotations']

selected = set(MEDICAL_CLASS_MAP.keys())
medical_anns = [a for a in annotations if a['label'] in selected]
ann_3d_lookup = {ann['frame_dir']: ann for ann in ann_3d}
medical_3d = [(ann, ann_3d_lookup[ann['frame_dir']])
              for ann in medical_anns if ann['frame_dir'] in ann_3d_lookup]
print(f'Medical samples with 3D data: {len(medical_3d)}')"""))

# ═══════════════════════════════════════════════════════════
# CELL 6: Preprocess functions
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
def prenormalize_2d(kp, ks, img_shape=(1080, 1920), threshold=0.01):
    kp = kp.astype(np.float32).copy()
    ks = ks.astype(np.float32)
    kp = np.concatenate([kp, ks[..., None]], axis=-1)
    mask = kp[..., 2] <= threshold
    h, w = img_shape
    kp[..., 0] = (kp[..., 0] - w/2) / (w/2)
    kp[..., 1] = (kp[..., 1] - h/2) / (h/2)
    kp[..., 0][mask] = 0; kp[..., 1][mask] = 0
    return kp

def uniform_sample(kp, clip_len=100):
    M, T, V, C = kp.shape
    if T == clip_len: return kp
    inds = np.arange(clip_len) % T if T < clip_len else np.linspace(0, T-1, clip_len, dtype=int)
    return kp[:, inds]

def format_gcn(kp, num_person=2):
    M, T, V, C = kp.shape
    if M < num_person:
        kp = np.concatenate([kp, np.zeros((num_person-M, T, V, C), dtype=kp.dtype)], 0)
    return kp[:num_person]

def preprocess(ann):
    kp = prenormalize_2d(ann['keypoint'], ann['keypoint_score'],
                         ann.get('img_shape', (1080, 1920)))
    return format_gcn(uniform_sample(kp, 100))

print('Preprocess functions defined ✓')"""))

# ═══════════════════════════════════════════════════════════
# CELL 7: IMU heading
# ═══════════════════════════════════════════════════════════
cells.append(md("""\
## 2. Synthetic IMU Generation (Extended: 4 Sensors)

| Sensor | NTU Joint | Rationale |
|--------|-----------|-----------|
| Right wrist | 11 | Touch gesture hand |
| Left wrist | 7 | Bilateral arm movement |
| Waist/spine | 0 | Core body dynamics |
| Right ankle | 15 | Gait, falls, lower body |"""))

# ═══════════════════════════════════════════════════════════
# CELL 8: IMU generation
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
def skeleton_to_imu(skeleton_3d, joint_idx, fps=30):
    \"\"\"Convert 3D joint trajectory → synthetic 3-axis accelerometer.\"\"\"
    pos = skeleton_3d[0, :, joint_idx, :]  # (T, 3) first person
    dt = 1.0 / fps
    accel = np.diff(pos, n=2, axis=0) / (dt ** 2)
    if len(accel) < 4:
        return accel
    try:
        nyq = fps / 2.0
        cutoff = min(12.0, nyq * 0.8)
        b, a = butter(2, cutoff / nyq, btype='low')
        accel = filtfilt(b, a, accel, axis=0)
    except Exception:
        pass
    accel += np.random.normal(0, 0.05, accel.shape)
    return accel.astype(np.float32)

# 2-sensor config (same as v1)
IMU_JOINTS_2 = {'right_wrist': 11, 'waist': 0}

# 4-sensor config (extended)
IMU_JOINTS_4 = {
    'right_wrist': 11,
    'left_wrist': 7,
    'waist': 0,
    'right_ankle': 15,
}

def generate_imu_features(ann_3d, clip_len=100, imu_joints=None):
    \"\"\"Generate multi-channel synthetic IMU from 3D skeleton.\"\"\"
    if imu_joints is None:
        imu_joints = IMU_JOINTS_2
    kp3d = ann_3d['keypoint']  # (M, T, 25, 3)
    imu_channels = []
    for joint_name, joint_idx in imu_joints.items():
        accel = skeleton_to_imu(kp3d, joint_idx)
        imu_channels.append(accel)
    imu = np.concatenate(imu_channels, axis=-1)
    T_imu = len(imu)
    if T_imu < clip_len:
        inds = np.arange(clip_len) % T_imu
    else:
        inds = np.linspace(0, T_imu - 1, clip_len, dtype=int)
    imu_sampled = imu[inds]
    mx = np.abs(imu_sampled).max()
    if mx > 0:
        imu_sampled = imu_sampled / mx
    return imu_sampled

# Test
test_imu_6 = generate_imu_features(medical_3d[0][1], imu_joints=IMU_JOINTS_2)
test_imu_12 = generate_imu_features(medical_3d[0][1], imu_joints=IMU_JOINTS_4)
print(f'2-sensor IMU: {test_imu_6.shape}')   # (100, 6)
print(f'4-sensor IMU: {test_imu_12.shape}')  # (100, 12)"""))

# ═══════════════════════════════════════════════════════════
# CELL 9: STGCN++ model
# ═══════════════════════════════════════════════════════════
cells.append(md("## 3. Models"))

# ═══════════════════════════════════════════════════════════
# CELL 10: STGCN++ model code
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
# ═══ STGCN++ (supports variable num_joints and custom adjacency) ═══
COCO_INWARD = [
    (15,13),(13,11),(16,14),(14,12),(11,5),(12,6),
    (9,7),(7,5),(10,8),(8,6),(5,0),(6,0),(1,0),(3,1),(2,0),(4,2)]
COCO_NUM_JOINTS = 17; COCO_CENTER = 0

def _build_adj(num_node, edges, center):
    def _norm(mx):
        d = mx.sum(0); d[d==0]=1; return mx/d
    I = np.eye(num_node, dtype=np.float32)
    inw = np.zeros((num_node, num_node), dtype=np.float32)
    outw = np.zeros((num_node, num_node), dtype=np.float32)
    hop = np.full(num_node, 1e9, dtype=int); hop[center] = 0
    adj = {i: [] for i in range(num_node)}
    for u, v in edges: adj[u].append(v); adj[v].append(u)
    q = [center]
    while q:
        cur = q.pop(0)
        for nb in adj[cur]:
            if hop[nb] > hop[cur]+1: hop[nb]=hop[cur]+1; q.append(nb)
    for u, v in edges:
        if hop[u] < hop[v]: inw[v,u] = 1
        else: outw[u,v] = 1
    return np.stack([_norm(I), _norm(inw), _norm(outw)])

def get_coco_adj():
    return _build_adj(COCO_NUM_JOINTS, COCO_INWARD, COCO_CENTER)

class MsTCN(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1,2,3,4), stride=1, dropout=0.0):
        super().__init__()
        mid = out_ch // (len(dilations)+2); rem = out_ch - mid*(len(dilations)+2)
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch,mid,1), nn.BatchNorm2d(mid), nn.ReLU(True),
                nn.Conv2d(mid,mid,(3,1),stride=(stride,1),padding=(d,0),dilation=(d,1)),
                nn.BatchNorm2d(mid)))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch,mid,1), nn.BatchNorm2d(mid), nn.ReLU(True),
            nn.MaxPool2d((3,1),stride=(stride,1),padding=(1,0)), nn.BatchNorm2d(mid)))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_ch,mid+rem,1), nn.BatchNorm2d(mid+rem)))
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        outs = [br(x) for br in self.branches]
        mt = min(o.size(2) for o in outs)
        return self.drop(torch.cat([o[:,:,:mt,:] for o in outs], dim=1))

class UnitGCN(nn.Module):
    def __init__(self, in_ch, out_ch, A, adaptive=True):
        super().__init__()
        K = A.size(0); self.K = K; self.register_buffer('A', A)
        self.conv = nn.ModuleList([nn.Conv2d(in_ch,out_ch,1) for _ in range(K)])
        self.PA = nn.Parameter(A.clone()) if adaptive else None
        self.bn = nn.BatchNorm2d(out_ch)
        self.res = nn.Sequential(nn.Conv2d(in_ch,out_ch,1),nn.BatchNorm2d(out_ch)) if in_ch!=out_ch else nn.Identity()
        for m in self.conv: nn.init.kaiming_normal_(m.weight, mode='fan_out')
    def forward(self, x):
        adj = self.A + self.PA if self.PA is not None else self.A
        out = sum(torch.einsum('nctv,vw->nctw', self.conv[k](x), adj[k]) for k in range(self.K))
        return F.relu(self.bn(out) + self.res(x), inplace=True)

class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, dropout=0.0):
        super().__init__()
        self.gcn = UnitGCN(in_ch, out_ch, A)
        self.tcn = MsTCN(out_ch, out_ch, stride=stride, dropout=dropout)
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,1), nn.BatchNorm2d(out_ch),
            nn.AvgPool2d((stride,1)) if stride>1 else nn.Identity()
        ) if in_ch!=out_ch or stride!=1 else nn.Identity()
    def forward(self, x):
        return F.relu(self.tcn(self.gcn(x)) + self.residual(x), inplace=True)

class STGCN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, base_channels=64,
                 num_stages=6, inflate_stages=None, down_stages=None,
                 num_person=2, dropout=0.0, num_joints=17, adj_fn=get_coco_adj):
        super().__init__()
        if inflate_stages is None: inflate_stages = [3,5]
        if down_stages is None: down_stages = [3,5]
        self.num_person = num_person; self.num_joints = num_joints
        A = torch.tensor(adj_fn(), dtype=torch.float32)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        channels = [base_channels]
        for i in range(1, num_stages):
            channels.append(channels[-1]*2 if i in inflate_stages else channels[-1])
        self.blocks = nn.ModuleList()
        c_in = in_channels
        for i in range(num_stages):
            self.blocks.append(STGCNBlock(c_in, channels[i], A,
                               stride=2 if i in down_stages else 1, dropout=dropout))
            c_in = channels[i]
        self.out_channels = channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0/num_classes))
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        N, M, T, V, C = x.shape
        x = x.permute(0,1,4,2,3).contiguous().view(N*M, C, T, V)
        x = x.permute(0,1,3,2).contiguous().view(N*M, C*V, T)
        x = self.data_bn(x)
        x = x.view(N*M, C, V, T).permute(0,1,3,2).contiguous()
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).view(N, M, self.out_channels).mean(dim=1)
        return self.fc(self.drop(x))

    def get_features(self, x):
        N, M, T, V, C = x.shape
        x = x.permute(0,1,4,2,3).contiguous().view(N*M, C, T, V)
        x = x.permute(0,1,3,2).contiguous().view(N*M, C*V, T)
        x = self.data_bn(x)
        x = x.view(N*M, C, V, T).permute(0,1,3,2).contiguous()
        for block in self.blocks:
            x = block(x)
        return self.pool(x).view(N, M, self.out_channels).mean(dim=1)

def build_model(num_classes=10, in_channels=3, **kw):
    return STGCN(num_classes=num_classes, in_channels=in_channels, **kw)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print('STGCN++ defined ✓')"""))

# ═══════════════════════════════════════════════════════════
# CELL 11: IMU Encoder + Late Fusion (reused from v1)
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
class IMUEncoder(nn.Module):
    \"\"\"1D-CNN for IMU time series.\"\"\"
    def __init__(self, in_channels=6, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (N, T, C) → (N, C, T)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class LateFusionModel(nn.Module):
    \"\"\"STGCN++ + IMU encoder → concat → classifier.\"\"\"
    def __init__(self, num_classes=10, imu_channels=6, imu_feat=128, **stgcn_kwargs):
        super().__init__()
        self.skeleton_enc = STGCN(num_classes=num_classes, **stgcn_kwargs)
        self.skeleton_feat_dim = self.skeleton_enc.out_channels
        self.skeleton_enc.fc = nn.Identity()
        self.imu_enc = IMUEncoder(in_channels=imu_channels, feat_dim=imu_feat)
        self.classifier = nn.Linear(self.skeleton_feat_dim + imu_feat, num_classes)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0/num_classes))
        nn.init.zeros_(self.classifier.bias)

    def forward(self, skel, imu):
        skel_feat = self.skeleton_enc.get_features(skel)
        imu_feat = self.imu_enc(imu)
        return self.classifier(torch.cat([skel_feat, imu_feat], dim=1))

print('Late Fusion model defined ✓')"""))

# ═══════════════════════════════════════════════════════════
# CELL 12: Fusion-GCN heading
# ═══════════════════════════════════════════════════════════
cells.append(md("""\
### Fusion-GCN: IMU as Virtual Graph Nodes

Instead of separate encoders, we **embed IMU sensors directly into the skeleton graph** as additional nodes. Each virtual node is connected to its anatomical joint, enabling the GCN to learn cross-modal spatial relationships.

```
Standard COCO-17:  0─1─3     Fusion-21:  0─1─3
                   │                     │╲
                   5─11                  5─11─[V17] ← wrist IMU
                   │                     │
                  13─15                 13─15─[V20] ← ankle IMU
                               [V18]─0          ← waist IMU
                               [V19]─7          ← left wrist IMU
```

Each virtual node carries 3 channels = (accel_x, accel_y, accel_z), same channel dim as skeleton (x, y, score)."""))

# ═══════════════════════════════════════════════════════════
# CELL 13: Fusion-GCN graph + model
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
# ═══ Fusion-GCN: Extended graph with virtual IMU nodes ═══

# Virtual node indices (appended after COCO-17 joints 0-16)
VIRTUAL_NODES = {
    17: 11,  # right_wrist IMU → connected to joint 11
    18: 0,   # waist IMU → connected to joint 0
    19: 7,   # left_wrist IMU → connected to joint 7
    20: 15,  # right_ankle IMU → connected to joint 15
}
FUSION_NUM_JOINTS = 21  # 17 skeleton + 4 virtual

# Build extended adjacency
FUSION_EDGES = list(COCO_INWARD) + [(vn, sj) for vn, sj in VIRTUAL_NODES.items()]

def get_fusion_adj():
    return _build_adj(FUSION_NUM_JOINTS, FUSION_EDGES, COCO_CENTER)

# Verify
adj = get_fusion_adj()
print(f'Fusion adjacency shape: {adj.shape}')  # (3, 21, 21)
# Check virtual node connections
for vn, sj in VIRTUAL_NODES.items():
    connected = (adj[1, vn, :].sum() + adj[2, vn, :].sum()) > 0
    print(f'  Virtual node {vn} → Joint {sj}: connected={connected}')

def build_fusion_gcn(num_classes=10, in_channels=3, **kw):
    \"\"\"STGCN++ with 21-node fusion graph.\"\"\"
    return STGCN(num_classes=num_classes, in_channels=in_channels,
                 num_joints=FUSION_NUM_JOINTS, adj_fn=get_fusion_adj, **kw)

_fg = build_fusion_gcn(NUM_CLASSES, dropout=0.3, base_channels=64, num_stages=6, num_person=2)
print(f'\\nFusion-GCN params: {count_params(_fg):,}')
_sk = build_model(NUM_CLASSES, dropout=0.3, base_channels=64, num_stages=6, num_person=2)
print(f'Skeleton-only params: {count_params(_sk):,}')
print(f'Extra params from 4 virtual nodes: {count_params(_fg) - count_params(_sk):,}')
del _fg, _sk"""))

# ═══════════════════════════════════════════════════════════
# CELL 14: Helpers
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
def evaluate(model, loader, device, multimodal_type=None):
    \"\"\"Evaluate model. multimodal_type: None (single input), 'late' (skel+imu), 'fusion_gcn' (merged tensor).\"\"\"
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            if multimodal_type == 'late':
                xb, imu_b, yb = batch
                xb, imu_b, yb = xb.to(device), imu_b.to(device), yb.to(device)
                logits = model(xb, imu_b)
            else:
                xb, yb = batch[0].to(device), batch[-1].to(device)
                logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / max(total, 1)

def train_model(model, train_loader, test_loader, epochs, device, multimodal_type=None, label=''):
    \"\"\"Generic training loop.\"\"\"
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[6, 8], gamma=0.1)
    crit = nn.CrossEntropyLoss()
    history = []
    for ep in range(1, epochs+1):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for batch in train_loader:
            if multimodal_type == 'late':
                xb, imu_b, yb = batch
                xb, imu_b, yb = xb.to(device), imu_b.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb, imu_b)
            else:
                xb, yb = batch[0].to(device), batch[-1].to(device)
                opt.zero_grad()
                logits = model(xb)
            loss = crit(logits, yb); loss.backward(); opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0); running_loss += loss.item() * yb.size(0)
        sched.step()
        test_acc = evaluate(model, test_loader, device, multimodal_type)
        history.append({'epoch': ep, 'train_acc': correct/total,
                        'train_loss': running_loss/total, 'test_acc': float(test_acc)})
        print(f'  [{label}] Epoch {ep:2d}/{epochs} train={correct/total:.4f} '
              f'loss={running_loss/total:.4f} test={test_acc:.4f}')
    best = max(h['test_acc'] for h in history)
    print(f'  ✓ {label} — Best: {best:.4f}')
    return history, best

print('Training helpers defined ✓')"""))

# ═══════════════════════════════════════════════════════════
# CELL 15: Build datasets heading
# ═══════════════════════════════════════════════════════════
cells.append(md("## 4. Build All Datasets"))

# ═══════════════════════════════════════════════════════════
# CELL 16: Build datasets
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
%%time
print('Building datasets for all experiments...')

train_dirs_set = set(split_info['xsub_train'])
test_dirs_set = set(split_info['xsub_val'])

# Accumulators
tr_skel, tr_imu6, tr_imu12, tr_y = [], [], [], []
te_skel, te_imu6, te_imu12, te_y = [], [], [], []

for ann_2d, ann_3d_item in medical_3d:
    skel = preprocess(ann_2d)  # (2, 100, 17, 3)
    imu6  = generate_imu_features(ann_3d_item, imu_joints=IMU_JOINTS_2)   # (100, 6)
    imu12 = generate_imu_features(ann_3d_item, imu_joints=IMU_JOINTS_4)   # (100, 12)
    label = MEDICAL_CLASS_MAP[ann_2d['label']]

    if ann_2d['frame_dir'] in train_dirs_set:
        tr_skel.append(skel); tr_imu6.append(imu6); tr_imu12.append(imu12); tr_y.append(label)
    elif ann_2d['frame_dir'] in test_dirs_set:
        te_skel.append(skel); te_imu6.append(imu6); te_imu12.append(imu12); te_y.append(label)

tr_skel = np.stack(tr_skel).astype(np.float32); te_skel = np.stack(te_skel).astype(np.float32)
tr_imu6 = np.stack(tr_imu6).astype(np.float32); te_imu6 = np.stack(te_imu6).astype(np.float32)
tr_imu12 = np.stack(tr_imu12).astype(np.float32); te_imu12 = np.stack(te_imu12).astype(np.float32)
tr_y = np.array(tr_y, dtype=np.int64); te_y = np.array(te_y, dtype=np.int64)

print(f'Train: {tr_skel.shape[0]}  Test: {te_skel.shape[0]}')
print(f'Skeleton: {tr_skel.shape}  IMU-6: {tr_imu6.shape}  IMU-12: {tr_imu12.shape}')

# ═══ Build Fusion-GCN tensors ═══
# Merge skeleton (17 joints) + IMU (4 virtual nodes) → (N, 2, 100, 21, 3)
def build_fusion_tensor(skel, imu12):
    \"\"\"Merge skeleton (N,2,100,17,3) + IMU-12 (N,100,12) → (N,2,100,21,3).\"\"\"
    N = skel.shape[0]
    # IMU-12 → (N, 100, 4, 3): 4 sensors × 3 axes
    imu_reshaped = imu12.reshape(N, 100, 4, 3)
    # Expand to 2 persons: copy IMU to both person slots (IMU is shared)
    imu_persons = np.stack([imu_reshaped, imu_reshaped], axis=1)  # (N, 2, 100, 4, 3)
    # Concatenate: (N, 2, 100, 17, 3) + (N, 2, 100, 4, 3) → (N, 2, 100, 21, 3)
    return np.concatenate([skel, imu_persons], axis=3).astype(np.float32)

tr_fusion = build_fusion_tensor(tr_skel, tr_imu12)
te_fusion = build_fusion_tensor(te_skel, te_imu12)
print(f'Fusion-GCN tensors: train={tr_fusion.shape} test={te_fusion.shape}')
assert tr_fusion.shape[1:] == (2, 100, 21, 3), f'Unexpected shape: {tr_fusion.shape}'"""))

# ═══════════════════════════════════════════════════════════
# CELL 17: Create loaders
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
# Skeleton-only
skel_train_dl = DataLoader(TensorDataset(torch.from_numpy(tr_skel), torch.from_numpy(tr_y)),
                           batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
skel_test_dl = DataLoader(TensorDataset(torch.from_numpy(te_skel), torch.from_numpy(te_y)),
                          batch_size=128)

# Late Fusion 12ch
late12_train_dl = DataLoader(
    TensorDataset(torch.from_numpy(tr_skel), torch.from_numpy(tr_imu12), torch.from_numpy(tr_y)),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
late12_test_dl = DataLoader(
    TensorDataset(torch.from_numpy(te_skel), torch.from_numpy(te_imu12), torch.from_numpy(te_y)),
    batch_size=128)

# Fusion-GCN
fgcn_train_dl = DataLoader(TensorDataset(torch.from_numpy(tr_fusion), torch.from_numpy(tr_y)),
                           batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
fgcn_test_dl = DataLoader(TensorDataset(torch.from_numpy(te_fusion), torch.from_numpy(te_y)),
                          batch_size=128)

print('All loaders ready ✓')"""))

# ═══════════════════════════════════════════════════════════
# CELL 18: Training heading
# ═══════════════════════════════════════════════════════════
cells.append(md("""\
## 5. Run Experiments

Three experiments, each 10 epochs centralized training."""))

# ═══════════════════════════════════════════════════════════
# CELL 19: Exp A — Skeleton-only
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
%%time
print('═══ Experiment A: Skeleton-only (baseline) ═══')
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
model_a = build_model(NUM_CLASSES, 3, base_channels=64, num_stages=6, num_person=2, dropout=0.3).to(DEVICE)
print(f'Params: {count_params(model_a):,}')
hist_a, best_a = train_model(model_a, skel_train_dl, skel_test_dl, QUICK_EPOCHS, DEVICE, label='Skel-only')
model_a.cpu(); torch.cuda.empty_cache()"""))

# ═══════════════════════════════════════════════════════════
# CELL 20: Exp B — Late Fusion 12ch
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
%%time
print('═══ Experiment B: Late Fusion 12ch (4 sensors) ═══')
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
model_b = LateFusionModel(NUM_CLASSES, imu_channels=12, imu_feat=128,
                           in_channels=3, base_channels=64, num_stages=6,
                           num_person=2, dropout=0.3).to(DEVICE)
print(f'Params: {count_params(model_b):,}')
hist_b, best_b = train_model(model_b, late12_train_dl, late12_test_dl, QUICK_EPOCHS, DEVICE,
                              multimodal_type='late', label='Late-12ch')
model_b.cpu(); torch.cuda.empty_cache()"""))

# ═══════════════════════════════════════════════════════════
# CELL 21: Exp C — Fusion-GCN
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
%%time
print('═══ Experiment C: Fusion-GCN (IMU as virtual graph nodes) ═══')
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
model_c = build_fusion_gcn(NUM_CLASSES, 3, base_channels=64, num_stages=6,
                            num_person=2, dropout=0.3).to(DEVICE)
print(f'Params: {count_params(model_c):,}')
hist_c, best_c = train_model(model_c, fgcn_train_dl, fgcn_test_dl, QUICK_EPOCHS, DEVICE, label='Fusion-GCN')
model_c.cpu(); torch.cuda.empty_cache()"""))

# ═══════════════════════════════════════════════════════════
# CELL 22: Results heading
# ═══════════════════════════════════════════════════════════
cells.append(md("## 6. Results Comparison"))

# ═══════════════════════════════════════════════════════════
# CELL 23: Comparison chart
# ═══════════════════════════════════════════════════════════
cells.append(code("""\
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 10})

# All results including v1
methods = ['Skel-only\\n(v1)', 'Late-6ch\\n(v1)', 'Skel-only\\n(v2)', 'Late-12ch\\n(v2)', 'Fusion-GCN\\n(v2)']
accs = [V1_SKEL_BEST, V1_LATE_BEST, best_a, best_b, best_c]
colors = ['#95a5a6', '#bdc3c7', '#3498db', '#e67e22', '#e74c3c']
hatches = ['/', '/', '', '', '']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Bar chart — all methods
bars = axes[0].bar(methods, accs, color=colors, edgecolor='k', linewidth=0.5)
for b, h in zip(bars, hatches):
    b.set_hatch(h)
for bar, acc in zip(bars, accs):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                 f'{acc:.3f}', ha='center', fontweight='bold', fontsize=9)
axes[0].set_ylabel('Best Test Accuracy')
axes[0].set_title('All Multimodal Methods (v1 hatched)')
axes[0].set_ylim(min(accs)-0.05, max(accs)+0.05)
axes[0].axhline(y=V1_SKEL_BEST, color='gray', linestyle='--', alpha=0.5, label=f'v1 skel baseline ({V1_SKEL_BEST})')
axes[0].legend(fontsize=8)

# Panel 2: Test accuracy curves — v2 experiments
for hist, lbl, clr in [(hist_a, 'Skel-only', '#3498db'),
                        (hist_b, 'Late-12ch', '#e67e22'),
                        (hist_c, 'Fusion-GCN', '#e74c3c')]:
    axes[1].plot([h['epoch'] for h in hist], [h['test_acc'] for h in hist],
                 '-o', color=clr, label=lbl, linewidth=1.5, markersize=4)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Test Accuracy')
axes[1].set_title('Test Accuracy (v2)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Panel 3: Training loss curves
for hist, lbl, clr in [(hist_a, 'Skel-only', '#3498db'),
                        (hist_b, 'Late-12ch', '#e67e22'),
                        (hist_c, 'Fusion-GCN', '#e74c3c')]:
    axes[2].plot([h['epoch'] for h in hist], [h['train_loss'] for h in hist],
                 '-o', color=clr, label=lbl, linewidth=1.5, markersize=4)
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Training Loss')
axes[2].set_title('Training Loss (v2)'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.suptitle('Multimodal v2: Extended Sensors & Fusion-GCN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/multimodal_v2_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary table
print('\\n' + '═'*65)
print(f'{\"Method\":<25s} {\"Best Acc\":>10s} {\"vs Skel\":>10s}')
print('─'*65)
for name, acc in zip(['Skel-only (v1)', 'Late-6ch (v1)',
                       'Skel-only (v2)', 'Late-12ch (v2)', 'Fusion-GCN (v2)'], accs):
    delta = acc - V1_SKEL_BEST
    print(f'{name:<25s} {acc:>10.4f} {delta:>+10.4f}')
print('═'*65)"""))

# ═══════════════════════════════════════════════════════════
# CELL 24: Per-class analysis for best model
# ═══════════════════════════════════════════════════════════
cells.append(md("## 7. Per-Class Analysis (Best Multimodal Model)"))

cells.append(code("""\
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Pick best multimodal model for per-class analysis
results_v2 = {'Skel-only': (best_a, model_a, skel_test_dl, None),
              'Late-12ch': (best_b, model_b, late12_test_dl, 'late'),
              'Fusion-GCN': (best_c, model_c, fgcn_test_dl, None)}
best_name = max(results_v2, key=lambda k: results_v2[k][0])
best_acc_val, best_model, best_dl, best_mm_type = results_v2[best_name]
print(f'Best multimodal: {best_name} ({best_acc_val:.4f})')

# Also evaluate skeleton-only for comparison
skel_preds, skel_true = [], []
model_a.to(DEVICE).eval()
with torch.no_grad():
    for xb, yb in skel_test_dl:
        xb = xb.to(DEVICE)
        skel_preds.extend(model_a(xb).argmax(1).cpu().tolist())
        skel_true.extend(yb.tolist())
model_a.cpu()

best_preds, best_true = [], []
best_model.to(DEVICE).eval()
with torch.no_grad():
    for batch in best_dl:
        if best_mm_type == 'late':
            xb, imu_b, yb = batch
            xb, imu_b = xb.to(DEVICE), imu_b.to(DEVICE)
            logits = best_model(xb, imu_b)
        else:
            xb, yb = batch[0].to(DEVICE), batch[-1]
            logits = best_model(xb)
        best_preds.extend(logits.argmax(1).cpu().tolist())
        best_true.extend(yb.tolist())
best_model.cpu(); torch.cuda.empty_cache()

print(f'\\n═══ Skeleton-only per-class ═══')
print(classification_report(skel_true, skel_preds, target_names=MEDICAL_LABELS, digits=3))
print(f'\\n═══ {best_name} per-class ═══')
print(classification_report(best_true, best_preds, target_names=MEDICAL_LABELS, digits=3))

# Confusion matrix comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, preds, true, title in [
    (axes[0], skel_preds, skel_true, 'Skeleton-only'),
    (axes[1], best_preds, best_true, best_name)]:
    cm = confusion_matrix(true, preds)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([l[:8] for l in MEDICAL_LABELS], rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels([l[:8] for l in MEDICAL_LABELS], fontsize=7)
    ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i,j], ha='center', va='center', fontsize=7,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/multimodal_v2_confusion.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ═══════════════════════════════════════════════════════════
# CELL 26: Save results
# ═══════════════════════════════════════════════════════════
cells.append(md("## 8. Save Results"))

cells.append(code("""\
results = {
    'experiment': 'multimodal_v2',
    'date': time.strftime('%Y-%m-%d %H:%M'),
    'config': {
        'seed': SEED, 'epochs': QUICK_EPOCHS, 'lr': LR,
        'batch_size': BATCH_SIZE,
    },
    'v1_reference': {
        'skeleton_only': V1_SKEL_BEST,
        'late_fusion_6ch': V1_LATE_BEST,
    },
    'results': {
        'skeleton_only': {'history': hist_a, 'best_acc': best_a},
        'late_fusion_12ch': {'history': hist_b, 'best_acc': best_b},
        'fusion_gcn': {'history': hist_c, 'best_acc': best_c},
    },
    'best_method': best_name,
}
with open('outputs/multimodal_v2_scan.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved: outputs/multimodal_v2_scan.json')

try:
    import shutil
    from google.colab import drive
    drive.mount('/content/drive')
    dst = '/content/drive/MyDrive/demo1_multimodal'
    os.makedirs(dst, exist_ok=True)
    for fn in ['outputs/multimodal_v2_scan.json', 'outputs/multimodal_v2_comparison.png',
               'outputs/multimodal_v2_confusion.png']:
        if os.path.exists(fn): shutil.copy2(fn, dst)
    print(f'Copied to Drive: {dst}')
except Exception as e:
    print(f'Drive mount skipped ({e})')"""))

# ═══════════════════════════════════════════════════════════
# CELL 28: Next steps
# ═══════════════════════════════════════════════════════════
cells.append(md("""\
## 9. Summary & Next Steps

### If Fusion-GCN improves over skeleton-only:
1. **More epochs / tuning**: Run 20-30 epochs to see full convergence
2. **Combine with FL**: Replace STGCN++ in the FL pipeline with Fusion-GCN
3. **Cross-modal attention**: Add attention between skeleton and IMU node features
4. **More virtual sensors**: Add head (joint 3), elbows (joints 9, 5)

### If Fusion-GCN ≈ skeleton-only (redundancy dominates):
1. Synthetic IMU derived from same skeleton → inherently redundant
2. **Real IMU is the key differentiator** — different sensor physics captures distinct information
3. **UTD-MHAD dataset**: 27 actions, has real IMU + skeleton (Kinect v1)
4. **Focus research effort on FL methods** (other notebook) as primary contribution

### Fundamental insight
> Synthetic IMU = f(skeleton) → information is a strict subset of skeleton.
> Real IMU captures vibrations, drift, and noise patterns absent from optical tracking.
> Graph fusion architecture is sound — test with real multi-sensor data next."""))

# ═══════════════════════════════════════════════════════════
# ASSEMBLE
# ═══════════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4, "nbformat_minor": 2
}
with open(TARGET, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f'Created: {TARGET}')
print(f'Cells: {len(cells)}')
