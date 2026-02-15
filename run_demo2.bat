@echo off
REM ==========================================================================
REM  run_demo2.bat — Real-time skeleton-based action recognition demo
REM  Repository : pyskl
REM  Model      : STGCN++ (Joint modality, NTU-120 XSub, HRNet 2D skeletons)
REM  Device     : GPU (CUDA) — falls back to CPU automatically if no GPU
REM ==========================================================================

REM ---------- 0. Config (edit these if needed) ----------
SET PYSKL_ROOT=%~dp0
SET CONDA_ENV=pyskl
SET DEVICE=cuda:0

SET ACTION_CONFIG=configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py
SET ACTION_CKPT=http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
SET LABEL_MAP=tools/data/label_map/nturgbd_120.txt

SET DET_CONFIG=demo/faster_rcnn_r50_fpn_1x_coco-person.py
SET DET_CKPT=https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth
SET POSE_CONFIG=demo/hrnet_w32_coco_256x192.py
SET POSE_CKPT=https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth

REM ---------- 1. Environment setup (run once) ----------
REM conda env create -f pyskl.yaml          &:: Python 3.7
REM   — OR —
REM conda env create -f pyskl_310.yaml      &:: Python 3.10
REM conda activate %CONDA_ENV%
REM pip install -e .
REM
REM Verify:
REM   python -c "import pyskl; print('pyskl OK')"
REM   python -c "import mmdet; print('mmdet OK')"
REM   python -c "import mmpose; print('mmpose OK')"
REM   python -c "import torch; print('CUDA:', torch.cuda.is_available())"

REM ---------- 2. (Optional) Pre-download checkpoints ----------
REM The scripts auto-download on first run, but you can pre-cache:
REM   mkdir checkpoints
REM   curl -L -o checkpoints/stgcnpp_ntu120_xsub_hrnet_j.pth %ACTION_CKPT%
REM   curl -L -o checkpoints/faster_rcnn_r50_fpn_1x_coco-person.pth %DET_CKPT%
REM   curl -L -o checkpoints/hrnet_w32_coco_256x192.pth %POSE_CKPT%
REM Then point the --checkpoint / --det-checkpoint / --pose-checkpoint args
REM at the local paths above.

cd /d %PYSKL_ROOT%

REM ---------- 3a. STREAM MODE (default) — live webcam + sliding window ----------
REM Action label updates every ~3-5s, webcam feed stays smooth.
echo.
echo ===== STREAM MODE (webcam) =====
echo Press ESC in the OpenCV window to quit.
echo.
python demo/demo_realtime.py ^
    --mode stream ^
    --config %ACTION_CONFIG% ^
    --checkpoint %ACTION_CKPT% ^
    --det-config %DET_CONFIG% ^
    --det-checkpoint %DET_CKPT% ^
    --pose-config %POSE_CONFIG% ^
    --pose-checkpoint %POSE_CKPT% ^
    --label-map %LABEL_MAP% ^
    --device %DEVICE% ^
    --clip-len 30 ^
    --clip-stride 15 ^
    --skip-frames 2 ^
    --short-side 480
goto :eof

REM ---------- 3b. CLIP MODE — press 'r' to record 3s, then infer ----------
REM Uncomment the block below (and comment the stream block above) to use clip mode.
REM echo.
REM echo ===== CLIP MODE (webcam, press r to record) =====
REM python demo/demo_realtime.py ^
REM     --mode clip ^
REM     --config %ACTION_CONFIG% ^
REM     --checkpoint %ACTION_CKPT% ^
REM     --det-config %DET_CONFIG% ^
REM     --det-checkpoint %DET_CKPT% ^
REM     --pose-config %POSE_CONFIG% ^
REM     --pose-checkpoint %POSE_CKPT% ^
REM     --label-map %LABEL_MAP% ^
REM     --device %DEVICE% ^
REM     --record-seconds 3.0

REM ---------- 3c. OFFLINE — process the bundled sample video ----------
REM python demo/demo_realtime.py ^
REM     --mode stream ^
REM     --video demo/ntu_sample.avi ^
REM     --config %ACTION_CONFIG% ^
REM     --checkpoint %ACTION_CKPT% ^
REM     --device %DEVICE%

REM ---------- 3d. Original offline demo (writes output .mp4) ----------
REM python demo/demo_skeleton.py demo/ntu_sample.avi demo/demo_output.mp4 ^
REM     --config %ACTION_CONFIG% ^
REM     --checkpoint %ACTION_CKPT% ^
REM     --device %DEVICE%

REM ==========================================================================
REM Expected output (stream mode):
REM   [1/3] Loading person detector (Faster-RCNN) ...
REM   [2/3] Loading pose estimator (HRNet-w32) ...
REM   [3/3] Loading action recogniser (STGCN++) ...
REM     → 120 action classes loaded.
REM   All models loaded ✓
REM   Video source opened  (fps=30)
REM   Sliding window: 30 frames, stride 15, skip 2
REM   Press ESC to quit.
REM
REM   [Live webcam window shows]:
REM     Action: clapping
REM     1. clapping: 0.847
REM     2. hand waving: 0.053
REM     3. reading: 0.021
REM     ...
REM     [STREAM] Inference: 0.28 Hz  |  ESC=quit
REM ==========================================================================

:eof
