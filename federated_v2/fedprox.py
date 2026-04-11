import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 1. SETUP
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. DATA PREPROCESSING
def load_and_preprocess(file_list):
    dfs = [pd.read_parquet(f) for f in file_list if os.path.exists(f)]
    df = pd.concat(dfs, ignore_index=True)
    
    # Clean data to prevent NaNs
    features = ['HR', 'Pulse', 'SpO2', 'etCO2', 'awRR']
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['class_name'])
    
    X = df[features].values
    le = LabelEncoder()
    y = le.fit_transform(df['class_name'])
    
    # RobustScaler is better for medical vitals with outliers
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    return X, y, len(le.classes_), le

# 3. MODEL DEFINITION (Added BatchNorm for stability)
class HealthModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HealthModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        logits = self.fc(x)
        probs = torch.softmax(logits, dim=1)
        return logits, probs

# 4. CENTRALIZED TRAINING
def train_centralized(X, y, n_classes, epochs=100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_ds = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    model = HealthModel(X.shape[1], n_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history = {'acc': []}
    for epoch in range(epochs):
        model.train()
        correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Safety clip
            optimizer.step()
            correct += (logits.argmax(1) == target).sum().item()
        history['acc'].append(correct / len(X_train))
    
    torch.save(model.state_dict(), f"{MODEL_DIR}/centralized_model.pth")
    return history

# 5. FEDPROX CLIENT
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, mu=0.1):
        self.model = model
        self.train_loader = train_loader
        self.mu = mu # FedProx proximal term

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict({k: torch.tensor(v) for k, v in params_dict}, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_params = [p.detach().clone() for p in self.model.parameters()]
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for _ in range(3): # Local epochs
            for data, target in self.train_loader:
                optimizer.zero_grad()
                logits, _ = self.model(data)
                loss = nn.CrossEntropyLoss()(logits, target)
                
                # Proximal term calculation
                prox_term = 0.0
                for local_p, global_p in zip(self.model.parameters(), global_params):
                    prox_term += (local_p - global_p).pow(2).sum()
                
                (loss + (self.mu / 2) * prox_term).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.train_loader:
                logits, _ = self.model(data)
                total += target.size(0)
                correct += (logits.argmax(1) == target).sum().item()
        return float(correct/total), total, {"accuracy": float(correct/total)}

# 6. SIMULATION
def run_simulation(X, y, n_classes, n_clients):
    X_split, y_split = np.array_split(X, n_clients), np.array_split(y, n_clients)
    
    def client_fn(cid):
        idx = int(cid)
        ds = TensorDataset(torch.Tensor(X_split[idx]), torch.LongTensor(y_split[idx]))
        return FlowerClient(HealthModel(X.shape[1], n_classes), DataLoader(ds, batch_size=32))

    strategy = fl.server.strategy.FedProx(
        proximal_mu=0.1,
        fraction_fit=0.4, # Train 40% of clients per round for better generalization
        min_fit_clients=max(2, int(n_clients * 0.4)),
        min_available_clients=n_clients,
        evaluate_metrics_aggregation_fn=lambda m: {"accuracy": np.mean([x[1]["accuracy"] for x in m])}
    )

    return fl.simulation.start_simulation(client_fn=client_fn, num_clients=n_clients, 
                                         config=fl.server.ServerConfig(num_rounds=100), strategy=strategy)

# --- RUN ---
files = ['/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0050_0100_ntu_actions.parquet', '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0100_0150_ntu_actions.parquet', 
         '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0150_0200_ntu_actions.parquet', '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0200_0250_ntu_actions.parquet', 
         '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0250_0300_ntu_actions.parquet']

X, y, n_classes, encoder = load_and_preprocess(files)

cent_hist = train_centralized(X, y, n_classes, 100)
fed_10 = run_simulation(X, y, n_classes, 10)
fed_20 = run_simulation(X, y, n_classes, 20)
fed_50 = run_simulation(X, y, n_classes, 50)

# --- PLOTTING ---
plt.figure(figsize=(10, 6))
plt.plot(cent_hist['acc'], label='Centralized (Baseline)', linewidth=2)
plt.plot([x[1] for x in fed_10.metrics_distributed['accuracy']], label='FedProx (10 Clients)')
plt.plot([x[1] for x in fed_20.metrics_distributed['accuracy']], label='FedProx (20 Clients)')
plt.plot([x[1] for x in fed_50.metrics_distributed['accuracy']], label='FedProx (50 Clients)')
plt.title('Accuracy Improvement with FedProx')
plt.xlabel('Rounds/Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.show()