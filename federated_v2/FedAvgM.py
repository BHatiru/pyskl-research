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

# 1. DIRECTORIES
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. DATA PREPROCESSING
def load_and_preprocess(file_list):
    dfs = [pd.read_parquet(f) for f in file_list if os.path.exists(f)]
    df = pd.concat(dfs, ignore_index=True)
    
    features = ['HR', 'Pulse', 'SpO2', 'etCO2', 'awRR']
    # Clean and handle outliers with RobustScaler
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ['class_name'])
    
    X = RobustScaler().fit_transform(df[features].values)
    le = LabelEncoder()
    y = le.fit_transform(df['class_name'])
    
    # Anomaly Column for logic (0=Normal, 1=Anomaly)
    df['is_anomaly'] = df['class_name'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)
    
    return X, y, len(le.classes_), le

# 3. DEEPER MODEL FOR COMPLEX PATTERNS
class HealthModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HealthModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=1)
        return logits, probs

# 4. CENTRALIZED TRAINING (The Baseline)
def train_centralized(X, y, n_classes, epochs=100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    model = HealthModel(X.shape[1], n_classes)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            correct += (logits.argmax(1) == target).sum().item()
        history['acc'].append(correct / len(X_train))
    
    torch.save(model.state_dict(), f"{MODEL_DIR}/centralized_model.pth")
    return history

# 5. OPTIMIZED FL CLIENT
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict({k: torch.tensor(v) for k, v in params_dict}, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config.get("lr", 0.001) # Get decayed LR from server
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        for _ in range(3): # More local epochs to find local optima
            for data, target in self.train_loader:
                logits, _ = self.model(data)
                loss = nn.CrossEntropyLoss()(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
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

# 6. SIMULATION (FedAvgM + Adaptive Learning Rate)
def run_simulation(X, y, n_classes, n_clients):
    X_split, y_split = np.array_split(X, n_clients), np.array_split(y, n_clients)
    
    # --- ADD THIS: Initialize parameters for the server ---
    initial_model = HealthModel(X.shape[1], n_classes)
    # Convert model parameters to list of NumPy arrays
    ndarrays = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    # Convert NumPy arrays to Flower Parameters object
    initial_parameters = fl.common.ndarrays_to_parameters(ndarrays)
    # ------------------------------------------------------

    def client_fn(cid):
        ds = TensorDataset(torch.Tensor(X_split[int(cid)]), torch.LongTensor(y_split[int(cid)]))
        return FlowerClient(HealthModel(X.shape[1], n_classes), DataLoader(ds, batch_size=32), cid)

    def on_fit_config_fn(server_round: int):
        lr = 0.001 * (0.95 ** (server_round // 10))
        return {"lr": max(lr, 1e-5)}

    strategy = fl.server.strategy.FedAvgM(
        server_learning_rate=1.0,
        server_momentum=0.9,
        initial_parameters=initial_parameters, # <--- PASS INITIAL PARAMETERS HERE
        fraction_fit=0.3,
        min_fit_clients=3,
        min_available_clients=n_clients,
        on_fit_config_fn=on_fit_config_fn,
        evaluate_metrics_aggregation_fn=lambda m: {"accuracy": np.mean([x[1]["accuracy"] for x in m])}
    )

    return fl.simulation.start_simulation(
        client_fn=client_fn, 
        num_clients=n_clients, 
        config=fl.server.ServerConfig(num_rounds=100), 
        strategy=strategy
    )

# --- EXECUTION ---
files = ['/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0050_0100_ntu_actions.parquet', '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0100_0150_ntu_actions.parquet', 
         '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0150_0200_ntu_actions.parquet', '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0200_0250_ntu_actions.parquet', 
         '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0250_0300_ntu_actions.parquet']

X, y, n_classes, encoder = load_and_preprocess(files)

# Training all scenarios
cent_hist = train_centralized(X, y, n_classes, 100)
fed_10 = run_simulation(X, y, n_classes, 10)
fed_20 = run_simulation(X, y, n_classes, 20)
fed_50 = run_simulation(X, y, n_classes, 50)

# --- PLOTTING ---
plt.figure(figsize=(10, 6))
plt.plot(cent_hist['acc'], label='Centralized (Target)', linewidth=3)
plt.plot([x[1] for x in fed_10.metrics_distributed['accuracy']], label='SOTA Fed-10')
plt.plot([x[1] for x in fed_20.metrics_distributed['accuracy']], label='SOTA Fed-20')
plt.plot([x[1] for x in fed_50.metrics_distributed['accuracy']], label='SOTA Fed-50')
plt.title('Comparison: Centralized vs Optimized Federated Approaches')
plt.xlabel('Rounds / Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.show()