import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 1. SETUP & DIRECTORIES
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. DATA PREPROCESSING (With NaN Handling)
def load_and_preprocess(file_list):
    dfs = []
    for f in file_list:
        if os.path.exists(f):
            dfs.append(pd.read_parquet(f))
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Selecting core vitals
    features = ['HR', 'Pulse', 'SpO2', 'etCO2', 'awRR']
    
    # CRITICAL: Remove rows with NaNs or infinite values to prevent Loss: nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=features + ['class_name'])
    
    X = df[features].values
    
    # Anomaly Logic: 0 if Normal, 1 otherwise
    df['is_anomaly'] = df['class_name'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)
    
    le = LabelEncoder()
    y = le.fit_transform(df['class_name'])
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, len(le.classes_), le

# 3. MODEL DEFINITION
class HealthModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HealthModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
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
def train_centralized(X, y, num_classes, epochs=100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    model = HealthModel(X.shape[1], num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'loss': [], 'acc': []}
    
    print("\n--- Starting Centralized Training ---")
    for epoch in range(epochs):
        model.train()
        running_loss, correct = 0.0, 0
        for data, target in train_loader:
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, target)
            loss.backward()
            
            # Gradient Clipping to prevent NaNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            correct += (logits.argmax(1) == target).sum().item()
        
        history['loss'].append(running_loss / len(train_loader))
        history['acc'].append(correct / len(X_train))
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss {history['loss'][-1]:.4f}, Acc {history['acc'][-1]:.4f}")
    
    torch.save(model.state_dict(), f"{MODEL_DIR}/centralized_model.pth")
    return history

# 5. FL CLIENT
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        for _ in range(2): # Local epochs
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

# 6. SIMULATION LOGIC
def run_simulation(X, y, num_classes, num_clients, num_rounds):
    X_split, y_split = np.array_split(X, num_clients), np.array_split(y, num_clients)
    
    def client_fn(cid: str):
        idx = int(cid)
        ds = TensorDataset(torch.Tensor(X_split[idx]), torch.LongTensor(y_split[idx]))
        return FlowerClient(HealthModel(X.shape[1], num_classes), DataLoader(ds, batch_size=32))

    class SaveStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            agg = super().aggregate_fit(server_round, results, failures)
            if agg is not None and server_round == num_rounds:
                weights = fl.common.parameters_to_ndarrays(agg[0])
                m = HealthModel(X.shape[1], num_classes)
                params_dict = zip(m.state_dict().keys(), weights)
                m.load_state_dict({k: torch.tensor(v) for k, v in params_dict})
                torch.save(m.state_dict(), f"{MODEL_DIR}/fed_model_{num_clients}_clients.pth")
            return agg

    strategy = SaveStrategy(min_fit_clients=num_clients, min_available_clients=num_clients,
                            evaluate_metrics_aggregation_fn=lambda metrics: {"accuracy": np.mean([m[1]["accuracy"] for m in metrics])})

    return fl.simulation.start_simulation(client_fn=client_fn, num_clients=num_clients, 
                                         config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strategy)

# --- EXECUTION ---
files = ['/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0050_0100_ntu_actions.parquet', '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0100_0150_ntu_actions.parquet', 
         '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0150_0200_ntu_actions.parquet', '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0200_0250_ntu_actions.parquet', 
         '/home/temirlan/Downloads/integrated_v2/integrated_v2/synthetic_chunk_0250_0300_ntu_actions.parquet']

X, y, n_classes, encoder = load_and_preprocess(files)

cent_hist = train_centralized(X, y, n_classes, 100)
fed_10 = run_simulation(X, y, n_classes, 10, 100)
fed_20 = run_simulation(X, y, n_classes, 20, 100)
fed_50 = run_simulation(X, y, n_classes, 50, 100)

# --- GRAPHING ---
plt.figure(figsize=(10, 5))
plt.plot(cent_hist['acc'], label='Centralized')
plt.plot([x[1] for x in fed_10.metrics_distributed['accuracy']], label='Fed-10')
plt.plot([x[1] for x in fed_20.metrics_distributed['accuracy']], label='Fed-20')
plt.plot([x[1] for x in fed_50.metrics_distributed['accuracy']], label='Fed-50')
plt.title('Accuracy vs Rounds/Epochs')
plt.legend(); plt.show()