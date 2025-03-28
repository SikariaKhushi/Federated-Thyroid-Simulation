import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load Thyroid Dataset (Simulated Example)
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhyper.data", delim_whitespace=True, header=None)
data = data.dropna()

# Splitting dataset
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]  # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a simple Deep Neural Network for classification
class ThyroidDNN(nn.Module):
    def __init__(self, input_size):
        super(ThyroidDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Initialize Model
dnn_model = ThyroidDNN(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)

# Federated Learning Client
class ThyroidClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train, self.y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
        self.X_test, self.y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(3):  # Local Training
            optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
        return self.get_parameters(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_test)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = accuracy_score(self.y_test, predictions)
        return accuracy, len(self.X_test), {}

# Start Federated Learning Simulation
fl.client.start_numpy_client("localhost:8080", client=ThyroidClient(dnn_model, X_train, y_train, X_test, y_test))
