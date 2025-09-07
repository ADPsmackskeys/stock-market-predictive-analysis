import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.baseline_unimodal import BaseModel
import numpy as np
import pandas as pd
class LSTMModel(BaseModel):
    def __init__(self, ticker, data_dir, reports_dir, seq_len=10, hidden_size=50, num_layers=1, epochs=50, batch_size=32, lr=0.001):
        super().__init__(ticker, data_dir, reports_dir)
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        input_size = None  

        class LSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                out = self.fc(out)
                out = self.sigmoid(out)
                return out

        self.model_net_class = LSTMNet
        self.model = None

    def _create_sequences(self, X, y):
        if isinstance(y, pd.Series) and (y.dtype == object or y.dtype == str):
            label_map = {"Decrease": 0, "Increase": 1}
            y = y.map(label_map)
        elif isinstance(y, np.ndarray):
            y = y.astype(np.float32)
        else:
            y = np.array(y, dtype=np.float32)

        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values.astype(np.float32)

        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_len):
            X_seq.append(X[i:i+self.seq_len])
            y_seq.append(y[i+self.seq_len])
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)
        return X_seq, y_seq


    def train(self, X_train, y_train):
        input_size = X_train.shape[1]
        self.model = self.model_net_class(input_size, self.hidden_size, self.num_layers).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X_seq, y_seq = self._create_sequences(X_train, y_train)
        dataset = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

    def predict(self, X_test):
        self.model.eval()
        dummy_y = np.zeros(len(X_test), dtype=np.float32)
        X_seq, _ = self._create_sequences(X_test, dummy_y)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.model(X_tensor)
        preds = (preds.cpu().numpy() > 0.5).astype(int).ravel()

        label_map = {0: "Decrease", 1: "Increase"}
        preds_str = np.array([label_map[i] for i in preds])
        return preds_str
