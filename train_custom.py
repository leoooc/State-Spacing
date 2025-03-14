import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=20):
        self.sequence_length = sequence_length
        self.data = data
        self.X, self.y = self.create_sequences(data, sequence_length)
    
    def create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return (torch.tensor(self.X[index], dtype=torch.float32),
                torch.tensor(self.y[index], dtype=torch.float32))


# MambaLayer Definition

class MambaLayer(nn.Module):
    def __init__(self, input_size, hidden_size, activation=nn.Tanh()):
        """
        A custom state-space layer (MambaLayer) that learns state dynamics.
        
        Args:
            input_size (int): Number of features per time step.
            hidden_size (int): Dimension of the hidden state.
            activation (nn.Module): Nonlinearity (default: Tanh).
        """
        super(MambaLayer, self).__init__()
        self.hidden_size = hidden_size
        self.A = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.B = nn.Parameter(torch.randn(input_size, hidden_size))     
        self.initial_state = nn.Parameter(torch.zeros(hidden_size))
        self.activation = activation

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch, sequence_length, input_size)
        Returns:
            Tensor: Final state, shape (batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
       
        state = self.initial_state.unsqueeze(0).expand(batch_size, self.hidden_size)
    
        for t in range(seq_len):
            xt = x[:, t, :]  
            state = self.activation(state @ self.A + xt @ self.B)
        return state


class MambaModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, dropout=0.1):
        """
        A multi-layer state-space model using MambaLayer.
        
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Hidden dimension used in MambaLayer.
            output_size (int): Number of output features.
            num_layers (int): Number of stacked MambaLayer blocks.
            dropout (float): Dropout rate between layers.
        """
        super(MambaModel, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([
            MambaLayer(input_size=hidden_size, hidden_size=hidden_size)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
       
        x = self.input_proj(x)  # Now shape: (batch, sequence_length, hidden_size)
      
        for layer in self.layers:
          
            state = layer(x)
            x = state.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = self.dropout(x)
     
        x = state  # shape: (batch, hidden_size)
        return self.output_proj(x)


def main():
    df = pd.read_csv("preprocessed_airquality.csv", index_col=0)
    data = df.values  # shape: (num_time_steps, num_features)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    train_val_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, shuffle=False)

    sequence_length = 20  # Adjust as needed.
    train_dataset = TimeSeriesDataset(train_data, sequence_length)
    val_dataset = TimeSeriesDataset(val_data, sequence_length)
    test_dataset = TimeSeriesDataset(test_data, sequence_length)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    input_size = data.shape[1]    # number of features
    hidden_size = 128             # hidden dimension for MambaLayer
    output_size = data.shape[1]   # predicting the full feature vector
    num_layers = 4
    model = MambaModel(input_size, hidden_size, output_size, num_layers=num_layers, dropout=0.1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
