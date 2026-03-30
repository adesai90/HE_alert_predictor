import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class AstroLightcurveDataset(Dataset):
    """
    Predicts DELTAS instead of absolute values.

    y = x(t+1:t+H) - x(t)
    This improves stationarity and flare modeling.
    """
    def __init__(self, data, seq_len=30, forecast_horizon=5):
        print(f'Initializing AstroLightcurveDataset with seq_length {seq_len} and forecast_horizon {forecast_horizon}')
        self.seq_len = seq_len
        self.horizon = forecast_horizon
        self.data = data

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]

        last_state = self.data[idx+self.seq_len-1]
        future = self.data[idx+self.seq_len:idx+self.seq_len+self.horizon]

        # Predict DELTAS relative to last observed point
        y = future - last_state

        return torch.FloatTensor(x), torch.FloatTensor(y)
    

class AstroForecastModel(nn.Module):
    """
    Input Dimension is kept 2 (flux and index in this case)
    This undergoes linear projection to give 128  features
    Architecture:
    Input → Linear Projection
           → 2-layer LSTM
           → Multi-head Attention
           → Residual MLP
           → Predict mean + log variance
    """

    def __init__(self, input_dim=2, hidden_dim=64,
                 forecast_horizon=5, num_heads=2):
        super().__init__() # Calling init from the parent class nn.Module

        self.horizon = forecast_horizon
        self.hidden_dim = hidden_dim

        # Linear projection (helps feature mixing)
        # This allows the neural network to try different representations of the data (say for example.. flux-index, 2flux+index to help predict)
        # In more tecgnical terms, This lets the network learn interactions like: flux variability patterns; spectral hardening/softening; correlations between flux and index
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM (Long short-term memory) backbone
        self.lstm = nn.LSTM(
            hidden_dim, # Input dimension
            hidden_dim, # LSTM internal dimension state kept at 128 for balanced capacity use, change to 32-64 for small datasetts and 256+ for large
            num_layers=2, # This stacks the LSTM layers
            batch_first=True #Instead of (sequence_length, batch, features), pytorch expects (batch, sequence_length, features) eg (32 samples,30 timestamps,128 features)
        )

        # Multi-head attention (better long-term modeling)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Residual MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output: mean + logvar for flux and index
        # shape = horizon × 2 variables × (mean + logvar)
        self.output_layer = nn.Linear(
            hidden_dim,
            forecast_horizon * 2 * 2
        )

    def forward(self, x):
        # x: (batch, seq_len, 2)

        last_input = x[:, -1, :]  # Save last observed state (residual use)

        x = self.input_proj(x)

        lstm_out, _ = self.lstm(x)

        # Self attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)

        # Global context
        context = attn_out.mean(dim=1)

        context = context + self.mlp(context)  # Residual connection

        out = self.output_layer(context)

        # Reshape output
        out = out.view(
            x.size(0),
            self.horizon,
            2,  # flux + index
            2   # mean + logvar
        )

        mean = out[..., 0]
        logvar = out[..., 1]

        return mean, logvar
    
def gaussian_nll(mean, logvar, target):
    # Clamp logvar to [-6, -1] instead of [-6, 6]
    # This prevents the model from becoming overconfident
    logvar = torch.clamp(logvar, min=-6, max=-1)  
    precision = torch.exp(-logvar)
    squared_error = (target - mean) ** 2
    loss = 0.5 * (precision * squared_error + logvar + np.log(2 * np.pi))  # ← Add constant
    return loss.mean()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train_model(model, train_loader, val_loader,
                epochs=120, lr=1e-3, device='cuda',
                patience=20):
    

    if os.path.exists("best_astro_model.pth"):

        print(f"Loading existing model ")

        model.load_state_dict(torch.load("best_astro_model.pth", map_location=device))

    else:

        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr,
            weight_decay=1e-5  # L2 regularization
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        model.to(device)
        best_val = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                mean, logvar = model(x)
                loss = gaussian_nll(mean, logvar, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    mean, logvar = model(x)
                    loss = gaussian_nll(mean, logvar, y)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}")
                print(f"  Train Loss: {train_loss:.5f}")
                print(f"  Val Loss:   {val_loss:.5f}")

            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), "best_astro_model.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                print(f"  Best val loss: {best_val:.5f} at epoch {best_epoch}")
                break

        model.load_state_dict(torch.load("best_astro_model.pth"))
    return model

def forecast(model, last_sequence, flux_scaler, index_scaler,
             flux_is_log=True, device='cuda'):

    """
    Multi-step direct prediction.
    No recursive feeding → no error explosion.
    """

    model.eval()

    x = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)

    with torch.no_grad():
        mean, logvar = model(x)

    mean = mean.cpu().numpy()[0]  # (horizon, 2)
    std = np.exp(0.5 * np.clip(logvar.cpu().numpy()[0], -6, -1))

    # Add deltas to last state (residual reconstruction)
    last_state = last_sequence[-1]
    #predictions_scaled = last_state + np.cumsum(mean, axis=0)
    predictions_scaled = last_state + mean


    # Inverse scaling
    flux_predictions = flux_scaler.inverse_transform(predictions_scaled)
    flux_lower = flux_scaler.inverse_transform(predictions_scaled - std)
    flux_upper = flux_scaler.inverse_transform(predictions_scaled + std)

    index_predictions = index_scaler.inverse_transform(predictions_scaled)
    index_lower = index_scaler.inverse_transform(predictions_scaled - std)
    index_upper = index_scaler.inverse_transform(predictions_scaled + std)

    flux_pred = flux_predictions[:, 0]
    index_pred = index_predictions[:, 1]
    flux_lower = flux_lower[:, 0]
    flux_upper = flux_upper[:, 0]
    index_lower = index_lower[:, 1]
    index_upper = index_upper[:, 1]

    if flux_is_log:
        flux_pred = 10**flux_pred
        flux_lower = 10**flux_lower
        flux_upper = 10**flux_upper

    return {
        'flux': flux_pred,
        'flux_lower': flux_lower,
        'flux_upper': flux_upper,
        'index': index_pred,
        'index_lower': index_lower,
        'index_upper': index_upper,
        'std': std
    }
