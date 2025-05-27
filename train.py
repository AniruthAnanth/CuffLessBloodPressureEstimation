import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchdiffeq import odeint
from scipy.signal import find_peaks
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import argparse
import random
import json

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

INPUT_DIR = 'data/mat/'

# Data loading and preprocessing functions
def load_waveforms(filename):
    """
    Load raw PPG, ABP, ECG signals from a .mat file.
    """
    name = os.path.splitext(filename)[0]
    path = os.path.join(INPUT_DIR, filename)
    mat = h5py.File(path, 'r')
    refs = mat[name]              # group containing records
    element_ref = refs[0][0]      # first record reference
    data = np.array(mat[element_ref]).T
    mat.close()
    ppg, abp, ecg = data[0, :], data[1, :], data[2, :]
    return ppg, abp, ecg

# Detect R-peaks in ECG (simple Pan-Tompkins approximation)
def detect_r_peaks(ecg, fs=125, distance_sec=0.6):
    distance = int(distance_sec * fs)
    threshold = np.mean(ecg) + 0.5 * np.std(ecg)
    peaks, _ = find_peaks(ecg, distance=distance, height=threshold)
    return peaks

# Segment PPG, ABP, ECG into beats around each R-peak
def segment_beats(signals, r_peaks, fs=125, pre_sec=0.2, post_sec=0.4):
    pre = int(pre_sec * fs)
    post = int(post_sec * fs)
    ppg, abp, ecg = signals
    ppg_beats, abp_beats, ecg_beats = [], [], []
    for r in r_peaks:
        if r - pre >= 0 and r + post <= len(ppg):
            ppg_beats.append(ppg[r-pre:r+post])
            abp_beats.append(abp[r-pre:r+post])
            ecg_beats.append(ecg[r-pre:r+post])
    return np.array(ppg_beats), np.array(abp_beats), np.array(ecg_beats)

# Normalize each beat (zero-mean, unit-variance)
def normalize_beats(beats):
    return np.array([(b - np.mean(b)) / np.std(b) for b in beats])

# Compute systolic (max) and diastolic (min) BP for each ABP-beat
def compute_bp_features(abp_beats):
    systolic = np.max(abp_beats, axis=1)
    diastolic = np.min(abp_beats, axis=1)
    return systolic, diastolic

# Dataset class
class BPDataset(Dataset):
    def __init__(self, X, Y_sys, Y_dia):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Reshape to [batch, 2 channels, time] assuming each beat has length 75
        self.X = self.X.view(-1, 2, 75)
        # Only store diastolic BP
        self.Y = torch.tensor(Y_dia, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# New Model Components for Latent ODE Modeling

# Encoder: maps beat input to latent state z0
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        # Remove CNN, add more recurrent layers
        self.rnn1 = nn.LSTM(input_size=2, hidden_size=64, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        # x shape: [batch, 2, time]
        x = x.transpose(1, 2)  # [batch, time, 2]
        h1, (hn1, _) = self.rnn1(x)
        h2, (hn2, _) = self.rnn2(h1)
        hn2 = hn2.squeeze(0)
        return self.fc(hn2)

# Latent ODE Function integrating physics (parameterized Windkessel)
class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        # Learnable Windkessel parameters (baseline values)
        self.log_R = nn.Parameter(torch.log(torch.tensor(1.2)))
        self.log_C = nn.Parameter(torch.log(torch.tensor(1.5)))
        # Additional NN to model characteristic impedance/time-varying compliance
        self.comp_net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
    def forward(self, t, z):
        # Compute dynamic compliance as a function of latent state
        comp = self.comp_net(z)
        # Transform learnable parameters
        R = torch.exp(self.log_R)
        C = torch.exp(self.log_C)
        # ODE: dz/dt = ( - z / R + comp )/ C
        dzdt = (-z / R + comp) / C
        return dzdt

# Decoder: reconstruct BP from latent state
class Decoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1)  # Only diastolic BP
        )
    def forward(self, z):
        return self.net(z)

# Full BP Model using latent ODE
class BPModel(nn.Module):
    def __init__(self, input_dim, latent_dim=64, use_ode=True):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.use_ode = use_ode
        # Increase hidden dimension in ODE or keep as is
        self.odefunc = LatentODEFunc(latent_dim) if use_ode else None
        self.decoder = Decoder(latent_dim)
    def forward(self, x):
        # Encode input beats into latent state
        z0 = self.encoder(x)
        if self.use_ode:
            # evolve latent ODE over time; here we use [0,1]
            ts = torch.tensor([0.0, 1.0], device=x.device)
            zt = odeint(self.odefunc, z0, ts)
            zT = zt[-1]
        else:
            # If ODE is not used, bypass evolution
            zT = z0
        # Decode latent state to BP predictions
        return self.decoder(zT)

# EarlyStopping class (unchanged)
class EarlyStopping:
    def __init__(self, patience=7, delta=1e-3):
        self.patience, self.delta = patience, delta
        self.best_loss = float('inf'); self.counter = 0; self.stop = False
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.stop = True

def evaluate_metrics(preds, targets):
    # Compute MSE, MAE, and Pearson correlation for diastolic BP only
    mse = nn.MSELoss()(preds, targets).item()
    mae = nn.L1Loss()(preds, targets).item()
    preds_np = preds.detach().cpu().numpy().ravel()
    targets_np = targets.detach().cpu().numpy().ravel()
    corr = pearsonr(preds_np, targets_np)[0]
    return mse, mae, corr

import numpy as np

def evaluate_bhs(differences_mmHg):
    differences_mmHg = np.abs(differences_mmHg)
    n = len(differences_mmHg)
    pct_5 = np.sum(differences_mmHg <= 5) / n * 100
    pct_10 = np.sum(differences_mmHg <= 10) / n * 100
    pct_15 = np.sum(differences_mmHg <= 15) / n * 100
    if pct_5 >= 60 and pct_10 >= 85 and pct_15 >= 95:
        grade = 'A'
    elif pct_5 >= 50 and pct_10 >= 75 and pct_15 >= 90:
        grade = 'B'
    elif pct_5 >= 40 and pct_10 >= 65 and pct_15 >= 85:
        grade = 'C'
    else:
        grade = 'D'
    return {
        'grade': grade,
        'percent_within_5mmHg': pct_5,
        'percent_within_10mmHg': pct_10,
        'percent_within_15mmHg': pct_15
    }

def evaluate_aami(differences_mmHg):
    differences_mmHg = np.array(differences_mmHg)
    mean_error = np.mean(differences_mmHg)
    std_dev = np.std(differences_mmHg, ddof=1)
    pass_fail = 'Pass' if abs(mean_error) <= 5 and std_dev <= 8 else 'Fail'
    return {
        'pass_fail': pass_fail,
        'mean_error_mmHg': mean_error,
        'std_dev_mmHg': std_dev
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ode', action='store_true', help='Toggle latent ODE component')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study using only soft penalties')
    args = parser.parse_args()

    # Save configuration for reproducibility
    config = vars(args)
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Load and preprocess data
    files = os.listdir(INPUT_DIR)
    all_ppg, all_ecg = [], []
    all_sys, all_dia = [], []
    for fn in files:
        ppg, abp, ecg = load_waveforms(fn)
        r_peaks = detect_r_peaks(ecg, fs=125)
        ppg_b, abp_b, ecg_b = segment_beats((ppg, abp, ecg), r_peaks, fs=125)
        ppg_n = normalize_beats(ppg_b)
        ecg_n = normalize_beats(ecg_b)
        sys_vals, dia_vals = compute_bp_features(abp_b)
        all_ppg.append(ppg_n)
        all_ecg.append(ecg_n)
        all_sys.extend(sys_vals)
        all_dia.extend(dia_vals)
    X_ppg = np.vstack(all_ppg)
    X_ecg = np.vstack(all_ecg)
    X = np.hstack([X_ppg, X_ecg])
    Y_sys = np.array(all_sys)
    Y_dia = np.array(all_dia)

    dataset = BPDataset(X, Y_sys, Y_dia)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold = 0
    all_metrics = []
    all_sys_err = []
    all_dia_err = []
    for train_index, test_index in kf.split(dataset):
        fold += 1
        print(f"\nFold {fold}")
        train_subset = Subset(dataset, train_index)
        test_subset = Subset(dataset, test_index)
        # Further split train into train and validation (80-20)
        n_train = len(train_subset)
        indices = list(range(n_train))
        split_at = int(0.8 * n_train)
        train_indices, val_indices = indices[:split_at], indices[split_at:]
        train_ds = Subset(train_subset, train_indices)
        val_ds = Subset(train_subset, val_indices)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=32)
        test_loader  = DataLoader(test_subset, batch_size=32)

        # Build model
        input_dim = X.shape[1]
        model = BPModel(input_dim, latent_dim=16, use_ode=args.use_ode and not args.ablation).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        criterion = nn.MSELoss()
        early = EarlyStopping(patience=10)
        phys_weight = 100    # weight for physiological penalty

        # Training loop for this fold
        epochs = 100
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                # Soft physiological constraints (diastolic only)
                dia_p = preds[:,0]
                c4 = torch.relu(40 - dia_p).mean()                        # diastolic lower bound
                c5 = torch.relu(dia_p - 120).mean()                       # diastolic upper bound
                phys_pen = c4 + c5
                total_loss = loss + phys_weight * phys_pen
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)
            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            early(val_loss)
            if early.stop:
                print("Early stopping triggered.")
                break

        # Testing on fold
        model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                test_loss += criterion(preds, yb).item() * xb.size(0)
                all_preds.append(preds)
                all_targets.append(yb)
        test_loss /= len(test_loader.dataset)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mse, mae, corr = evaluate_metrics(all_preds, all_targets)
        print(f"Fold {fold} Test Loss: {test_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, Corr: {corr:.4f}")
        all_metrics.append({'fold': fold, 'test_loss': test_loss, 'mse': mse, 'mae': mae, 'corr': corr})

        preds_np = all_preds.detach().cpu().numpy()
        targets_np = all_targets.detach().cpu().numpy()
        # Evaluate BHS/AAMI only for diastolic BP
        dif_dia = preds_np - targets_np
        bhs_dia = evaluate_bhs(dif_dia)
        aami_dia = evaluate_aami(dif_dia)
        print(f"BHS DIA (fold {fold}): {bhs_dia}")
        print(f"AAMI DIA (fold {fold}): {aami_dia}")
        all_dia_err.extend(dif_dia.tolist())

    # Reporting average metrics across folds
    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_mae = np.mean([m['mae'] for m in all_metrics])
    avg_corr = np.mean([m['corr'] for m in all_metrics])
    print("\nFinal Report:")
    print(f"Average MSE: {avg_mse:.4f}, Average MAE: {avg_mae:.4f}, Average Correlation: {avg_corr:.4f}")

    # Print final results only for diastolic BP
    bhs_dia_final = evaluate_bhs(all_dia_err)
    aami_dia_final = evaluate_aami(all_dia_err)
    print("Final BHS DIA:", bhs_dia_final)
    print("Final AAMI DIA:", aami_dia_final)

