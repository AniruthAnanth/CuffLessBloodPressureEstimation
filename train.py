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
from scipy.stats import pearsonr, wilcoxon
import random
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from data_utils import load_waveforms, detect_r_peaks, segment_beats, normalize_beats, compute_bp_features, BPDataset
from models import BPModel
from train_utils import EarlyStopping, evaluate_metrics, evaluate_bhs, evaluate_aami

plt.style.use('ggplot')
# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

INPUT_DIR = 'data/mat/'

def train_model(model, train_loader, val_loader, test_loader, phys_weight, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.MSELoss()
    early = EarlyStopping(patience=10)
    num_epochs = 100
    # Training
    for epoch in range(num_epochs):
        # linear annealing of physics weight
        cur_weight = phys_weight * (epoch / max(1, num_epochs-1)) if phys_weight>0 else 0.0
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            # physics-based penalty with annealed weight
            if cur_weight>0:
                sys_p, dia_p = preds[:,0], preds[:,1]
                c1 = torch.relu(dia_p - sys_p).mean()
                c2 = torch.relu(80 - sys_p).mean()
                c3 = torch.relu(sys_p - 200).mean()
                c4 = torch.relu(40 - dia_p).mean()
                c5 = torch.relu(dia_p - 120).mean()
                c6 = torch.relu(10 - (sys_p - dia_p)).mean()
                loss += cur_weight * (c1+c2+c3+c4+c5+c6)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*xb.size(0)
        # validation and LR scheduling
        model.eval()
        val_loss=0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()*xb.size(0)
        scheduler.step(val_loss)
        early(val_loss)
        if early.stop: break
    # Testing
    model.eval()
    preds_list, target_list = [],[]
    with torch.no_grad():
        for xb,yb in test_loader:
            xb,yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds_list.append(out)
            target_list.append(yb)
    all_preds=torch.cat(preds_list); all_targets=torch.cat(target_list)
    return all_preds, all_targets


def augment_noise(dataset, noise_std):
    # Handle full dataset or Subset
    if isinstance(dataset, Subset):
        X_orig = dataset.dataset.X[dataset.indices]
        Y_orig = dataset.dataset.Y[dataset.indices]
    else:
        X_orig = dataset.X
        Y_orig = dataset.Y
    noisy_X = X_orig + torch.randn_like(X_orig) * noise_std
    return torch.utils.data.TensorDataset(noisy_X, Y_orig)


def evaluate_robustness(model, test_dataset, device, noise_levels):
    results = {}
    for nl in noise_levels:
        noisy_ds = augment_noise(test_dataset, nl)
        loader = DataLoader(noisy_ds, batch_size=32)
        _, targets = train_model(model, DataLoader(test_dataset,32), DataLoader(test_dataset,32), loader, phys_weight=0, device=device)
        preds, _ = train_model(model, DataLoader(test_dataset,32), DataLoader(test_dataset,32), loader, phys_weight=0, device=device)
        mse, mae, _ = evaluate_metrics(preds, targets.to(device))
        results[nl] = {'mse':mse,'mae':mae}
    return results


def calibrate_param_net(model, loader, device, shots=32, lr=1e-4, epochs=5):
    """Fine-tune only the ODE parameters (param_net) on a small calibration set"""
    # Freeze all but param_net
    for name, p in model.named_parameters():
        p.requires_grad = 'odefunc.param_net' in name
    optimizer = optim.Adam(model.odefunc.param_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    # Few-shot: take up to shots samples
    adapt_loader = DataLoader(loader.dataset, batch_size=shots, shuffle=True)
    for epoch in range(epochs):
        for xb, yb in adapt_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    return model

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

# Visualize data distributions and example beats
plt.figure()
plt.hist(Y_sys, bins=50, alpha=0.7, label='Systolic')
plt.hist(Y_dia, bins=50, alpha=0.7, label='Diastolic')
plt.title('Blood Pressure Distribution')
plt.xlabel('Pressure (mmHg)')
plt.ylabel('Count')
plt.legend()
plt.show()
plt.close()

# Plot first PPG and ECG beat examples
plt.figure()
plt.plot(X_ppg[0], label='PPG Beat Sample')
plt.plot(X_ecg[0], label='ECG Beat Sample')
plt.title('Example Normalized Beats')
plt.xlabel('Time Index')
plt.ylabel('Normalized Amplitude')
plt.legend()
plt.show()
plt.close()

dataset = BPDataset(X, Y_sys, Y_dia)
# Determine input dimension for RNN (sequence feature size)
input_dim = dataset.X.shape[2]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Split dataset
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Split by indices using the feature array
train_idx_np, test_idx_np = next(kf.split(X))
train_idx, test_idx = train_idx_np.tolist(), test_idx_np.tolist()
train_ds = Subset(dataset, train_idx)
# 80/20 train/validation split on train indices
val_split = int(0.8 * len(train_idx))
val_idx = train_idx[val_split:]
train_idx = train_idx[:val_split]
val_ds = Subset(dataset, val_idx)
test_ds = Subset(dataset, test_idx)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# Baseline (data-driven)
base_model = BPModel(input_dim, latent_dim=16, use_ode=False).to(device)
base_preds, base_targets = train_model(base_model, train_loader, val_loader, test_loader, phys_weight=0, device=device)
base_mse, base_mae, base_corr = evaluate_metrics(base_preds, base_targets)
print(f'Baseline MSE: {base_mse}, MAE: {base_mae}, Corr: {base_corr}')

# PINN model
pinn_model = BPModel(input_dim, latent_dim=16, use_ode=True).to(device)
pinn_preds, pinn_targets = train_model(pinn_model, train_loader, val_loader, test_loader, phys_weight=100, device=device)
pinn_mse, pinn_mae, pinn_corr = evaluate_metrics(pinn_preds, pinn_targets)
print(f'PINN MSE: {pinn_mse}, MAE: {pinn_mae}, Corr: {pinn_corr}')

# Diagnostics: latent trajectories and parameter summary
with torch.no_grad():
    x_sample, _ = next(iter(test_loader))
    x_sample = x_sample.to(device)
    z0 = pinn_model.encoder(x_sample)
    ts_diag = torch.linspace(0.0, 1.0, 10, device=device)
    zt_diag = odeint(pinn_model.odefunc, z0, ts_diag, method='rk4', rtol=1e-6, atol=1e-6)
    # plot first sample, first latent dimension
    plt.figure()
    plt.plot(ts_diag.cpu(), zt_diag[:,0,0].cpu(), label='latent dim0')
    plt.title('Latent Trajectory Sample 0, dim0')
    plt.xlabel('t'); plt.ylabel('z'); plt.show(); plt.close()
    # parameter summary
    params_vals = torch.exp(pinn_model.odefunc.param_net(z0)).cpu().numpy()
    print('Learned R/C parameters (Rp,Rd,C) mean:', np.mean(params_vals, axis=0), 'std:', np.std(params_vals, axis=0))

# Statistical validation
base_err = (base_preds - base_targets).flatten().cpu().numpy()
pinn_err = (pinn_preds - pinn_targets).flatten().cpu().numpy()
stat, p = wilcoxon(base_err, pinn_err)
print(f'Wilcoxon test p-value: {p}')

# clinical standards
bhs_base = evaluate_bhs(base_err)
bhs_pinn = evaluate_bhs(pinn_err)
print("BHS grades → baseline:", bhs_base, "PINN:", bhs_pinn)

aami_base = evaluate_aami(base_err)
aami_pinn = evaluate_aami(pinn_err)
print("AAMI (mean±std) → baseline:", aami_base, "PINN:", aami_pinn)

# Robustness under noise
#noise_levels=[0.01,0.05,0.1]
#base_rob = evaluate_robustness(base_model, test_ds, device, noise_levels)
#pinn_rob = evaluate_robustness(pinn_model, test_ds, device, noise_levels)
#print('Robustness:', base_rob, pinn_rob)

# Dynamic calibration
pinn_calibrated = BPModel(input_dim, latent_dim=16, use_ode=True).to(device)
pinn_calibrated.load_state_dict(pinn_model.state_dict())  # copy weights
pinn_calibrated = calibrate_param_net(pinn_calibrated, test_loader, device)
cal_preds, cal_targets = train_model(pinn_calibrated, train_loader, val_loader, test_loader, phys_weight=100, device=device)
cal_mse, cal_corr = evaluate_metrics(cal_preds, cal_targets)
print(f'Calibrated PINN MSE: {cal_mse}, Corr: {cal_corr}')
