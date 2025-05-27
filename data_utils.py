import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

INPUT_DIR = 'data/mat/'

def load_waveforms(filename):
    name = os.path.splitext(filename)[0]
    path = os.path.join(INPUT_DIR, filename)
    mat = h5py.File(path, 'r')
    refs = mat[name]
    element_ref = refs[0][0]
    data = np.array(mat[element_ref]).T
    mat.close()
    ppg, abp, ecg = data[0, :], data[1, :], data[2, :]
    return ppg, abp, ecg

def detect_r_peaks(ecg, fs=125, distance_sec=0.6):
    from scipy.signal import find_peaks
    distance = int(distance_sec * fs)
    threshold = np.mean(ecg) + 0.5 * np.std(ecg)
    peaks, _ = find_peaks(ecg, distance=distance, height=threshold)
    return peaks

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

def normalize_beats(beats):
    return np.array([(b - np.mean(b)) / np.std(b) for b in beats])

def compute_bp_features(abp_beats):
    systolic = np.max(abp_beats, axis=1)
    diastolic = np.min(abp_beats, axis=1)
    return systolic, diastolic

class BPDataset(Dataset):
    def __init__(self, X, Y_sys, Y_dia):
        # X is expected to be [N, 150] where 150 = 75 PPG + 75 ECG features
        # For RNN, we need to reshape it to [N, seq_len, input_dim]
        # Let's assume we want to treat each beat as a sequence of 75 time steps with 2 features (PPG and ECG)
        self.X = torch.tensor(X, dtype=torch.float32)
        
        # Reshape from [N, 150] to [N, 75, 2] for RNN processing
        # First 75 features are PPG, next 75 are ECG
        N = self.X.shape[0]
        ppg_features = self.X[:, :75]  # [N, 75]
        ecg_features = self.X[:, 75:]  # [N, 75]
        
        # Stack to create [N, 75, 2] where dim 2 contains [PPG, ECG] at each time step
        self.X = torch.stack([ppg_features, ecg_features], dim=2)  # [N, 75, 2]
        
        self.Y = torch.stack([
            torch.tensor(Y_sys, dtype=torch.float32),
            torch.tensor(Y_dia, dtype=torch.float32)
        ], dim=1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
