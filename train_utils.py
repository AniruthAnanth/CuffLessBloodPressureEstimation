import torch
import torch.nn as nn
from scipy.stats import pearsonr
import numpy as np

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
    mse = nn.MSELoss()(preds, targets).item()
    mae = nn.L1Loss()(preds, targets).item()
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    # pearsonr returns (correlation, p-value) or PearsonRResult, always extract the first element as float
    def get_corr_val(corr_result):
        if hasattr(corr_result, 'statistic'):
            return float(corr_result.statistic)
        elif hasattr(corr_result, '__getitem__'):
            return float(corr_result[0])
        else:
            return float(corr_result)
    corr_sys = get_corr_val(pearsonr(preds_np[:,0], targets_np[:,0]))
    corr_dia = get_corr_val(pearsonr(preds_np[:,1], targets_np[:,1]))
    avg_corr = (corr_sys + corr_dia) / 2.0
    return mse, mae, avg_corr

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
