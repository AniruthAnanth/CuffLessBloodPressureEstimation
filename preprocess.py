# A script for preprocessing
# This script processes .mat files containing physiological signals (PPG, ABP, ECG)
# and extracts features for further analysis. It includes smoothing, filtering,

import h5py
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks, correlate

INPUT_DIR = 'data/mat/'
OUTPUT_DIR = 'data/csv/'

def convert_file(filename: str) -> pd.DataFrame:
    """
    Convert .mat file to .csv format, extracting various features including
    PTT, heart rate, blood pressure, and additional PPG-based features
    (AI, LASI, S1, S2, S3, S4).
    """
    name = filename.split('.')[0]
    fs = 125  # Hz
    block_size = 10 * fs  # 10-second blocks

    print(f'Loading {filename}...')
    mat_file = h5py.File(INPUT_DIR + filename, 'r')
    part1_refs = mat_file[name]

    results = []
    print(f'Iterating through {len(part1_refs)} records...')
    for i in range(part1_refs.shape[0]):
        element_ref = part1_refs[i][0]
        element_data = np.array(mat_file[element_ref]).T

        ppg = element_data[0, :]
        abp = element_data[1, :]
        ecg = element_data[2, :]

        num_blocks = len(ppg) // block_size
        for b in range(num_blocks):
            idx_start = b * block_size
            idx_end = (b + 1) * block_size

            ppg_block = ppg[idx_start:idx_end]
            abp_block = abp[idx_start:idx_end]
            ecg_block = ecg[idx_start:idx_end]

            smooth_ppg = np.convolve(ppg_block, np.ones(5)/5, mode='same')
            smooth_abp = np.convolve(abp_block, np.ones(5)/5, mode='same')
            smooth_ecg = np.convolve(ecg_block, np.ones(5)/5, mode='same')

            sbp = np.max(smooth_abp)
            dbp = np.min(smooth_abp)
            if sbp > 180 or sbp < 80 or dbp > 130 or dbp < 50:
                continue

            peaks, _ = find_peaks(smooth_ecg, height=np.mean(smooth_ecg), distance=int(0.3 * fs))
            if len(peaks) < 2:
                continue
            rr_intervals = np.diff(peaks) / fs
            hr = 60 / rr_intervals
            mean_hr = np.mean(hr)
            if mean_hr < 40 or mean_hr > 180:
                continue

            if np.max(np.abs(np.diff(smooth_ppg))) > 3 * np.std(smooth_ppg):
                continue

            acf = correlate(smooth_ppg - np.mean(smooth_ppg), smooth_ppg - np.mean(smooth_ppg), mode='full')
            acf = acf[acf.size // 2:]
            if np.max(acf[1:]) / acf[0] < 0.3:
                continue

            r_peaks, _ = find_peaks(smooth_ecg, height=np.mean(smooth_ecg), distance=int(0.3 * fs))
            ppg_peaks, _ = find_peaks(smooth_ppg)
            ppg_troughs, _ = find_peaks(-smooth_ppg)
            max_slope_idx = np.argmax(np.diff(smooth_ppg))

            ptt_p, ptt_f, ptt_d = [], [], []
            for r in r_peaks:
                future_ppg_peaks = ppg_peaks[ppg_peaks > r]
                future_ppg_troughs = ppg_troughs[ppg_troughs > r]
                if future_ppg_peaks.size > 0:
                    ptt_p.append((future_ppg_peaks[0] - r) / fs)
                if future_ppg_troughs.size > 0:
                    ptt_f.append((future_ppg_troughs[0] - r) / fs)
                if max_slope_idx > r:
                    ptt_d.append((max_slope_idx - r) / fs)

            if not ptt_p or not ptt_f or not ptt_d:
                continue

            # Additional PPG-based features
            if ppg_peaks.size < 2:
                continue
            # Find systolic peak (largest peak)
            systolic_idx = ppg_peaks[np.argmax(smooth_ppg[ppg_peaks])]
            systolic_amp = smooth_ppg[systolic_idx]
            # Find diastolic peak (largest after systolic)
            diastolic_candidates = ppg_peaks[ppg_peaks > systolic_idx]
            if diastolic_candidates.size < 1:
                continue
            diastolic_idx = diastolic_candidates[np.argmax(smooth_ppg[diastolic_candidates])]
            diastolic_amp = smooth_ppg[diastolic_idx]

            AI = diastolic_amp / systolic_amp
            LASI = abs(diastolic_idx - systolic_idx) / fs

            # Areas for S1, S2, S3, S4
            S1 = np.trapz(smooth_ppg[:systolic_idx], dx=1/fs) if systolic_idx > 0 else 0
            S2 = np.trapz(smooth_ppg[systolic_idx:diastolic_idx], dx=1/fs) if diastolic_idx > systolic_idx else 0
            S3 = np.trapz(smooth_ppg[diastolic_idx:], dx=1/fs) if diastolic_idx < len(smooth_ppg) else 0
            S4 = np.trapz(smooth_ppg, dx=1/fs)

            feature_row = [
                np.mean(ptt_p),
                np.mean(ptt_f),
                np.mean(ptt_d),
                mean_hr,
                sbp,
                dbp,
                AI,
                LASI,
                S1,
                S2,
                S3,
                S4
            ]
            results.append(feature_row)

    print(f'Extracted {len(results)} valid records.')
    columns = [
        'PTTp', 'PTTf', 'PTTd', 'HeartRate', 'SBP', 'DBP',
        'AI', 'LASI', 'S1', 'S2', 'S3', 'S4'
    ]
    df = pd.DataFrame(results, columns=columns)
    if __name__ == '__main__':
        print(f'Saving {name}.csv...')
        df.to_csv(os.path.join(OUTPUT_DIR, name + '.csv'), index=False)

    return df

if __name__ == '__main__':
    all_dfs = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.mat'):
            print(f'Processing {filename}...')
            df = convert_file(filename)
            all_dfs.append(df)
    if all_dfs:
        mega_df = pd.concat(all_dfs, ignore_index=True)
        print(f'Saving all.csv with {len(mega_df)} records...')
        mega_df.to_csv(os.path.join(OUTPUT_DIR, 'All.csv'), index=False)