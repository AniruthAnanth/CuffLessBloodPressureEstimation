# A script for preprocessing
# This script processes .mat files containing physiological signals (PPG, ABP, ECG)
# and extracts features for further analysis. It includes smoothing, filtering,

import h5py
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks, correlate, welch, butter, filtfilt
from scipy.integrate import trapz

INPUT_DIR = 'data/mat/'
OUTPUT_DIR = 'data/csv/'

def calculate_hrv_time_domain(rr_intervals):
    """Calculate time-domain HRV features"""
    if len(rr_intervals) < 2:
        return {'SDNN': 0, 'RMSSD': 0, 'pNN50': 0}
    
    rr_ms = rr_intervals * 1000  # Convert to ms
    sdnn = np.std(rr_ms, ddof=1)
    
    rr_diff = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(rr_diff**2))
    
    pnn50 = np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100
    
    return {'SDNN': sdnn, 'RMSSD': rmssd, 'pNN50': pnn50}

def calculate_hrv_frequency_domain(rr_intervals, fs=4):
    """Calculate frequency-domain HRV features"""
    if len(rr_intervals) < 10:
        return {'LF_power': 0, 'HF_power': 0, 'LF_HF_ratio': 0}
    
    # Resample RR intervals to uniform sampling
    time_original = np.cumsum(rr_intervals)
    time_uniform = np.arange(0, time_original[-1], 1/fs)
    rr_uniform = np.interp(time_uniform, time_original, rr_intervals)
    
    # Calculate PSD
    freqs, psd = welch(rr_uniform, fs=fs, nperseg=min(len(rr_uniform)//2, 256))
    
    # Define frequency bands
    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
    hf_band = (freqs >= 0.15) & (freqs <= 0.4)
    
    lf_power = trapz(psd[lf_band], freqs[lf_band])
    hf_power = trapz(psd[hf_band], freqs[hf_band])
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    
    return {'LF_power': lf_power, 'HF_power': hf_power, 'LF_HF_ratio': lf_hf_ratio}

def calculate_ppg_morphology(ppg_signal, peaks, troughs, fs):
    """Calculate PPG morphology features"""
    if len(peaks) < 2 or len(troughs) < 2:
        return {'pulse_amp_var': 0, 'rise_time': 0, 'decay_time': 0, 'half_width': 0}
    
    # Pulse amplitude variability
    peak_amps = ppg_signal[peaks]
    pulse_amp_var = np.std(peak_amps) if len(peak_amps) > 1 else 0
    
    # Rise and decay times (average across pulses)
    rise_times, decay_times, half_widths = [], [], []
    
    for i, peak in enumerate(peaks):
        # Find corresponding trough before peak
        prev_troughs = troughs[troughs < peak]
        if len(prev_troughs) > 0:
            trough_before = prev_troughs[-1]
            rise_time = (peak - trough_before) / fs
            rise_times.append(rise_time)
            
            # Half-width calculation
            peak_amp = ppg_signal[peak]
            trough_amp = ppg_signal[trough_before]
            half_amp = trough_amp + (peak_amp - trough_amp) / 2
            
            # Find indices where signal crosses half amplitude
            pulse_segment = ppg_signal[trough_before:peak+1]
            half_crossings = np.where(np.diff(np.sign(pulse_segment - half_amp)))[0]
            if len(half_crossings) >= 2:
                half_width = (half_crossings[-1] - half_crossings[0]) / fs
                half_widths.append(half_width)
        
        # Find trough after peak for decay time
        next_troughs = troughs[troughs > peak]
        if len(next_troughs) > 0:
            trough_after = next_troughs[0]
            decay_time = (trough_after - peak) / fs
            decay_times.append(decay_time)
    
    return {
        'pulse_amp_var': pulse_amp_var,
        'rise_time': np.mean(rise_times) if rise_times else 0,
        'decay_time': np.mean(decay_times) if decay_times else 0,
        'half_width': np.mean(half_widths) if half_widths else 0
    }

def calculate_augmentation_indices(ppg_signal, peaks):
    """Calculate augmentation and stiffness indices"""
    if len(peaks) < 1:
        return {'aug_index': 0, 'stiffness_index': 0}
    
    aug_indices, stiffness_indices = [], []
    
    for peak_idx in peaks:
        # Find pulse boundaries (simplified approach)
        start_idx = max(0, peak_idx - 50)
        end_idx = min(len(ppg_signal), peak_idx + 100)
        pulse = ppg_signal[start_idx:end_idx]
        
        if len(pulse) < 10:
            continue
            
        # Augmentation index (ratio of late systolic component)
        peak_amp = np.max(pulse)
        # Find secondary peak (dicrotic notch area)
        peak_in_pulse = np.argmax(pulse)
        if peak_in_pulse < len(pulse) - 10:
            late_portion = pulse[peak_in_pulse + 5:]
            if len(late_portion) > 0:
                late_peak = np.max(late_portion)
                aug_index = late_peak / peak_amp if peak_amp > 0 else 0
                aug_indices.append(aug_index)
        
        # Stiffness index (height/time ratio)
        pulse_duration = len(pulse) / 125  # assuming 125 Hz
        if pulse_duration > 0:
            stiffness_index = peak_amp / pulse_duration
            stiffness_indices.append(stiffness_index)
    
    return {
        'aug_index': np.mean(aug_indices) if aug_indices else 0,
        'stiffness_index': np.mean(stiffness_indices) if stiffness_indices else 0
    }

def calculate_abp_metrics(abp_signal, sbp, dbp):
    """Calculate ABP-derived metrics"""
    pulse_pressure = sbp - dbp
    map_pressure = dbp + pulse_pressure / 3
    
    # dP/dtmax (maximum slope)
    dp_dt = np.diff(abp_signal)
    dp_dt_max = np.max(dp_dt) if len(dp_dt) > 0 else 0
    
    return {
        'pulse_pressure': pulse_pressure,
        'map_pressure': map_pressure,
        'dp_dt_max': dp_dt_max
    }

def estimate_respiratory_rate(signal, fs):
    """Estimate respiratory rate from signal baseline"""
    # Apply low-pass filter to extract baseline
    nyquist = fs / 2
    low_cutoff = 0.5 / nyquist
    b, a = butter(3, low_cutoff, btype='low')
    baseline = filtfilt(b, a, signal)
    
    # Find peaks in baseline (breathing cycles)
    resp_peaks, _ = find_peaks(baseline, distance=int(fs * 2))  # Min 2 seconds between breaths
    
    if len(resp_peaks) < 2:
        return 0
    
    resp_intervals = np.diff(resp_peaks) / fs
    resp_rate = 60 / np.mean(resp_intervals) if len(resp_intervals) > 0 else 0
    
    return min(resp_rate, 60)  # Cap at 60 breaths/min

def convert_file(filename: str) -> pd.DataFrame:
    """
    Convert .mat file to .csv format, extracting comprehensive features
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

            # Calculate HRV features
            hrv_time = calculate_hrv_time_domain(rr_intervals)
            hrv_freq = calculate_hrv_frequency_domain(rr_intervals, fs)
            
            # Calculate PPG morphology features
            ppg_morphology = calculate_ppg_morphology(smooth_ppg, ppg_peaks, ppg_troughs, fs)
            
            # Calculate augmentation indices
            aug_indices = calculate_augmentation_indices(smooth_ppg, ppg_peaks)
            
            # Calculate ABP metrics
            abp_metrics = calculate_abp_metrics(smooth_abp, sbp, dbp)
            
            # Calculate respiratory rate
            resp_rate = estimate_respiratory_rate(smooth_ppg, fs)

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
                S4,
                # HRV features
                hrv_time['SDNN'],
                hrv_time['RMSSD'],
                hrv_time['pNN50'],
                hrv_freq['LF_power'],
                hrv_freq['HF_power'],
                hrv_freq['LF_HF_ratio'],
                # PPG morphology
                ppg_morphology['pulse_amp_var'],
                ppg_morphology['rise_time'],
                ppg_morphology['decay_time'],
                ppg_morphology['half_width'],
                # Augmentation indices
                aug_indices['aug_index'],
                aug_indices['stiffness_index'],
                # ABP metrics
                abp_metrics['pulse_pressure'],
                abp_metrics['map_pressure'],
                abp_metrics['dp_dt_max'],
                # Respiratory
                resp_rate,
            ]
            results.append(feature_row)

    print(f'Extracted {len(results)} valid records.')
    columns = [
        'PTTp', 'PTTf', 'PTTd', 'HeartRate', 'SBP', 'DBP',
        'AI', 'LASI', 'S1', 'S2', 'S3', 'S4',
        # HRV features
        'SDNN', 'RMSSD', 'pNN50', 'LF_power', 'HF_power', 'LF_HF_ratio',
        # PPG morphology
        'pulse_amp_var', 'rise_time', 'decay_time', 'half_width',
        # Augmentation indices
        'aug_index', 'stiffness_index',
        # ABP metrics
        'pulse_pressure', 'map_pressure', 'dp_dt_max',
        # Respiratory
        'respiratory_rate',
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