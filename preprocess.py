# A script for preprocessing
# This script processes .mat files containing physiological signals (PPG, ABP, ECG)
# and extracts features for further analysis. It includes smoothing, filtering,

import h5py
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks, correlate, welch, butter, filtfilt
from scipy.integrate import trapz
from scipy.stats import entropy

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

def calculate_time_to_peak(ppg_signal, peaks, troughs, fs):
    """Calculate time-to-peak features"""
    if len(peaks) < 2 or len(troughs) < 2:
        return {'time_to_peak': 0, 'peak_to_peak_time': 0}
    
    time_to_peaks = []
    peak_to_peak_times = []
    
    for i, peak in enumerate(peaks):
        # Find corresponding trough before peak
        prev_troughs = troughs[troughs < peak]
        if len(prev_troughs) > 0:
            trough_before = prev_troughs[-1]
            time_to_peak = (peak - trough_before) / fs
            time_to_peaks.append(time_to_peak)
    
    # Peak-to-peak intervals
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks) / fs
        peak_to_peak_times.extend(peak_intervals)
    
    return {
        'time_to_peak': np.mean(time_to_peaks) if time_to_peaks else 0,
        'peak_to_peak_time': np.mean(peak_to_peak_times) if peak_to_peak_times else 0
    }

def calculate_ppg_asymmetry_harmonics(ppg_signal, peaks, fs):
    """Calculate PPG asymmetry and harmonic ratios"""
    if len(peaks) < 2:
        return {'asymmetry_ratio': 0, 'harmonic_ratio_2nd': 0, 'harmonic_ratio_3rd': 0}
    
    asymmetry_ratios = []
    
    for i, peak in enumerate(peaks):
        # Define pulse boundaries
        if i == 0:
            start_idx = max(0, peak - int(0.5 * fs))
        else:
            start_idx = (peaks[i-1] + peak) // 2
        
        if i == len(peaks) - 1:
            end_idx = min(len(ppg_signal), peak + int(0.5 * fs))
        else:
            end_idx = (peak + peaks[i+1]) // 2
        
        pulse = ppg_signal[start_idx:end_idx]
        if len(pulse) < 10:
            continue
        
        # Calculate asymmetry ratio
        peak_in_pulse = np.argmax(pulse)
        if peak_in_pulse > 0 and peak_in_pulse < len(pulse) - 1:
            upstroke_area = np.trapz(pulse[:peak_in_pulse])
            downstroke_area = np.trapz(pulse[peak_in_pulse:])
            if downstroke_area != 0:
                asymmetry_ratio = upstroke_area / downstroke_area
                asymmetry_ratios.append(asymmetry_ratio)
    
    # Calculate harmonic ratios using FFT
    freqs, psd = welch(ppg_signal, fs=fs, nperseg=min(len(ppg_signal)//4, 512))
    
    # Find fundamental frequency (around heart rate)
    hr_freq_range = (0.8, 3.0)  # 48-180 bpm
    hr_mask = (freqs >= hr_freq_range[0]) & (freqs <= hr_freq_range[1])
    if np.sum(hr_mask) > 0:
        fundamental_idx = np.argmax(psd[hr_mask])
        fundamental_freq = freqs[hr_mask][fundamental_idx]
        fundamental_power = psd[hr_mask][fundamental_idx]
        
        # Find 2nd and 3rd harmonics
        harmonic_2_freq = 2 * fundamental_freq
        harmonic_3_freq = 3 * fundamental_freq
        
        # Find closest frequency bins
        h2_idx = np.argmin(np.abs(freqs - harmonic_2_freq))
        h3_idx = np.argmin(np.abs(freqs - harmonic_3_freq))
        
        harmonic_2_power = psd[h2_idx] if h2_idx < len(psd) else 0
        harmonic_3_power = psd[h3_idx] if h3_idx < len(psd) else 0
        
        harmonic_ratio_2nd = harmonic_2_power / fundamental_power if fundamental_power > 0 else 0
        harmonic_ratio_3rd = harmonic_3_power / fundamental_power if fundamental_power > 0 else 0
    else:
        harmonic_ratio_2nd = 0
        harmonic_ratio_3rd = 0
    
    return {
        'asymmetry_ratio': np.mean(asymmetry_ratios) if asymmetry_ratios else 0,
        'harmonic_ratio_2nd': harmonic_ratio_2nd,
        'harmonic_ratio_3rd': harmonic_ratio_3rd
    }

def calculate_enhanced_ptt_pat(ecg_signal, ppg_signal, fs):
    """Calculate enhanced PTT and PAT with better alignment"""
    # Find R-peaks in ECG with improved detection
    ecg_filtered = filtfilt(*butter(3, [0.5, 15], btype='band', fs=fs), ecg_signal)
    r_peaks, _ = find_peaks(ecg_filtered, height=np.percentile(ecg_filtered, 80), distance=int(0.3 * fs))
    
    # Find PPG peaks and feet with improved detection
    ppg_filtered = filtfilt(*butter(3, [0.5, 8], btype='band', fs=fs), ppg_signal)
    ppg_peaks, _ = find_peaks(ppg_filtered, distance=int(0.3 * fs))
    ppg_feet, _ = find_peaks(-ppg_filtered, distance=int(0.3 * fs))
    
    # Calculate maximum slope points (dicrotic notch)
    ppg_diff = np.diff(ppg_filtered)
    ppg_max_slope = find_peaks(ppg_diff, distance=int(0.1 * fs))[0]
    
    if len(r_peaks) < 2:
        return {'ptt_peak': 0, 'ptt_foot': 0, 'pat_slope': 0, 'ptt_variability': 0}
    
    ptt_peaks, ptt_feet, pat_slopes = [], [], []
    
    for r_peak in r_peaks:
        # Find next PPG peak
        future_ppg_peaks = ppg_peaks[ppg_peaks > r_peak]
        if len(future_ppg_peaks) > 0:
            next_ppg_peak = future_ppg_peaks[0]
            if (next_ppg_peak - r_peak) / fs < 0.5:  # Reasonable PTT range
                ptt_peaks.append((next_ppg_peak - r_peak) / fs)
        
        # Find next PPG foot
        future_ppg_feet = ppg_feet[ppg_feet > r_peak]
        if len(future_ppg_feet) > 0:
            next_ppg_foot = future_ppg_feet[0]
            if (next_ppg_foot - r_peak) / fs < 0.5:
                ptt_feet.append((next_ppg_foot - r_peak) / fs)
        
        # Find next maximum slope point
        future_slopes = ppg_max_slope[ppg_max_slope > r_peak]
        if len(future_slopes) > 0:
            next_slope = future_slopes[0]
            if (next_slope - r_peak) / fs < 0.5:
                pat_slopes.append((next_slope - r_peak) / fs)
    
    # Calculate variability
    ptt_variability = np.std(ptt_peaks) if len(ptt_peaks) > 1 else 0
    
    return {
        'ptt_peak': np.mean(ptt_peaks) if ptt_peaks else 0,
        'ptt_foot': np.mean(ptt_feet) if ptt_feet else 0,
        'pat_slope': np.mean(pat_slopes) if pat_slopes else 0,
        'ptt_variability': ptt_variability
    }

def calculate_spectral_entropy(signal, fs):
    """Calculate spectral entropy of the signal"""
    try:
        # Calculate power spectral density
        freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal)//4, 512))
        
        # Normalize PSD to get probability distribution
        psd_norm = psd / np.sum(psd)
        
        # Calculate entropy
        spectral_entropy = entropy(psd_norm)
        
        return spectral_entropy
    except:
        return 0

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
            
            # Calculate new features
            time_to_peak = calculate_time_to_peak(smooth_ppg, ppg_peaks, ppg_troughs, fs)
            ppg_asymmetry = calculate_ppg_asymmetry_harmonics(smooth_ppg, ppg_peaks, fs)
            enhanced_ptt = calculate_enhanced_ptt_pat(smooth_ecg, smooth_ppg, fs)
            ppg_spectral_entropy = calculate_spectral_entropy(smooth_ppg, fs)
            ecg_spectral_entropy = calculate_spectral_entropy(smooth_ecg, fs)

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
                # More features
                time_to_peak['time_to_peak'],
                time_to_peak['peak_to_peak_time'],
                ppg_asymmetry['asymmetry_ratio'],
                ppg_asymmetry['harmonic_ratio_2nd'],
                ppg_asymmetry['harmonic_ratio_3rd'],
                enhanced_ptt['ptt_peak'],
                enhanced_ptt['ptt_foot'],
                enhanced_ptt['pat_slope'],
                enhanced_ptt['ptt_variability'],
                ppg_spectral_entropy,
                ecg_spectral_entropy,
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
        # New features
        'time_to_peak', 'peak_to_peak_time', 'asymmetry_ratio',
        'harmonic_ratio_2nd', 'harmonic_ratio_3rd', 'ptt_peak_enhanced',
        'ptt_foot_enhanced', 'pat_slope', 'ptt_variability',
        'ppg_spectral_entropy', 'ecg_spectral_entropy',
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