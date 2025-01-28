import wfdb
import numpy as np
from scipy.signal import resample
import os

def load_data(record_path):
    # Load signals and annotations
    signals, fields = wfdb.rdsamp(record_path)
    annotations = wfdb.rdann(record_path, 'apn')  # Load apnea annotations (.st file)
    
    # Extract channels
    ecg = signals[:, 0]  # ECG signal (channel 0)
    resp = signals[:, 1]  # Respiration signal (channel 1)

    # Create labeled data
    data = []
    for i, label in enumerate(annotations.sample):
        start = label
        end = annotations.sample[i + 1] if i + 1 < len(annotations.sample) else len(ecg)
        apnea_label = 1 if annotations.symbol[i] == 'A' else 0  # 'A' denotes apnea
        data.append((ecg[start:end], resp[start:end], apnea_label))

    return data

def preprocess_data(data, sampling_rate=100, window_size=10):
    X, y = [], []
    window_samples = sampling_rate * window_size
    for ecg, resp, label in data:
        # Resample to fixed size
        ecg_resampled = resample(ecg, window_samples)
        resp_resampled = resample(resp, window_samples)
        
        # Normalize
        ecg_resampled = (ecg_resampled - np.mean(ecg_resampled)) / np.std(ecg_resampled)
        resp_resampled = (resp_resampled - np.mean(resp_resampled)) / np.std(resp_resampled)
        
        # Stack ECG and respiration
        X.append(np.stack((ecg_resampled, resp_resampled), axis=1))
        y.append(label)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Path to records
    record_path = "data/raw/record1"
    
    # Load and preprocess data
    data = load_data(record_path)
    X, y = preprocess_data(data)
    
    # Save processed data
    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)
    print(f"Data saved: X shape = {X.shape}, y shape = {y.shape}")
