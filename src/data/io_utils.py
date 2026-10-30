from __future__ import annotations
import json
from fractions import Fraction
from pathlib import Path
import mne
import numpy as np
from scipy import signal as scipy_signal
from .channel_map import build_montage_signal, canonicalize_channel_name
mne.set_log_level('ERROR')

def load_eeg_file(path: str):
    p = Path(path)
    if p.suffix.lower() == '.npy':
        arr = np.load(p)
        meta = json.loads(p.with_suffix('.json').read_text())
        return arr.astype(np.float32), float(meta['sampling_rate']), list(meta['channel_names'])
    if p.suffix.lower() == '.edf':
        raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
        return raw.get_data().astype(np.float32), float(raw.info['sfreq']), [str(ch) for ch in raw.ch_names]
    raise ValueError(f'Unsupported format: {p}')

def bandpass_and_notch(signal: np.ndarray, sampling_rate: float, low_hz: float, high_hz: float, notch_hz: float | None = 50.0):
    out = signal.astype(np.float32)
    if notch_hz is not None and sampling_rate / 2.0 > notch_hz + 1.0:
        b, a = scipy_signal.iirnotch(w0=notch_hz, Q=30.0, fs=sampling_rate)
        out = scipy_signal.filtfilt(b, a, out, axis=-1).astype(np.float32)
    b, a = scipy_signal.butter(4, [low_hz, high_hz], btype='bandpass', fs=sampling_rate)
    out = scipy_signal.filtfilt(b, a, out, axis=-1).astype(np.float32)
    return out

def resample_signal(signal: np.ndarray, orig_sr: float, target_sr: float):
    if abs(orig_sr - target_sr) < 1e-6:
        return signal.astype(np.float32)
    ratio = Fraction(float(target_sr) / float(orig_sr)).limit_denominator(1000)
    out = scipy_signal.resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1)
    target_len = int(round(signal.shape[-1] * target_sr / orig_sr))
    if out.shape[-1] > target_len:
        out = out[..., :target_len]
    elif out.shape[-1] < target_len:
        out = np.pad(out, [(0, 0)] * (out.ndim - 1) + [(0, target_len - out.shape[-1])], mode='edge')
    return out.astype(np.float32)

def standardize_record(eeg_path: str, target_channels, target_sampling_rate: float, low_hz: float, high_hz: float, notch_hz: float | None):
    signal, sr, ch_names = load_eeg_file(eeg_path)
    ch_names = [canonicalize_channel_name(c) for c in ch_names]
    target_channels = [canonicalize_channel_name(c) for c in target_channels]
    signal = build_montage_signal(signal, ch_names, target_channels)
    anti_alias_high_hz = min(float(high_hz), 0.45 * float(target_sampling_rate))
    signal = bandpass_and_notch(signal, sr, low_hz, anti_alias_high_hz, notch_hz)
    signal = resample_signal(signal, sr, target_sampling_rate)
    return signal.astype(np.float32), target_sampling_rate, target_channels
