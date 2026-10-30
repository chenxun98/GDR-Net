from __future__ import annotations
import re
import numpy as np

CHANNEL_POSITIONS = {
    "FP1": (-0.50, 1.00), "FP2": (0.50, 1.00), "F7": (-1.00, 0.55), "F3": (-0.45, 0.55),
    "FZ": (0.00, 0.60), "F4": (0.45, 0.55), "F8": (1.00, 0.55), "T3": (-1.10, 0.00),
    "C3": (-0.50, 0.00), "CZ": (0.00, 0.00), "C4": (0.50, 0.00), "T4": (1.10, 0.00),
    "T5": (-1.00, -0.55), "P3": (-0.45, -0.55), "PZ": (0.00, -0.60), "P4": (0.45, -0.55),
    "T6": (1.00, -0.55), "O1": (-0.40, -1.00), "O2": (0.40, -1.00),
    "T7": (-1.10, 0.00), "T8": (1.10, 0.00), "P7": (-1.00, -0.55), "P8": (1.00, -0.55),
    "FT9": (-1.25, 0.28), "FT10": (1.25, 0.28),
}
ALIASES = {
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
    "EEG FP1-REF": "FP1", "EEG FP2-REF": "FP2", "EEG F3-REF": "F3", "EEG F4-REF": "F4",
    "EEG C3-REF": "C3", "EEG C4-REF": "C4", "EEG P3-REF": "P3", "EEG P4-REF": "P4",
    "EEG O1-REF": "O1", "EEG O2-REF": "O2", "EEG F7-REF": "F7", "EEG F8-REF": "F8",
    "EEG T3-REF": "T3", "EEG T4-REF": "T4", "EEG T5-REF": "T5", "EEG T6-REF": "T6",
    "EEG FZ-REF": "FZ", "EEG CZ-REF": "CZ", "EEG PZ-REF": "PZ",
}

def canonicalize_channel_name(name: str) -> str:
    cleaned = name.strip().upper().replace('.', '')
    cleaned = cleaned.removeprefix('EEG ').removesuffix('-REF').removesuffix('-LE')
    cleaned = re.sub(r'^(.+-.+)-\d+$', r'\1', cleaned)
    return ALIASES.get(cleaned, cleaned)

def _channel_position(name: str, fallback_index: int):
    name = canonicalize_channel_name(name)
    if name in CHANNEL_POSITIONS:
        return CHANNEL_POSITIONS[name]
    if '-' in name:
        parts = [p for p in name.split('-') if p]
        coords = [CHANNEL_POSITIONS[ALIASES.get(p, p)] for p in parts if ALIASES.get(p, p) in CHANNEL_POSITIONS]
        if coords:
            arr = np.asarray(coords, dtype=np.float32)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    return float(fallback_index), 0.0

def build_static_graph(channel_names, sigma: float = 0.7) -> np.ndarray:
    names = [canonicalize_channel_name(c) for c in channel_names]
    coords = [_channel_position(n, i) for i, n in enumerate(names)]
    coords = np.asarray(coords, dtype=np.float32)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    adj = np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj

def reorder_channels(signal: np.ndarray, signal_channels, target_channels):
    signal_channels = [canonicalize_channel_name(c) for c in signal_channels]
    target_channels = [canonicalize_channel_name(c) for c in target_channels]
    idx = {}
    for i, c in enumerate(signal_channels):
        idx.setdefault(c, i)
    out = []
    for c in target_channels:
        if c not in idx:
            raise ValueError(f'Missing required channel {c}. Available={signal_channels}')
        out.append(signal[idx[c]])
    return np.stack(out, axis=0)

def build_montage_signal(signal: np.ndarray, signal_channels, target_channels):
    signal_channels = [canonicalize_channel_name(c) for c in signal_channels]
    target_channels = [canonicalize_channel_name(c) for c in target_channels]
    idx = {}
    for i, c in enumerate(signal_channels):
        idx.setdefault(c, i)
    out = []
    for ch in target_channels:
        if ch in idx:
            out.append(signal[idx[ch]])
            continue
        if '-' in ch:
            anode, cathode = [canonicalize_channel_name(part) for part in ch.split('-', 1)]
            if anode in idx and cathode in idx:
                out.append(signal[idx[anode]] - signal[idx[cathode]])
                continue
        raise ValueError(f'Missing required channel or derivable montage {ch}. Available={signal_channels}')
    return np.stack(out, axis=0)
