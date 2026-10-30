from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit


def probabilities_to_logits(probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(probabilities / (1.0 - probabilities))


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return expit(values)


def fit_binary_temperature(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=np.float64)
    logits = probabilities_to_logits(probabilities)

    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, scaled) - y_true * scaled))

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method='bounded')
    return float(np.exp(result.x))


def apply_temperature(probabilities, temperature: float):
    return sigmoid(probabilities_to_logits(probabilities) / max(float(temperature), 1e-6))


def calibration_metrics(y_true, probabilities, bins: int = 15):
    y_true = np.asarray(y_true, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((probabilities - y_true) ** 2))
    nll = float(-np.mean(y_true * np.log(probabilities) + (1.0 - y_true) * np.log(1.0 - probabilities)))
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    rows = []
    for idx in range(int(bins)):
        keep = (probabilities >= edges[idx]) & (probabilities < edges[idx + 1] if idx < bins - 1 else probabilities <= edges[idx + 1])
        if not keep.any():
            continue
        confidence = float(probabilities[keep].mean())
        observed = float(y_true[keep].mean())
        weight = float(keep.mean())
        ece += weight * abs(confidence - observed)
        rows.append({'bin_left': edges[idx], 'bin_right': edges[idx + 1], 'n': int(keep.sum()), 'mean_probability': confidence, 'observed_rate': observed})
    return {'brier': brier, 'nll': nll, 'ece': float(ece)}, pd.DataFrame(rows)


def calibrate_prediction_frame(predictions: pd.DataFrame, temperature: float):
    out = predictions.copy()
    out['det_prob_uncalibrated'] = out['det_prob'].astype(float)
    out['det_prob'] = apply_temperature(out['det_prob_uncalibrated'], temperature)
    return out
