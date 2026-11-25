"""Cryptographic strength evaluation metrics for 8x8 S-Boxes.

Includes Balance, Bijectivity, Nonlinearity (NL), Strict Avalanche Criterion (SAC),
Bit Independence Criterion (BIC-NL, BIC-SAC), Linear Approximation Probability (LAP),
Differential Approximation Probability (DAP), composite score S, and Strength Value (SV).
"""

from __future__ import annotations

import numpy as np

_PARITY_LUT = np.array([x.bit_count() & 1 for x in range(256)], dtype=np.uint8)
_X_ALL = np.arange(256, dtype=np.uint8)
_MASKS_NONZERO = np.arange(1, 256, dtype=np.uint8)
_DX_POW2 = np.array([1 << i for i in range(8)], dtype=np.uint8)


def _fwht_batch(arr: np.ndarray) -> np.ndarray:
    """Vectorized Fast Walsh-Hadamard Transform along the last axis (length 256)."""
    w = np.asarray(arr, dtype=np.int32).copy()
    orig_shape = w.shape
    h = 1
    while h < 256:
        w = w.reshape(*orig_shape[:-1], -1, 2 * h)
        a = w[..., :h].copy()
        b = w[..., h:].copy()
        w[..., :h] = a + b
        w[..., h:] = a - b
        h *= 2
    return w.reshape(orig_shape)


def check_balance(sbox: np.ndarray) -> dict:
    """Evaluate output-bit balance across coordinate Boolean functions f(0)..f(7)."""
    sb = np.asarray(sbox, dtype=np.uint8)
    bit_stats = []
    for table_idx, bit_pos in enumerate(range(7, -1, -1)):
        ones = int(np.sum((sb >> bit_pos) & 1))
        zeros = 256 - ones
        bit_stats.append(
            {
                "Fungsi": f"f({table_idx})",
                "Bit": f"b{bit_pos}",
                "Jumlah 0": zeros,
                "Jumlah 1": ones,
                "Balanced": zeros == 128 and ones == 128,
            }
        )
    return {
        "is_balanced": all(item["Balanced"] for item in bit_stats),
        "bit_stats": bit_stats,
    }


def check_bijective(sbox: np.ndarray) -> dict:
    """Check bijectivity (all 256 outputs unique in 0..255) and fixed points."""
    sb = np.asarray(sbox)
    in_range = bool(sb.shape == (256,) and np.all((sb >= 0) & (sb <= 255)))
    unique_count = int(len(np.unique(sb))) if sb.ndim == 1 else 0
    is_bij = in_range and unique_count == 256
    fixed_pts = int(np.sum(sb == np.arange(256))) if sb.shape == (256,) else -1
    opp_fixed_pts = int(np.sum(sb == (255 - np.arange(256)))) if sb.shape == (256,) else -1
    return {
        "is_bijective": is_bij,
        "unique_count": unique_count,
        "fixed_points": fixed_pts,
        "opposite_fixed_points": opp_fixed_pts,
    }


def compute_nl(sbox: np.ndarray) -> dict:
    """Compute Nonlinearity (NL) of all 8 coordinate Boolean functions via FWHT."""
    sb = np.asarray(sbox, dtype=np.uint8)
    bits = np.array([(sb >> b) & 1 for b in range(8)], dtype=np.int32)
    w = _fwht_batch(1 - 2 * bits)
    per_bit = (128 - np.max(np.abs(w), axis=1) // 2).tolist()
    return {
        "nl": int(min(per_bit)),
        "nl_mean": float(np.mean(per_bit)),
        "per_bit": [int(v) for v in per_bit],
    }


def compute_sac(sbox: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute mean SAC and 8x8 SAC matrix (MSB-to-LSB orientation)."""
    sb = np.asarray(sbox, dtype=np.uint8)
    diffs = sb[_X_ALL[None, :]] ^ sb[_X_ALL[None, :] ^ _DX_POW2[:, None]]
    sac_lsb = np.array(
        [[np.mean((diffs[i] >> j) & 1) for j in range(8)] for i in range(8)],
        dtype=np.float64,
    )
    sac_display = sac_lsb[::-1, ::-1].copy()
    return float(np.mean(sac_lsb)), sac_display


def compute_bic_nl_and_sac(
    sbox: np.ndarray,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Compute BIC-NL, BIC-SAC, and 8x8 pairwise matrices (MSB-to-LSB orientation)."""
    sb = np.asarray(sbox, dtype=np.uint8)
    diffs = sb[_X_ALL[None, :]] ^ sb[_X_ALL[None, :] ^ _DX_POW2[:, None]]

    bic_nl_mat = np.full((8, 8), np.nan, dtype=np.float64)
    bic_sac_mat = np.full((8, 8), np.nan, dtype=np.float64)

    nl_vals: list[int] = []
    sac_vals: list[float] = []
    for row_idx, bit_j in enumerate(range(7, -1, -1)):
        for col_idx, bit_k in enumerate(range(7, -1, -1)):
            if bit_j == bit_k:
                continue
            pair_tt = ((sb >> bit_j) ^ (sb >> bit_k)) & 1
            w = _fwht_batch(1 - 2 * pair_tt.astype(np.int32))
            pair_nl = int(128 - np.max(np.abs(w)) // 2)
            bic_nl_mat[row_idx, col_idx] = pair_nl
            nl_vals.append(pair_nl)

            pair_diff = ((diffs >> bit_j) ^ (diffs >> bit_k)) & 1
            pair_sac = float(np.mean(pair_diff))
            bic_sac_mat[row_idx, col_idx] = pair_sac
            sac_vals.append(pair_sac)

    return int(min(nl_vals)), float(np.mean(sac_vals)), bic_nl_mat, bic_sac_mat


def compute_lap(sbox: np.ndarray) -> float:
    """Compute Linear Approximation Probability (LAP) via vectorized FWHT."""
    sb = np.asarray(sbox, dtype=np.uint8)
    masked = np.bitwise_and.outer(_MASKS_NONZERO, sb)
    parities = _PARITY_LUT[masked]
    w = _fwht_batch(1 - 2 * parities.astype(np.int32))
    max_abs_walsh = int(np.max(np.abs(w[:, 1:])))
    return max_abs_walsh / 512.0


def compute_dap(sbox: np.ndarray) -> tuple[float, int]:
    """Compute Differential Approximation Probability (DAP) and Differential Uniformity."""
    sb = np.asarray(sbox, dtype=np.uint8)
    diffs = sb[_X_ALL[None, :]] ^ sb[_X_ALL[None, :] ^ _MASKS_NONZERO[:, None]]
    ddt = np.zeros((255, 256), dtype=np.int32)
    rows = np.arange(255)[:, None]
    np.add.at(ddt, (rows, diffs), 1)
    diff_uniformity = int(np.max(ddt))
    return diff_uniformity / 256.0, diff_uniformity


def evaluate_sbox_full(sbox: np.ndarray, name: str = "Custom") -> dict:
    """Run complete cryptographic evaluation on a 256-entry S-Box."""
    sb = np.asarray(sbox, dtype=np.uint8)
    bal = check_balance(sb)
    bij = check_bijective(sb)
    nl_res = compute_nl(sb)
    sac, sac_matrix = compute_sac(sb)
    bic_nl, bic_sac, bic_nl_matrix, bic_sac_matrix = compute_bic_nl_and_sac(sb)
    lap = compute_lap(sb)
    dap, diff_uniformity = compute_dap(sb)

    nl = nl_res["nl"]
    s_score = (abs(sac - 0.5) + abs(bic_sac - 0.5)) / 2.0
    sv_score = (120.0 - nl) + abs(0.5 - sac) + (120.0 - bic_nl) + abs(0.5 - bic_sac)

    return {
        "name": name,
        "is_valid": bool(bal["is_balanced"] and bij["is_bijective"]),
        "is_balanced": bal["is_balanced"],
        "balance_stats": bal["bit_stats"],
        "is_bijective": bij["is_bijective"],
        "unique_count": bij["unique_count"],
        "fixed_points": bij["fixed_points"],
        "opposite_fixed_points": bij["opposite_fixed_points"],
        "nl": nl,
        "nl_mean": nl_res["nl_mean"],
        "nl_per_bit": nl_res["per_bit"],
        "sac": sac,
        "sac_diff": abs(sac - 0.5),
        "sac_matrix": sac_matrix,
        "bic_nl": bic_nl,
        "bic_sac": bic_sac,
        "bic_sac_diff": abs(bic_sac - 0.5),
        "bic_nl_matrix": bic_nl_matrix,
        "bic_sac_matrix": bic_sac_matrix,
        "lap": lap,
        "dap": dap,
        "diff_uniformity": diff_uniformity,
        "S": s_score,
        "SV": sv_score,
    }


def compare_with_aes(metrics_p: dict, metrics_aes: dict) -> dict:
    """Compute proximity improvements relative to AES Standard."""
    err_sac_aes = abs(metrics_aes["sac"] - 0.5)
    err_sac_p = abs(metrics_p["sac"] - 0.5)
    sac_prox_imp = ((err_sac_aes - err_sac_p) / err_sac_aes) * 100.0 if err_sac_aes else 0.0

    err_bsac_aes = abs(metrics_aes["bic_sac"] - 0.5)
    err_bsac_p = abs(metrics_p["bic_sac"] - 0.5)
    bsac_prox_imp = ((err_bsac_aes - err_bsac_p) / err_bsac_aes) * 100.0 if err_bsac_aes else 0.0

    s_aes = metrics_aes["S"]
    s_p = metrics_p["S"]
    s_prox_imp = ((s_aes - s_p) / s_aes) * 100.0 if s_aes else 0.0

    return {
        "SAC_proximity_imp_pct": float(sac_prox_imp),
        "BICSAC_proximity_imp_pct": float(bsac_prox_imp),
        "S_score_imp_pct": float(s_prox_imp),
    }
