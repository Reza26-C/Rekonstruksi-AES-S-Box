"""Gradient Boosting surrogate modeling for non-polynomial Chaotic S-Box reconstruction."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from .metrics import _fwht_batch, evaluate_sbox_full
from .sbox_gen import generate_chaotic_sboxes_batch, refine_sbox_swaps

_X_ALL = np.arange(256, dtype=np.uint8)
_DX_POW2 = np.array([1 << i for i in range(8)], dtype=np.uint8)
_POPCOUNT_LUT = np.array([x.bit_count() for x in range(256)], dtype=np.uint8)
_BIT_SHIFTS = np.arange(8, dtype=np.uint8)

FEATURE_NAMES: list[str] = [
    "x0",
    "y0",
    "r",
    "alpha",
    "coord_nl_min",
    "coord_nl_mean",
    "coord_nl_std",
    "coord_nl_max",
    "walsh_l4_norm",
    "sac_mean",
    "sac_std",
    "sac_abs_err",
    "adj_diff_mean",
    "adj_diff_std",
    "fixed_points",
    "autocorr_lag1",
]


def extract_sbox_features(
    sboxes: np.ndarray, params: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Extract 16 fast spectral, chaotic-parameter, and permutation features from (N, 256) S-Boxes."""
    sbs = np.asarray(sboxes, dtype=np.uint8)
    prm = np.asarray(params, dtype=np.float64)
    if sbs.ndim == 1:
        sbs = sbs[None, :]
        prm = prm[None, :]

    bits = ((sbs[:, None, :] >> _BIT_SHIFTS[None, :, None]) & 1).astype(np.int32)
    w = _fwht_batch(1 - 2 * bits)
    nls = 128.0 - (np.max(np.abs(w), axis=2) / 2.0)
    walsh_l4 = np.mean(w.astype(np.float64) ** 4, axis=(1, 2)) / 1e6

    idx_flip = _X_ALL[None, :] ^ _DX_POW2[:, None]
    diffs = sbs[:, _X_ALL[None, :]] ^ sbs[:, idx_flip]
    sac_per_i = np.mean(_POPCOUNT_LUT[diffs], axis=2) / 8.0
    sac_mean = np.mean(sac_per_i, axis=1)
    sac_std = np.std(sac_per_i, axis=1)
    sac_abs_err = np.abs(sac_mean - 0.5)

    sb_f = sbs.astype(np.float64)
    adj_diff = np.abs(np.diff(sb_f, axis=1))
    adj_diff_mean = np.mean(adj_diff, axis=1)
    adj_diff_std = np.std(adj_diff, axis=1)
    fixed_pts = np.sum(sbs == _X_ALL[None, :], axis=1).astype(np.float64)

    a = sb_f[:, :-1] - 127.5
    b = sb_f[:, 1:] - 127.5
    autocorr = np.sum(a * b, axis=1) / (
        np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)) + 1e-12
    )

    features = np.column_stack(
        [
            prm[:, 0],
            prm[:, 1],
            prm[:, 2],
            prm[:, 3],
            np.min(nls, axis=1),
            np.mean(nls, axis=1),
            np.std(nls, axis=1),
            np.max(nls, axis=1),
            walsh_l4,
            sac_mean,
            sac_std,
            sac_abs_err,
            adj_diff_mean,
            adj_diff_std,
            fixed_pts,
            autocorr,
        ]
    )
    return features, list(FEATURE_NAMES)


def _sample_chaotic_params(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample (n, 4) parameters [x0, y0, r, alpha] inside chaotic regime."""
    return np.column_stack(
        [
            rng.uniform(0.05, 0.95, size=n),
            rng.uniform(0.05, 0.95, size=n),
            rng.uniform(3.57, 3.99, size=n),
            rng.uniform(0.20, 0.95, size=n),
        ]
    )


def run_ml_sbox_pipeline(
    n_train_samples: int = 400,
    n_candidates: int = 1500,
    n_opt_steps: int = 800,
    n_estimators: int = 180,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    subsample: float = 0.85,
    test_size: float = 0.2,
    top_k_verify: int = 10,
    seed: int = 42,
) -> dict:
    """Train GradientBoostingRegressor on chaotic S-Boxes, screen candidates, and refine winner."""
    t_pipeline_start = time.perf_counter()
    rng = np.random.default_rng(seed)

    train_params = _sample_chaotic_params(n_train_samples, rng)
    train_sboxes = generate_chaotic_sboxes_batch(train_params)

    # Diversify 35% of training samples with partial swap refinement
    for idx in range(n_train_samples):
        if rng.random() < 0.35:
            steps = int(rng.integers(40, 160))
            train_sboxes[idx] = refine_sbox_swaps(
                train_sboxes[idx], n_steps=steps, seed=seed + idx
            )

    x_data, feature_names = extract_sbox_features(train_sboxes, train_params)
    y_data = np.array(
        [evaluate_sbox_full(train_sboxes[idx])["SV"] for idx in range(n_train_samples)],
        dtype=np.float64,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size, random_state=seed
    )

    t_train_start = time.perf_counter()
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    train_time_sec = time.perf_counter() - t_train_start

    y_pred_test = model.predict(x_test)
    r2 = float(r2_score(y_test, y_pred_test))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    rmse = float(root_mean_squared_error(y_test, y_pred_test))

    importance_df = (
        pd.DataFrame(
            {
                "Fitur": feature_names,
                "Importance": model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    t_gen_start = time.perf_counter()
    cand_rng = np.random.default_rng(seed + 1000)
    cand_params = _sample_chaotic_params(n_candidates, cand_rng)
    cand_sboxes = generate_chaotic_sboxes_batch(cand_params)
    gen_time_sec = time.perf_counter() - t_gen_start

    t_screen_start = time.perf_counter()
    cand_features, _ = extract_sbox_features(cand_sboxes, cand_params)
    cand_pred_sv = model.predict(cand_features)
    ml_screening_time_sec = time.perf_counter() - t_screen_start

    top_k = min(top_k_verify, n_candidates)
    top_indices = np.argsort(cand_pred_sv)[:top_k]

    t_verify_start = time.perf_counter()
    verified_records = []
    verified_objects = []
    for rank_idx, cand_idx in enumerate(top_indices, start=1):
        sb = cand_sboxes[cand_idx]
        prm = cand_params[cand_idx]
        ev = evaluate_sbox_full(sb, name=f"Kandidat-GBR-{rank_idx}")
        verified_records.append(
            {
                "Peringkat": rank_idx,
                "x0": round(float(prm[0]), 4),
                "y0": round(float(prm[1]), 4),
                "r": round(float(prm[2]), 4),
                "alpha": round(float(prm[3]), 4),
                "Prediksi SV (GBR)": float(cand_pred_sv[cand_idx]),
                "SV Aktual": float(ev["SV"]),
                "NL": int(ev["nl"]),
                "SAC": float(ev["sac"]),
                "BIC-NL": int(ev["bic_nl"]),
                "BIC-SAC": float(ev["bic_sac"]),
                "LAP": float(ev["lap"]),
                "DAP": float(ev["dap"]),
                "Valid": bool(ev["is_valid"]),
            }
        )
        verified_objects.append((ev["SV"], -ev["nl"], prm, sb, ev))

    exact_verify_time_sec = time.perf_counter() - t_verify_start
    avg_full_eval_sec = exact_verify_time_sec / max(1, top_k)
    estimated_bruteforce_time_sec = avg_full_eval_sec * n_candidates
    speedup_factor = (
        estimated_bruteforce_time_sec / ml_screening_time_sec
        if ml_screening_time_sec > 0
        else 1.0
    )

    verified_objects.sort(key=lambda item: (item[0], item[1]))
    _, _, winner_prm, winner_raw_sb, _ = verified_objects[0]

    best_sbox = refine_sbox_swaps(winner_raw_sb, n_steps=n_opt_steps, seed=seed)
    best_eval = evaluate_sbox_full(best_sbox, name="S-Box Chaotic + GBR")
    best_params = {
        "x0": float(winner_prm[0]),
        "y0": float(winner_prm[1]),
        "r": float(winner_prm[2]),
        "alpha": float(winner_prm[3]),
        "n_opt_steps": int(n_opt_steps),
    }

    top_candidates_df = (
        pd.DataFrame(verified_records)
        .sort_values(["SV Aktual", "NL"], ascending=[True, False])
        .reset_index(drop=True)
    )
    top_candidates_df["Peringkat"] = np.arange(1, len(top_candidates_df) + 1)

    total_pipeline_time_sec = time.perf_counter() - t_pipeline_start

    return {
        "model": model,
        "best_params": best_params,
        "best_sbox": best_sbox,
        "best_eval": best_eval,
        "top_candidates_df": top_candidates_df,
        "metrics": {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "n_candidates": int(n_candidates),
            "top_k_verify": int(top_k),
            "train_time_sec": float(train_time_sec),
            "gen_time_sec": float(gen_time_sec),
            "ml_screening_time_sec": float(ml_screening_time_sec),
            "exact_verify_time_sec": float(exact_verify_time_sec),
            "estimated_bruteforce_time_sec": float(estimated_bruteforce_time_sec),
            "speedup_factor": float(speedup_factor),
            "total_pipeline_time_sec": float(total_pipeline_time_sec),
        },
        "test_actual": y_test,
        "test_pred": y_pred_test,
        "importance_df": importance_df,
    }
