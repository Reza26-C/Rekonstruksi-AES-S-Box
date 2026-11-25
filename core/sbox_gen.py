"""Non-polynomial S-Box generation via 2D Logistic-Sine Chaotic Map and swap optimization."""

from __future__ import annotations

import numpy as np

from .metrics import _fwht_batch

# Static AES standard S-Box lookup table (256-entry) for benchmark comparison and cipher verification
AES_STANDARD_SBOX = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    ],
    dtype=np.uint8,
)


def generate_chaotic_sbox(
    x0: float,
    y0: float,
    r: float,
    alpha: float,
    burn_in: int = 40,
) -> np.ndarray:
    """Generate a bijective 256-entry S-Box from the 2D Logistic-Sine chaotic map."""
    x = float(x0) % 1.0
    y = float(y0) % 1.0
    r_val = float(r)
    a_val = float(alpha)

    for _ in range(burn_in):
        x = (r_val * x * (1.0 - x) + (4.0 - r_val) * np.sin(np.pi * y) / 4.0) % 1.0
        y = (a_val * np.sin(np.pi * x) + (1.0 - a_val) * 4.0 * y * (1.0 - y)) % 1.0

    seq = np.empty(256, dtype=np.float64)
    for idx in range(256):
        x = (r_val * x * (1.0 - x) + (4.0 - r_val) * np.sin(np.pi * y) / 4.0) % 1.0
        y = (a_val * np.sin(np.pi * x) + (1.0 - a_val) * 4.0 * y * (1.0 - y)) % 1.0
        seq[idx] = x + 0.5 * y

    return np.argsort(seq, kind="mergesort").astype(np.uint8)


def generate_chaotic_sboxes_batch(params: np.ndarray, burn_in: int = 40) -> np.ndarray:
    """Vectorized generation of N bijective S-Boxes from (N, 4) [x0, y0, r, alpha] parameters."""
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 4:
        raise ValueError(f"params must have shape (N, 4), got {p.shape}")

    n = p.shape[0]
    x = p[:, 0].copy() % 1.0
    y = p[:, 1].copy() % 1.0
    r_val = p[:, 2]
    a_val = p[:, 3]

    for _ in range(burn_in):
        x = (r_val * x * (1.0 - x) + (4.0 - r_val) * np.sin(np.pi * y) / 4.0) % 1.0
        y = (a_val * np.sin(np.pi * x) + (1.0 - a_val) * 4.0 * y * (1.0 - y)) % 1.0

    seq = np.empty((n, 256), dtype=np.float64)
    for idx in range(256):
        x = (r_val * x * (1.0 - x) + (4.0 - r_val) * np.sin(np.pi * y) / 4.0) % 1.0
        y = (a_val * np.sin(np.pi * x) + (1.0 - a_val) * 4.0 * y * (1.0 - y)) % 1.0
        seq[:, idx] = x + 0.5 * y

    return np.argsort(seq, axis=1, kind="mergesort").astype(np.uint8)


def _swap_objective(sbox: np.ndarray) -> float:
    """Compute smooth Walsh-spectrum objective prioritizing min NL, mean NL, and L4 flatness."""
    bits = np.array([(sbox >> b) & 1 for b in range(8)], dtype=np.int32)
    w = _fwht_batch(1 - 2 * bits)
    nls = 128 - np.max(np.abs(w), axis=1) // 2
    l4_penalty = float(np.sum(w.astype(np.float64) ** 4)) / (8.0 * 65536.0)
    return float(np.min(nls)) + 0.05 * float(np.mean(nls)) - 1e-6 * l4_penalty


def refine_sbox_swaps(sbox: np.ndarray, n_steps: int = 800, seed: int = 42) -> np.ndarray:
    """Apply bijective 2-swap hill-climbing to maximize nonlinearity while keeping balance."""
    cur = np.asarray(sbox, dtype=np.uint8).copy()
    if n_steps <= 0:
        return cur

    rng = np.random.default_rng(seed)
    best_obj = _swap_objective(cur)
    for _ in range(n_steps):
        i, j = rng.integers(0, 256, size=2)
        if i == j:
            continue
        cur[i], cur[j] = cur[j], cur[i]
        obj = _swap_objective(cur)
        if obj >= best_obj:
            best_obj = obj
        else:
            cur[i], cur[j] = cur[j], cur[i]
    return cur


def construct_inverse_sbox(sbox: np.ndarray) -> np.ndarray:
    """Construct the inverse S-Box for a bijective 256-entry S-Box."""
    sb = np.asarray(sbox, dtype=np.uint8)
    if sb.shape != (256,) or len(np.unique(sb)) != 256:
        raise ValueError("S-Box must be a bijective permutation of 0..255")
    inv_sb = np.zeros(256, dtype=np.uint8)
    inv_sb[sb] = np.arange(256, dtype=np.uint8)
    return inv_sb
