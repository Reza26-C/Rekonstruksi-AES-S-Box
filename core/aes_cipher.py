"""AES-128 cipher parameterized by custom S-Box + image encryption security metrics."""

from __future__ import annotations

import base64
import numpy as np

from .sbox_gen import construct_inverse_sbox

_RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def _aes_mix_mul(a: int, b: int) -> int:
    """Byte multiplication helper for AES MixColumns and InvMixColumns diffusion."""
    p = 0
    a &= 0xFF
    b &= 0xFF
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B
        b >>= 1
    return p


_MUL2 = np.array([_aes_mix_mul(x, 2) for x in range(256)], dtype=np.uint8)
_MUL3 = np.array([_aes_mix_mul(x, 3) for x in range(256)], dtype=np.uint8)
_MUL9 = np.array([_aes_mix_mul(x, 9) for x in range(256)], dtype=np.uint8)
_MUL11 = np.array([_aes_mix_mul(x, 11) for x in range(256)], dtype=np.uint8)
_MUL13 = np.array([_aes_mix_mul(x, 13) for x in range(256)], dtype=np.uint8)
_MUL14 = np.array([_aes_mix_mul(x, 14) for x in range(256)], dtype=np.uint8)


def normalize_key_iv(text_or_bytes: str | bytes, length: int = 16) -> bytes:
    """Normalize a user key or IV string/bytes to exact 16-byte AES-128 length."""
    raw = text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else bytes(text_or_bytes)
    if len(raw) < length:
        return raw.ljust(length, b"\x00")
    return raw[:length]


def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    """Apply PKCS#7 padding to byte data."""
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)


def pkcs7_unpad(data: bytes, block_size: int = 16) -> bytes:
    """Remove and validate PKCS#7 padding from byte data."""
    if not data or len(data) % block_size != 0:
        raise ValueError("Panjang data dekripsi tidak valid (bukan kelipatan 16 byte).")
    pad_len = data[-1]
    if pad_len < 1 or pad_len > block_size or data[-pad_len:] != bytes([pad_len] * pad_len):
        raise ValueError("Padding PKCS#7 tidak valid (periksa kecocokan Kunci, IV, Mode, atau S-Box).")
    return data[:-pad_len]


def key_expansion(key: bytes, sbox: np.ndarray) -> np.ndarray:
    """Expand a 16-byte key into (11, 4, 4) AES-128 round keys using the provided S-Box."""
    if len(key) != 16:
        raise ValueError("Kunci AES-128 harus tepat 16 byte.")
    sb = np.asarray(sbox, dtype=np.uint8)
    words = np.zeros((44, 4), dtype=np.uint8)
    for i in range(4):
        words[i] = list(key[4 * i : 4 * i + 4])

    for i in range(4, 44):
        temp = words[i - 1].copy()
        if i % 4 == 0:
            temp = np.array(
                [sb[temp[1]] ^ _RCON[(i // 4) - 1], sb[temp[2]], sb[temp[3]], sb[temp[0]]],
                dtype=np.uint8,
            )
        words[i] = words[i - 4] ^ temp

    # Round r uses words[4r..4r+3] as columns 0..3 -> transpose to (4 rows, 4 cols)
    return words.reshape(11, 4, 4).transpose(0, 2, 1).copy()


def _encrypt_states_batch(blocks_4x4: np.ndarray, round_keys: np.ndarray, sbox: np.ndarray) -> np.ndarray:
    """Vectorized AES-128 encryption over (N, 4, 4) state blocks."""
    state = blocks_4x4 ^ round_keys[0][None, :, :]
    for rnd in range(1, 10):
        state = sbox[state]
        state[:, 1, :] = state[:, 1, [1, 2, 3, 0]]
        state[:, 2, :] = state[:, 2, [2, 3, 0, 1]]
        state[:, 3, :] = state[:, 3, [3, 0, 1, 2]]
        s0, s1, s2, s3 = state[:, 0, :], state[:, 1, :], state[:, 2, :], state[:, 3, :]
        mixed = np.empty_like(state)
        mixed[:, 0, :] = _MUL2[s0] ^ _MUL3[s1] ^ s2 ^ s3
        mixed[:, 1, :] = s0 ^ _MUL2[s1] ^ _MUL3[s2] ^ s3
        mixed[:, 2, :] = s0 ^ s1 ^ _MUL2[s2] ^ _MUL3[s3]
        mixed[:, 3, :] = _MUL3[s0] ^ s1 ^ s2 ^ _MUL2[s3]
        state = mixed ^ round_keys[rnd][None, :, :]

    state = sbox[state]
    state[:, 1, :] = state[:, 1, [1, 2, 3, 0]]
    state[:, 2, :] = state[:, 2, [2, 3, 0, 1]]
    state[:, 3, :] = state[:, 3, [3, 0, 1, 2]]
    return state ^ round_keys[10][None, :, :]


def _decrypt_states_batch(
    blocks_4x4: np.ndarray, round_keys: np.ndarray, inv_sbox: np.ndarray
) -> np.ndarray:
    """Vectorized AES-128 decryption over (N, 4, 4) state blocks."""
    state = blocks_4x4 ^ round_keys[10][None, :, :]
    for rnd in range(9, 0, -1):
        state[:, 1, :] = state[:, 1, [3, 0, 1, 2]]
        state[:, 2, :] = state[:, 2, [2, 3, 0, 1]]
        state[:, 3, :] = state[:, 3, [1, 2, 3, 0]]
        state = inv_sbox[state]
        state ^= round_keys[rnd][None, :, :]
        s0, s1, s2, s3 = state[:, 0, :], state[:, 1, :], state[:, 2, :], state[:, 3, :]
        inv_mixed = np.empty_like(state)
        inv_mixed[:, 0, :] = _MUL14[s0] ^ _MUL11[s1] ^ _MUL13[s2] ^ _MUL9[s3]
        inv_mixed[:, 1, :] = _MUL9[s0] ^ _MUL14[s1] ^ _MUL11[s2] ^ _MUL13[s3]
        inv_mixed[:, 2, :] = _MUL13[s0] ^ _MUL9[s1] ^ _MUL14[s2] ^ _MUL11[s3]
        inv_mixed[:, 3, :] = _MUL11[s0] ^ _MUL13[s1] ^ _MUL9[s2] ^ _MUL14[s3]
        state = inv_mixed

    state[:, 1, :] = state[:, 1, [3, 0, 1, 2]]
    state[:, 2, :] = state[:, 2, [2, 3, 0, 1]]
    state[:, 3, :] = state[:, 3, [1, 2, 3, 0]]
    state = inv_sbox[state]
    return state ^ round_keys[0][None, :, :]


def _build_enc_t_tables(sbox: np.ndarray) -> tuple[list[int], list[int], list[int], list[int]]:
    """Build 32-bit T-tables for fast sequential CBC encryption with any custom S-Box."""
    sb = [int(v) for v in sbox]
    m2 = [int(v) for v in _MUL2]
    m3 = [int(v) for v in _MUL3]
    t0, t1, t2, t3 = [0] * 256, [0] * 256, [0] * 256, [0] * 256
    for x in range(256):
        s = sb[x]
        w = (m2[s] << 24) | (s << 16) | (s << 8) | m3[s]
        t0[x] = w
        t1[x] = ((w >> 8) | (w << 24)) & 0xFFFFFFFF
        t2[x] = ((w >> 16) | (w << 16)) & 0xFFFFFFFF
        t3[x] = ((w >> 24) | (w << 8)) & 0xFFFFFFFF
    return t0, t1, t2, t3


def _encrypt_cbc_fast(padded_bytes: bytes, round_keys: np.ndarray, sbox: np.ndarray, iv: bytes) -> bytes:
    """Fast T-table AES-128 CBC encryption for arbitrary byte lengths (multiple of 16)."""
    t0, t1, t2, t3 = _build_enc_t_tables(sbox)
    sb = [int(v) for v in sbox]
    rk_words = [
        [
            (int(round_keys[r, 0, c]) << 24)
            | (int(round_keys[r, 1, c]) << 16)
            | (int(round_keys[r, 2, c]) << 8)
            | int(round_keys[r, 3, c])
            for c in range(4)
        ]
        for r in range(11)
    ]

    p0 = int.from_bytes(iv[0:4], "big")
    p1 = int.from_bytes(iv[4:8], "big")
    p2 = int.from_bytes(iv[8:12], "big")
    p3 = int.from_bytes(iv[12:16], "big")

    rk0, rk1, rk2, rk3, rk4, rk5, rk6, rk7, rk8, rk9, rk10 = rk_words
    out = bytearray(len(padded_bytes))

    for off in range(0, len(padded_bytes), 16):
        s0 = int.from_bytes(padded_bytes[off : off + 4], "big") ^ p0 ^ rk0[0]
        s1 = int.from_bytes(padded_bytes[off + 4 : off + 8], "big") ^ p1 ^ rk0[1]
        s2 = int.from_bytes(padded_bytes[off + 8 : off + 12], "big") ^ p2 ^ rk0[2]
        s3 = int.from_bytes(padded_bytes[off + 12 : off + 16], "big") ^ p3 ^ rk0[3]

        for rk in (rk1, rk2, rk3, rk4, rk5, rk6, rk7, rk8, rk9):
            n0 = t0[(s0 >> 24) & 0xFF] ^ t1[(s1 >> 16) & 0xFF] ^ t2[(s2 >> 8) & 0xFF] ^ t3[s3 & 0xFF] ^ rk[0]
            n1 = t0[(s1 >> 24) & 0xFF] ^ t1[(s2 >> 16) & 0xFF] ^ t2[(s3 >> 8) & 0xFF] ^ t3[s0 & 0xFF] ^ rk[1]
            n2 = t0[(s2 >> 24) & 0xFF] ^ t1[(s3 >> 16) & 0xFF] ^ t2[(s0 >> 8) & 0xFF] ^ t3[s1 & 0xFF] ^ rk[2]
            n3 = t0[(s3 >> 24) & 0xFF] ^ t1[(s0 >> 16) & 0xFF] ^ t2[(s1 >> 8) & 0xFF] ^ t3[s2 & 0xFF] ^ rk[3]
            s0, s1, s2, s3 = n0, n1, n2, n3

        p0 = ((sb[(s0 >> 24) & 0xFF] << 24) | (sb[(s1 >> 16) & 0xFF] << 16) | (sb[(s2 >> 8) & 0xFF] << 8) | sb[s3 & 0xFF]) ^ rk10[0]
        p1 = ((sb[(s1 >> 24) & 0xFF] << 24) | (sb[(s2 >> 16) & 0xFF] << 16) | (sb[(s3 >> 8) & 0xFF] << 8) | sb[s0 & 0xFF]) ^ rk10[1]
        p2 = ((sb[(s2 >> 24) & 0xFF] << 24) | (sb[(s3 >> 16) & 0xFF] << 16) | (sb[(s0 >> 8) & 0xFF] << 8) | sb[s1 & 0xFF]) ^ rk10[2]
        p3 = ((sb[(s3 >> 24) & 0xFF] << 24) | (sb[(s0 >> 16) & 0xFF] << 16) | (sb[(s1 >> 8) & 0xFF] << 8) | sb[s2 & 0xFF]) ^ rk10[3]

        out[off : off + 4] = p0.to_bytes(4, "big")
        out[off + 4 : off + 8] = p1.to_bytes(4, "big")
        out[off + 8 : off + 12] = p2.to_bytes(4, "big")
        out[off + 12 : off + 16] = p3.to_bytes(4, "big")

    return bytes(out)


def encrypt_bytes(
    data: bytes,
    sbox: np.ndarray,
    key: bytes,
    iv: bytes = b"\x00" * 16,
    mode: str = "CBC",
) -> bytes:
    """Encrypt raw bytes using AES-128 with PKCS#7 padding in ECB or CBC mode."""
    sb = np.asarray(sbox, dtype=np.uint8)
    _ = construct_inverse_sbox(sb)  # validate bijectivity
    rk = key_expansion(key, sb)
    padded = pkcs7_pad(data, 16)

    mode_upper = mode.upper()
    if mode_upper == "CBC":
        return _encrypt_cbc_fast(padded, rk, sb, iv)
    if mode_upper == "ECB":
        arr = np.frombuffer(padded, dtype=np.uint8).reshape(-1, 4, 4).transpose(0, 2, 1).copy()
        enc = _encrypt_states_batch(arr, rk, sb)
        return enc.transpose(0, 2, 1).tobytes()
    raise ValueError(f"Mode AES tidak dikenal: {mode}. Gunakan 'CBC' atau 'ECB'.")


def decrypt_bytes(
    ciphertext: bytes,
    sbox: np.ndarray,
    key: bytes,
    iv: bytes = b"\x00" * 16,
    mode: str = "CBC",
) -> bytes:
    """Decrypt raw bytes using AES-128 and remove PKCS#7 padding."""
    if len(ciphertext) == 0 or len(ciphertext) % 16 != 0:
        raise ValueError("Panjang ciphertext harus kelipatan 16 byte.")
    sb = np.asarray(sbox, dtype=np.uint8)
    inv_sb = construct_inverse_sbox(sb)
    rk = key_expansion(key, sb)

    cipher_states = (
        np.frombuffer(ciphertext, dtype=np.uint8).reshape(-1, 4, 4).transpose(0, 2, 1).copy()
    )
    plain_states = _decrypt_states_batch(cipher_states, rk, inv_sb)

    mode_upper = mode.upper()
    if mode_upper == "CBC":
        iv_state = np.frombuffer(iv, dtype=np.uint8).reshape(4, 4).T
        plain_states[0] ^= iv_state
        if plain_states.shape[0] > 1:
            plain_states[1:] ^= cipher_states[:-1]
    elif mode_upper != "ECB":
        raise ValueError(f"Mode AES tidak dikenal: {mode}. Gunakan 'CBC' atau 'ECB'.")

    plain_bytes = plain_states.transpose(0, 2, 1).tobytes()
    return pkcs7_unpad(plain_bytes, 16)


def encrypt_text(
    plaintext: str,
    sbox: np.ndarray,
    key_str: str,
    iv_str: str = "AES-SBOX-ML-2025",
    mode: str = "CBC",
) -> dict:
    """Encrypt UTF-8 text and return hex, base64, and raw ciphertext bytes."""
    key = normalize_key_iv(key_str, 16)
    iv = normalize_key_iv(iv_str, 16)
    ct = encrypt_bytes(plaintext.encode("utf-8"), sbox, key, iv, mode=mode)
    return {
        "ciphertext_bytes": ct,
        "hex": ct.hex(),
        "base64": base64.b64encode(ct).decode("ascii"),
        "key_hex": key.hex(),
        "iv_hex": iv.hex(),
    }


def decrypt_text(
    ciphertext_bytes: bytes,
    sbox: np.ndarray,
    key_str: str,
    iv_str: str = "AES-SBOX-ML-2025",
    mode: str = "CBC",
) -> str:
    """Decrypt ciphertext bytes back to a UTF-8 string."""
    key = normalize_key_iv(key_str, 16)
    iv = normalize_key_iv(iv_str, 16)
    pt = decrypt_bytes(ciphertext_bytes, sbox, key, iv, mode=mode)
    return pt.decode("utf-8")


def compute_text_avalanche(
    plaintext: str,
    sbox: np.ndarray,
    key_str: str,
    iv_str: str = "AES-SBOX-ML-2025",
    mode: str = "CBC",
) -> dict:
    """Flip 1 LSB in the first byte of plaintext and compute ciphertext bit-change %."""
    key = normalize_key_iv(key_str, 16)
    iv = normalize_key_iv(iv_str, 16)
    raw = plaintext.encode("utf-8") if plaintext else b"A"
    flipped = bytes([raw[0] ^ 1]) + raw[1:]

    ct1 = encrypt_bytes(raw, sbox, key, iv, mode=mode)
    ct2 = encrypt_bytes(flipped, sbox, key, iv, mode=mode)

    arr1 = np.frombuffer(ct1, dtype=np.uint8)
    arr2 = np.frombuffer(ct2, dtype=np.uint8)
    diff_bits = int(sum(int(v).bit_count() for v in (arr1 ^ arr2)))
    total_bits = len(arr1) * 8
    return {
        "changed_bits": diff_bits,
        "total_bits": total_bits,
        "avalanche_pct": (diff_bits / total_bits) * 100.0 if total_bits else 0.0,
        "ct1_hex": ct1.hex(),
        "ct2_hex": ct2.hex(),
    }


def encrypt_image(
    img_array: np.ndarray,
    sbox: np.ndarray,
    key: bytes,
    iv: bytes = b"\x00" * 16,
    mode: str = "CBC",
) -> dict:
    """Encrypt an image uint8 array and return visual cipher image + full ciphertext bytes."""
    arr = np.asarray(img_array, dtype=np.uint8)
    raw = arr.tobytes()
    orig_len = len(raw)
    cipher_bytes = encrypt_bytes(raw, sbox, key, iv, mode=mode)
    cipher_img = np.frombuffer(cipher_bytes[:orig_len], dtype=np.uint8).reshape(arr.shape).copy()
    return {
        "cipher_img": cipher_img,
        "cipher_bytes": cipher_bytes,
        "orig_shape": arr.shape,
    }


def decrypt_image(
    cipher_bytes: bytes,
    orig_shape: tuple[int, ...],
    sbox: np.ndarray,
    key: bytes,
    iv: bytes = b"\x00" * 16,
    mode: str = "CBC",
) -> np.ndarray:
    """Decrypt image ciphertext bytes and restore exact original shape."""
    plain_bytes = decrypt_bytes(cipher_bytes, sbox, key, iv, mode=mode)
    return np.frombuffer(plain_bytes, dtype=np.uint8).reshape(orig_shape).copy()


def compute_entropy(img_array: np.ndarray) -> float:
    """Compute Shannon entropy (bits per byte) of an 8-bit image."""
    counts = np.bincount(np.asarray(img_array, dtype=np.uint8).ravel(), minlength=256)
    probs = counts[counts > 0] / float(img_array.size)
    return float(-np.sum(probs * np.log2(probs)))


def compute_npcr_uaci(
    img_plain: np.ndarray,
    sbox: np.ndarray,
    key: bytes,
    iv: bytes = b"\x00" * 16,
    mode: str = "CBC",
) -> tuple[float, float]:
    """Compute NPCR (%) and UACI (%) after flipping 1 LSB at pixel (0, 0) of img_plain."""
    arr1 = np.asarray(img_plain, dtype=np.uint8)
    arr2 = arr1.copy()
    arr2.flat[0] ^= 1

    c1 = encrypt_image(arr1, sbox, key, iv, mode=mode)["cipher_img"].astype(np.int32)
    c2 = encrypt_image(arr2, sbox, key, iv, mode=mode)["cipher_img"].astype(np.int32)

    npcr = float(np.mean(c1 != c2) * 100.0)
    uaci = float(np.mean(np.abs(c1 - c2) / 255.0) * 100.0)
    return npcr, uaci


def compute_pixel_correlations(
    img_array: np.ndarray, n_samples: int = 3000, seed: int = 42
) -> dict:
    """Compute Horizontal, Vertical, and Diagonal adjacent-pixel Pearson correlations."""
    arr = np.asarray(img_array, dtype=np.float64)
    gray = arr[:, :, 0] if arr.ndim == 3 else arr
    h, w = gray.shape
    rng = np.random.default_rng(seed)

    def _sample_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        a_flat = a.ravel()
        b_flat = b.ravel()
        k = min(n_samples, len(a_flat))
        idx = rng.choice(len(a_flat), size=k, replace=False)
        xs, ys = a_flat[idx], b_flat[idx]
        if np.std(xs) == 0 or np.std(ys) == 0:
            return 0.0, xs, ys
        return float(np.corrcoef(xs, ys)[0, 1]), xs, ys

    corr_h, hx, hy = _sample_corr(gray[:, :-1], gray[:, 1:])
    corr_v, vx, vy = _sample_corr(gray[:-1, :], gray[1:, :])
    corr_d, dx, dy = _sample_corr(gray[:-1, :-1], gray[1:, 1:])

    return {
        "horizontal": corr_h,
        "vertical": corr_v,
        "diagonal": corr_d,
        "samples": {
            "horizontal": (hx, hy),
            "vertical": (vx, vy),
            "diagonal": (dx, dy),
        },
    }


def generate_sample_images(size: int = 256) -> dict[str, np.ndarray]:
    """Generate 3 built-in 256x256 RGB test images for instant demo and security testing."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size / 2.0, size / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # 1. Geometric Gradient & Concentric Rings
    r1 = np.clip((xx / size) * 255, 0, 255).astype(np.uint8)
    g1 = np.clip((yy / size) * 255, 0, 255).astype(np.uint8)
    b1 = np.clip(127 + 127 * np.sin(dist / 12.0), 0, 255).astype(np.uint8)
    circle_mask = dist < (size * 0.25)
    r1[circle_mask] = 240
    g1[circle_mask] = 80
    b1[circle_mask] = 60
    img_geo = np.stack([r1, g1, b1], axis=-1)

    # 2. Wave Interference Pattern
    r2 = np.clip(128 + 120 * np.sin(xx / 18.0) * np.cos(yy / 18.0), 0, 255).astype(np.uint8)
    g2 = np.clip(128 + 120 * np.sin((xx + yy) / 22.0), 0, 255).astype(np.uint8)
    b2 = np.clip(128 + 120 * np.cos((xx - yy) / 16.0), 0, 255).astype(np.uint8)
    img_wave = np.stack([r2, g2, b2], axis=-1)

    # 3. Structured Checkerboard + Smooth Sky
    tile = ((xx // 32 + yy // 32) % 2) == 0
    r3 = np.where(tile, 35, 225).astype(np.uint8)
    g3 = np.where(tile, 110, 200).astype(np.uint8)
    b3 = np.clip(80 + (yy / size) * 160, 0, 255).astype(np.uint8)
    img_checker = np.stack([r3, g3, b3], axis=-1)

    return {
        "Geometri & Gradien (256x256 RGB)": img_geo,
        "Pola Gelombang Interferensi (256x256 RGB)": img_wave,
        "Papan Catur Terstruktur (256x256 RGB)": img_checker,
    }
