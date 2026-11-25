"""CSV and multi-sheet Excel export utilities for S-Box tables and evaluation reports."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd


def sbox_to_16x16_dataframe(sbox: np.ndarray, hex_mode: bool = False) -> pd.DataFrame:
    """Format a 256-entry S-Box into a 16x16 DataFrame indexed by hex nibbles 0..F."""
    sb = np.asarray(sbox, dtype=np.uint8).reshape(16, 16)
    labels = [f"{i:X}" for i in range(16)]
    if hex_mode:
        data = [[f"{int(val):02X}" for val in row] for row in sb]
    else:
        data = sb.astype(int)
    df = pd.DataFrame(data, index=labels, columns=labels)
    df.index.name = "Baris\\Kolom"
    return df


def export_sbox_csv(sbox: np.ndarray, hex_mode: bool = False) -> bytes:
    """Export a 16x16 S-Box table as UTF-8 CSV bytes."""
    df = sbox_to_16x16_dataframe(sbox, hex_mode=hex_mode)
    return df.to_csv(index=True).encode("utf-8")


def export_comparison_csv(comparison_df: pd.DataFrame) -> bytes:
    """Export the cryptographic comparison table as UTF-8 CSV bytes."""
    return comparison_df.to_csv(index=False).encode("utf-8")


def export_full_report_excel(
    active_name: str,
    active_sbox: np.ndarray,
    active_eval: dict,
    comparison_df: pd.DataFrame,
    active_params: dict | None = None,
    top_candidates_df: pd.DataFrame | None = None,
) -> bytes:
    """Export S-Box tables, construction parameters, cryptographic tests, and comparisons to .xlsx bytes."""
    from .sbox_gen import construct_inverse_sbox

    buf = io.BytesIO()
    inv_sbox = construct_inverse_sbox(active_sbox)
    if active_params:
        params_df = pd.DataFrame(
            [{"Parameter": str(k), "Nilai": v} for k, v in active_params.items()]
        )
    else:
        params_df = pd.DataFrame([{"Parameter": "Konstruksi", "Nilai": "Standar AES"}])

    bit_labels = [f"b{i}" for i in range(7, -1, -1)]
    sac_df = pd.DataFrame(active_eval["sac_matrix"], index=bit_labels, columns=bit_labels)
    bicsac_df = pd.DataFrame(active_eval["bic_sac_matrix"], index=bit_labels, columns=bit_labels)
    balance_df = pd.DataFrame(active_eval["balance_stats"])

    info_df = pd.DataFrame(
        [
            {"Parameter": "Nama S-Box Aktif", "Nilai": active_name},
            {"Parameter": "Valid (Balanced & Bijective)", "Nilai": active_eval["is_valid"]},
            {"Parameter": "Nonlinearity (NL)", "Nilai": active_eval["nl"]},
            {"Parameter": "SAC", "Nilai": round(active_eval["sac"], 6)},
            {"Parameter": "BIC-NL", "Nilai": active_eval["bic_nl"]},
            {"Parameter": "BIC-SAC", "Nilai": round(active_eval["bic_sac"], 6)},
            {"Parameter": "LAP", "Nilai": active_eval["lap"]},
            {"Parameter": "DAP", "Nilai": active_eval["dap"]},
            {"Parameter": "Skor S", "Nilai": round(active_eval["S"], 6)},
            {"Parameter": "Strength Value (SV)", "Nilai": round(active_eval["SV"], 6)},
        ]
    )

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        info_df.to_excel(writer, sheet_name="Ringkasan_SBox_Aktif", index=False)
        comparison_df.to_excel(writer, sheet_name="Ringkasan_Komparasi", index=False)
        sbox_to_16x16_dataframe(active_sbox, hex_mode=False).to_excel(
            writer, sheet_name="SBox_16x16_Desimal"
        )
        sbox_to_16x16_dataframe(active_sbox, hex_mode=True).to_excel(
            writer, sheet_name="SBox_16x16_Hex"
        )
        sbox_to_16x16_dataframe(inv_sbox, hex_mode=False).to_excel(
            writer, sheet_name="Inverse_SBox_16x16"
        )
        params_df.to_excel(writer, sheet_name="Parameter_Konstruksi", index=False)
        sac_df.to_excel(writer, sheet_name="Matriks_SAC_8x8")
        bicsac_df.to_excel(writer, sheet_name="Matriks_BICSAC_8x8")
        balance_df.to_excel(writer, sheet_name="Uji_Balance_Bit", index=False)
        if top_candidates_df is not None and not top_candidates_df.empty:
            top_candidates_df.to_excel(writer, sheet_name="Top_Kandidat_ML", index=False)

    buf.seek(0)
    return buf.getvalue()
