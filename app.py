"""Aplikasi Streamlit: Rekonstruksi AES S-Box — Chaotic Map + Gradient Boosting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from core.aes_cipher import (
    compute_entropy,
    compute_npcr_uaci,
    compute_pixel_correlations,
    compute_text_avalanche,
    decrypt_bytes,
    decrypt_image,
    encrypt_image,
    encrypt_text,
    generate_sample_images,
    normalize_key_iv,
)
from core.exporter import (
    export_comparison_csv,
    export_full_report_excel,
    export_sbox_csv,
    sbox_to_16x16_dataframe,
)
from core.metrics import compare_with_aes, evaluate_sbox_full
from core.ml_sbox import run_ml_sbox_pipeline
from core.sbox_gen import (
    AES_STANDARD_SBOX,
    construct_inverse_sbox,
    generate_chaotic_sbox,
    refine_sbox_swaps,
)

st.set_page_config(
    page_title="Rekonstruksi AES S-Box | Chaotic + Gradient Boosting",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _get_aes_benchmark() -> dict:
    """Evaluasi lookup S-Box AES standar sebagai pembanding."""
    return {
        "sbox": AES_STANDARD_SBOX,
        "eval": evaluate_sbox_full(AES_STANDARD_SBOX, name="AES Standar"),
    }


@st.cache_data(show_spinner="Menjalankan Chaotic Map + Gradient Boosting...")
def _cached_ml_pipeline(
    n_train_samples: int,
    n_candidates: int,
    n_opt_steps: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    test_size: float,
    top_k_verify: int,
    seed: int,
) -> dict:
    return run_ml_sbox_pipeline(
        n_train_samples=n_train_samples,
        n_candidates=n_candidates,
        n_opt_steps=n_opt_steps,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        test_size=test_size,
        top_k_verify=top_k_verify,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def _get_sample_images() -> dict[str, np.ndarray]:
    return generate_sample_images(size=256)


def _build_comparison_dataframe(
    aes_bench: dict, ml_res: dict, active_name: str, active_eval: dict
) -> pd.DataFrame:
    aes_ev = aes_bench["eval"]
    ml_ev = ml_res["best_eval"]
    rows = [
        {
            "Nama S-Box": "AES Standar",
            "NL": aes_ev["nl"],
            "SAC": round(aes_ev["sac"], 5),
            "BIC-NL": aes_ev["bic_nl"],
            "BIC-SAC": round(aes_ev["bic_sac"], 5),
            "LAP": aes_ev["lap"],
            "DAP": aes_ev["dap"],
            "Skor S": round(aes_ev["S"], 5),
            "SV": round(aes_ev["SV"], 5),
            "Valid": aes_ev["is_valid"],
        },
        {
            "Nama S-Box": "S-Box Chaotic + GBR",
            "NL": ml_ev["nl"],
            "SAC": round(ml_ev["sac"], 5),
            "BIC-NL": ml_ev["bic_nl"],
            "BIC-SAC": round(ml_ev["bic_sac"], 5),
            "LAP": ml_ev["lap"],
            "DAP": ml_ev["dap"],
            "Skor S": round(ml_ev["S"], 5),
            "SV": round(ml_ev["SV"], 5),
            "Valid": ml_ev["is_valid"],
        },
    ]

    if active_name not in {r["Nama S-Box"] for r in rows}:
        rows.append(
            {
                "Nama S-Box": active_name,
                "NL": active_eval["nl"],
                "SAC": round(active_eval["sac"], 5),
                "BIC-NL": active_eval["bic_nl"],
                "BIC-SAC": round(active_eval["bic_sac"], 5),
                "LAP": active_eval["lap"],
                "DAP": active_eval["dap"],
                "Skor S": round(active_eval["S"], 5),
                "SV": round(active_eval["SV"], 5),
                "Valid": active_eval["is_valid"],
            }
        )
    return pd.DataFrame(rows)


if "ml_params" not in st.session_state:
    st.session_state["ml_params"] = {
        "n_train_samples": 400,
        "n_candidates": 1500,
        "n_opt_steps": 800,
        "n_estimators": 180,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.85,
        "test_size": 0.2,
        "top_k_verify": 10,
        "seed": 42,
    }

aes_bench = _get_aes_benchmark()
ml_res = _cached_ml_pipeline(**st.session_state["ml_params"])

# --- Sidebar ---
st.sidebar.title("🔐 Kontrol S-Box")
sbox_choices = [
    "S-Box Chaotic + GBR",
    "AES Standar",
    "Custom Chaotic",
]
selected_choice = st.sidebar.selectbox("S-Box Aktif", sbox_choices, index=0)

active_params: dict | None = None
if selected_choice == "S-Box Chaotic + GBR":
    active_name = "S-Box Chaotic + GBR"
    active_sbox = ml_res["best_sbox"]
    active_eval = ml_res["best_eval"]
    active_params = ml_res["best_params"]
elif selected_choice == "AES Standar":
    active_name = "AES Standar"
    active_sbox = aes_bench["sbox"]
    active_eval = aes_bench["eval"]
else:
    c_x0 = st.sidebar.slider("x0", 0.01, 0.99, 0.31, 0.01)
    c_y0 = st.sidebar.slider("y0", 0.01, 0.99, 0.72, 0.01)
    c_r = st.sidebar.slider("r", 3.57, 3.99, 3.89, 0.01)
    c_alpha = st.sidebar.slider("alpha", 0.10, 0.95, 0.62, 0.01)
    c_swaps = st.sidebar.slider("Iterasi Swap", 0, 1500, 500, 50)
    raw_sb = generate_chaotic_sbox(c_x0, c_y0, c_r, c_alpha)
    active_name = "Custom Chaotic"
    active_sbox = refine_sbox_swaps(raw_sb, n_steps=c_swaps, seed=42)
    active_eval = evaluate_sbox_full(active_sbox, name=active_name)
    active_params = {
        "x0": c_x0,
        "y0": c_y0,
        "r": c_r,
        "alpha": c_alpha,
        "n_opt_steps": c_swaps,
    }

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Aktif:** `{active_name}`  \n"
    f"**Status:** {'✅ Valid' if active_eval['is_valid'] else '❌ Tidak Valid'}"
)
sb1, sb2 = st.sidebar.columns(2)
sb1.metric("NL", f"{active_eval['nl']}")
sb2.metric("SAC", f"{active_eval['sac']:.5f}")
sb3, sb4 = st.sidebar.columns(2)
sb3.metric("BIC-SAC", f"{active_eval['bic_sac']:.5f}")
sb4.metric("Skor S", f"{active_eval['S']:.5f}")
st.sidebar.metric("SV", f"{active_eval['SV']:.5f}")

comparison_df = _build_comparison_dataframe(aes_bench, ml_res, active_name, active_eval)

# --- Main ---
st.title("🔐 Rekonstruksi AES S-Box — Chaotic Map + Gradient Boosting")

tab_rekon, tab_uji, tab_teks, tab_gambar, tab_export = st.tabs(
    [
        "Rekonstruksi S-Box",
        "Uji Metrik",
        "Demo Teks",
        "Demo Gambar",
        "Export",
    ]
)

# ==============================================================================
# TAB 1: REKONSTRUKSI S-BOX (CHAOTIC MAP + GRADIENT BOOSTING)
# ==============================================================================
with tab_rekon:
    with st.expander("Parameter Rekonstruksi & Gradient Boosting", expanded=False):
        with st.form("gbr_form"):
            p1, p2, p3, p4 = st.columns(4)
            in_train = p1.slider(
                "Sampel Latih", 200, 1200, st.session_state["ml_params"]["n_train_samples"], 50
            )
            in_cand = p2.slider(
                "Kandidat Chaotic", 500, 5000, st.session_state["ml_params"]["n_candidates"], 250
            )
            in_opt = p3.slider(
                "Iterasi Optimasi Swap", 0, 2000, st.session_state["ml_params"]["n_opt_steps"], 100
            )
            in_est = p4.slider(
                "n_estimators", 50, 400, st.session_state["ml_params"]["n_estimators"], 20
            )

            p5, p6, p7, p8 = st.columns(4)
            in_lr = p5.select_slider(
                "learning_rate",
                options=[0.02, 0.04, 0.05, 0.08, 0.1, 0.15],
                value=st.session_state["ml_params"]["learning_rate"],
            )
            in_depth = p6.slider("max_depth", 2, 8, st.session_state["ml_params"]["max_depth"], 1)
            in_topk = p7.slider(
                "Top-K Verifikasi", 5, 30, st.session_state["ml_params"]["top_k_verify"], 1
            )
            in_seed = p8.number_input(
                "Seed", min_value=1, max_value=9999, value=int(st.session_state["ml_params"]["seed"])
            )

            if st.form_submit_button("Rekonstruksi S-Box", type="primary"):
                st.session_state["ml_params"] = {
                    "n_train_samples": int(in_train),
                    "n_candidates": int(in_cand),
                    "n_opt_steps": int(in_opt),
                    "n_estimators": int(in_est),
                    "learning_rate": float(in_lr),
                    "max_depth": int(in_depth),
                    "subsample": 0.85,
                    "test_size": 0.2,
                    "top_k_verify": int(in_topk),
                    "seed": int(in_seed),
                }
                st.rerun()

    m_info = ml_res["metrics"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("R² (Test Set)", f"{m_info['r2']:.4f}")
    k2.metric("MAE (SV)", f"{m_info['mae']:.4f}")
    k3.metric("RMSE (SV)", f"{m_info['rmse']:.4f}")
    k4.metric("Waktu Screening", f"{m_info['ml_screening_time_sec']*1000:.1f} ms")
    k5.metric("Speedup", f"{m_info['speedup_factor']:.1f}×")

    ch1, ch2 = st.columns(2)
    with ch1:
        scatter_df = pd.DataFrame(
            {"SV Aktual": ml_res["test_actual"], "SV Prediksi (GBR)": ml_res["test_pred"]}
        )
        fig_pred = px.scatter(
            scatter_df,
            x="SV Aktual",
            y="SV Prediksi (GBR)",
            title="SV Aktual vs Prediksi (Gradient Boosting)",
            opacity=0.65,
        )
        min_v = float(min(scatter_df.min()))
        max_v = float(max(scatter_df.max()))
        fig_pred.add_trace(
            go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                name="y = x",
                line=dict(color="red", dash="dash"),
            )
        )
        st.plotly_chart(fig_pred, width="stretch")

    with ch2:
        top_imp = ml_res["importance_df"].head(12).iloc[::-1]
        fig_imp = px.bar(
            top_imp,
            x="Importance",
            y="Fitur",
            orientation="h",
            title="Feature Importance (Top 12)",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_imp, width="stretch")

    st.subheader("Parameter Chaotic Pemenang (2D Logistic-Sine + Swap)")
    prm_df = pd.DataFrame(
        [{"Parameter": k, "Nilai": round(v, 6) if isinstance(v, float) else v} for k, v in ml_res["best_params"].items()]
    )
    st.dataframe(prm_df, width="stretch", hide_index=True)

    st.subheader(f"Top-{m_info['top_k_verify']} Kandidat Hasil Screening GBR")
    st.dataframe(ml_res["top_candidates_df"], width="stretch", hide_index=True)

    st.subheader(f"S-Box & Inverse S-Box 16×16 ({active_name})")
    fmt_mode = st.radio("Format", ["Desimal", "Heksadesimal"], horizontal=True)
    is_hex = fmt_mode == "Heksadesimal"
    sb_col1, sb_col2 = st.columns(2)
    with sb_col1:
        st.markdown("**Forward S-Box**")
        st.dataframe(sbox_to_16x16_dataframe(active_sbox, hex_mode=is_hex), width="stretch")
    with sb_col2:
        st.markdown("**Inverse S-Box**")
        inv_sb = construct_inverse_sbox(active_sbox)
        st.dataframe(sbox_to_16x16_dataframe(inv_sb, hex_mode=is_hex), width="stretch")


# ==============================================================================
# TAB 2: UJI METRIK KRIPTOGRAFI
# ==============================================================================
with tab_uji:
    aes_ev = aes_bench["eval"]
    imp_vs_aes = compare_with_aes(active_eval, aes_ev)

    u1, u2, u3, u4 = st.columns(4)
    u1.metric("NL", f"{active_eval['nl']}", delta=f"{active_eval['nl'] - aes_ev['nl']:+d}")
    u2.metric(
        "SAC",
        f"{active_eval['sac']:.5f}",
        delta=f"{imp_vs_aes['SAC_proximity_imp_pct']:+.2f}%",
    )
    u3.metric(
        "BIC-NL",
        f"{active_eval['bic_nl']}",
        delta=f"{active_eval['bic_nl'] - aes_ev['bic_nl']:+d}",
    )
    u4.metric(
        "BIC-SAC",
        f"{active_eval['bic_sac']:.5f}",
        delta=f"{imp_vs_aes['BICSAC_proximity_imp_pct']:+.2f}%",
    )

    u5, u6, u7, u8 = st.columns(4)
    u5.metric("LAP", f"{active_eval['lap']:.4f}")
    u6.metric("DAP", f"{active_eval['dap']:.6f} (DU={active_eval['diff_uniformity']})")
    u7.metric("Skor S", f"{active_eval['S']:.5f}", delta=f"{imp_vs_aes['S_score_imp_pct']:+.2f}%")
    u8.metric("SV", f"{active_eval['SV']:.5f}")

    b_col1, b_col2 = st.columns([1.3, 1])
    with b_col1:
        st.subheader("Balance Test")
        st.dataframe(pd.DataFrame(active_eval["balance_stats"]), width="stretch", hide_index=True)
    with b_col2:
        st.subheader("Bijectivity")
        st.markdown(
            f"""
            <div class="metric-card">
            <b>Bijective:</b> {'✅ Ya' if active_eval['is_bijective'] else '❌ Tidak'}<br>
            <b>Nilai Unik:</b> {active_eval['unique_count']} / 256<br>
            <b>Balanced:</b> {'✅ Ya' if active_eval['is_balanced'] else '❌ Tidak'}<br>
            <b>Fixed Points:</b> {active_eval['fixed_points']}<br>
            <b>Opposite Fixed Points:</b> {active_eval['opposite_fixed_points']}<br>
            <b>NL per Bit:</b> <code>{active_eval['nl_per_bit']}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    bit_labels = [f"b{i}" for i in range(7, -1, -1)]
    hm1, hm2 = st.columns(2)
    with hm1:
        fig_sac = px.imshow(
            active_eval["sac_matrix"],
            x=bit_labels,
            y=bit_labels,
            text_auto=".4f",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0.5,
            title=f"Matriks SAC 8×8 ({active_eval['sac']:.5f})",
        )
        fig_sac.update_layout(xaxis_title="Bit Output", yaxis_title="Bit Input")
        st.plotly_chart(fig_sac, width="stretch")
    with hm2:
        fig_bsac = px.imshow(
            active_eval["bic_sac_matrix"],
            x=bit_labels,
            y=bit_labels,
            text_auto=".4f",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0.5,
            title=f"Matriks BIC-SAC 8×8 ({active_eval['bic_sac']:.5f})",
        )
        fig_bsac.update_layout(xaxis_title="Bit Output j", yaxis_title="Bit Output k")
        st.plotly_chart(fig_bsac, width="stretch")

    st.subheader("Perbandingan Metrik")
    st.dataframe(comparison_df, width="stretch", hide_index=True)


# ==============================================================================
# TAB 3: DEMO ENKRIPSI & DEKRIPSI TEKS
# ==============================================================================
with tab_teks:
    t_col1, t_col2 = st.columns([1.4, 1])
    with t_col1:
        plaintext_in = st.text_area(
            "Plaintext",
            value="Uji enkripsi dan dekripsi AES-128 menggunakan S-Box Chaotic + Gradient Boosting.",
            height=110,
        )
    with t_col2:
        key_in = st.text_input("Kunci AES-128", value="KunciRahasia2025")
        iv_in = st.text_input("IV (CBC)", value="InitVectorAESML1")
        mode_txt = st.radio("Mode", ["CBC", "ECB"], horizontal=True, key="text_mode")

    sbox_txt_choice = st.selectbox(
        "S-Box Enkripsi",
        [active_name, "S-Box Chaotic + GBR", "AES Standar"],
        index=0,
    )
    sbox_txt_map = {
        active_name: active_sbox,
        "S-Box Chaotic + GBR": ml_res["best_sbox"],
        "AES Standar": aes_bench["sbox"],
    }
    chosen_sb_txt = sbox_txt_map[sbox_txt_choice]

    enc_res = encrypt_text(plaintext_in, chosen_sb_txt, key_in, iv_in, mode=mode_txt)
    try:
        key_b = normalize_key_iv(key_in, 16)
        iv_b = normalize_key_iv(iv_in, 16)
        dec_bytes = decrypt_bytes(enc_res["ciphertext_bytes"], chosen_sb_txt, key_b, iv_b, mode=mode_txt)
        decrypt_text_res = dec_bytes.decode("utf-8")
        dec_match = decrypt_text_res == plaintext_in
    except Exception as exc:
        decrypt_text_res = f"Error: {exc}"
        dec_match = False

    r1, r2 = st.columns(2)
    with r1:
        st.text_area("Ciphertext (Hex)", value=enc_res["hex"], height=100, disabled=True)
        st.text_area("Ciphertext (Base64)", value=enc_res["base64"], height=90, disabled=True)
    with r2:
        st.text_area("Hasil Dekripsi", value=decrypt_text_res, height=150, disabled=True)
        if dec_match:
            st.success("✅ Plaintext == Hasil Dekripsi")
        else:
            st.error("❌ Hasil dekripsi tidak cocok")

    st.subheader("Avalanche Effect (1-Bit Flip)")
    av_rows = []
    for lbl, sb_ref in [
        ("AES Standar", aes_bench["sbox"]),
        ("S-Box Chaotic + GBR", ml_res["best_sbox"]),
    ]:
        av = compute_text_avalanche(plaintext_in, sb_ref, key_in, iv_in, mode=mode_txt)
        av_rows.append(
            {
                "S-Box": lbl,
                "Bit Berubah": f"{av['changed_bits']} / {av['total_bits']}",
                "Avalanche (%)": round(av["avalanche_pct"], 2),
            }
        )
    st.dataframe(pd.DataFrame(av_rows), width="stretch", hide_index=True)


# ==============================================================================
# TAB 4: DEMO ENKRIPSI & DEKRIPSI GAMBAR
# ==============================================================================
with tab_gambar:
    samples = _get_sample_images()
    g_col1, g_col2, g_col3 = st.columns([1.2, 1.1, 1])
    with g_col1:
        img_source = st.radio("Sumber Citra", ["Sampel Bawaan", "Upload Gambar"], horizontal=True)
        if img_source == "Sampel Bawaan":
            sample_key = st.selectbox("Sampel Citra", list(samples.keys()))
            plain_img = samples[sample_key]
        else:
            uploaded = st.file_uploader("Upload PNG / JPG / BMP", type=["png", "jpg", "jpeg", "bmp"])
            resize_opt = st.checkbox("Resize maksimal 256×256 px", value=True)
            if uploaded is not None:
                pil_img = Image.open(uploaded).convert("RGB")
                if resize_opt:
                    pil_img.thumbnail((256, 256))
                plain_img = np.array(pil_img, dtype=np.uint8)
            else:
                plain_img = samples[list(samples.keys())[0]]

    with g_col2:
        img_key_str = st.text_input("Kunci AES Gambar", value="ImageSecretKey25")
        img_iv_str = st.text_input("IV AES Gambar", value="ImageIVector2025")
    with g_col3:
        img_mode = st.radio("Mode", ["CBC", "ECB"], horizontal=True, key="img_mode")
        img_sb_choice = st.selectbox(
            "S-Box Citra",
            [active_name, "S-Box Chaotic + GBR", "AES Standar"],
            key="img_sb_choice",
        )

    img_sb_map = {
        active_name: active_sbox,
        "S-Box Chaotic + GBR": ml_res["best_sbox"],
        "AES Standar": aes_bench["sbox"],
    }
    chosen_sb_img = img_sb_map[img_sb_choice]
    key_bytes_img = normalize_key_iv(img_key_str, 16)
    iv_bytes_img = normalize_key_iv(img_iv_str, 16)

    enc_img_res = encrypt_image(plain_img, chosen_sb_img, key_bytes_img, iv_bytes_img, mode=img_mode)
    cipher_img = enc_img_res["cipher_img"]
    dec_img = decrypt_image(
        enc_img_res["cipher_bytes"],
        enc_img_res["orig_shape"],
        chosen_sb_img,
        key_bytes_img,
        iv_bytes_img,
        mode=img_mode,
    )
    mse_val = float(np.mean((plain_img.astype(np.float64) - dec_img.astype(np.float64)) ** 2))

    vi1, vi2, vi3 = st.columns(3)
    with vi1:
        st.image(plain_img, caption=f"Asli {plain_img.shape}", width="stretch")
    with vi2:
        st.image(cipher_img, caption=f"Cipher ({img_mode})", width="stretch")
    with vi3:
        st.image(dec_img, caption=f"Dekripsi (MSE = {mse_val:.4f})", width="stretch")

    ent_plain = compute_entropy(plain_img)
    ent_cipher = compute_entropy(cipher_img)
    npcr_val, uaci_val = compute_npcr_uaci(
        plain_img, chosen_sb_img, key_bytes_img, iv_bytes_img, mode=img_mode
    )
    corr_plain = compute_pixel_correlations(plain_img, n_samples=2500, seed=42)
    corr_cipher = compute_pixel_correlations(cipher_img, n_samples=2500, seed=42)

    im1, im2, im3, im4, im5 = st.columns(5)
    im1.metric("Entropy (Cipher)", f"{ent_cipher:.5f}", delta=f"Plain: {ent_plain:.3f}")
    im2.metric("NPCR (%)", f"{npcr_val:.4f}%")
    im3.metric("UACI (%)", f"{uaci_val:.4f}%")
    im4.metric("Korelasi H (Cipher)", f"{corr_cipher['horizontal']:.5f}")
    im5.metric("Korelasi V / D (Cipher)", f"{corr_cipher['vertical']:.4f} / {corr_cipher['diagonal']:.4f}")

    h1, h2 = st.columns(2)
    with h1:
        hist_p = np.bincount(plain_img.ravel(), minlength=256)
        fig_hp = px.bar(
            x=np.arange(256),
            y=hist_p,
            title="Histogram Citra Asli",
            labels={"x": "Intensitas", "y": "Frekuensi"},
        )
        st.plotly_chart(fig_hp, width="stretch")
    with h2:
        hist_c = np.bincount(cipher_img.ravel(), minlength=256)
        fig_hc = px.bar(
            x=np.arange(256),
            y=hist_c,
            title="Histogram Citra Terenkripsi",
            labels={"x": "Intensitas", "y": "Frekuensi"},
        )
        st.plotly_chart(fig_hc, width="stretch")

    sc1, sc2 = st.columns(2)
    with sc1:
        px_x, px_y = corr_plain["samples"]["horizontal"]
        fig_sp = px.scatter(
            x=px_x,
            y=px_y,
            opacity=0.4,
            title=f"Korelasi Piksel Plain (r = {corr_plain['horizontal']:.4f})",
            labels={"x": "Piksel (x, y)", "y": "Piksel (x+1, y)"},
        )
        st.plotly_chart(fig_sp, width="stretch")
    with sc2:
        cx_x, cx_y = corr_cipher["samples"]["horizontal"]
        fig_sc = px.scatter(
            x=cx_x,
            y=cx_y,
            opacity=0.4,
            title=f"Korelasi Piksel Cipher (r = {corr_cipher['horizontal']:.4f})",
            labels={"x": "Piksel (x, y)", "y": "Piksel (x+1, y)"},
        )
        st.plotly_chart(fig_sc, width="stretch")


# ==============================================================================
# TAB 5: EXPORT
# ==============================================================================
with tab_export:
    csv_dec_bytes = export_sbox_csv(active_sbox, hex_mode=False)
    csv_hex_bytes = export_sbox_csv(active_sbox, hex_mode=True)
    csv_comp_bytes = export_comparison_csv(comparison_df)
    xlsx_bytes = export_full_report_excel(
        active_name=active_name,
        active_sbox=active_sbox,
        active_eval=active_eval,
        comparison_df=comparison_df,
        active_params=active_params,
        top_candidates_df=ml_res["top_candidates_df"],
    )

    dl1, dl2, dl3, dl4 = st.columns(4)
    dl1.download_button(
        label="Download Laporan (.xlsx)",
        data=xlsx_bytes,
        file_name="laporan_aes_sbox_chaotic_gbr.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    dl2.download_button(
        label="Download S-Box Desimal (.csv)",
        data=csv_dec_bytes,
        file_name="sbox_16x16_desimal.csv",
        mime="text/csv",
    )
    dl3.download_button(
        label="Download S-Box Hex (.csv)",
        data=csv_hex_bytes,
        file_name="sbox_16x16_hex.csv",
        mime="text/csv",
    )
    dl4.download_button(
        label="Download Komparasi (.csv)",
        data=csv_comp_bytes,
        file_name="komparasi_metrik_sbox.csv",
        mime="text/csv",
    )

    p_tab1, p_tab2 = st.tabs(["Komparasi Metrik", "S-Box 16×16"])
    with p_tab1:
        st.dataframe(comparison_df, width="stretch", hide_index=True)
    with p_tab2:
        st.dataframe(sbox_to_16x16_dataframe(active_sbox, hex_mode=False), width="stretch")
