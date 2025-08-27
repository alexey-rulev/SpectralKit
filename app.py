import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from modules.data_io import assemble_dataset, read_basis_vectors, ParserConfig, GridConfig
from modules.utils import clip_nonneg, normalize_rows, apply_smoothing
from modules.analysis_nmf import run_nmf
from modules.analysis_pca import run_pca
from modules.analysis_nnls import run_nnls
from modules.analysis_lstsq import run_lstsq
from modules import plotting as pltmod

st.set_page_config(page_title="Spectral Analyzer", layout="wide")
st.title("🔬 Spectral Analyzer")
st.caption("NMF, PCA, NNLS, and LSQ analysis of multi-file 1D spectra/signals")

with st.expander("About data formats", expanded=False):
    st.markdown("""
    Upload multiple **text files**. Choose how to interpret them:
    - **Auto**: try whitespace/csv; 2 columns -> x,y; 1 column -> y-only
    - **1-column (y)**: treat each file as y-only
    - **2-column (x,y)**: enforce first two columns as x,y
    - **Custom**: choose delimiter/comment and which columns are x and y
    """)

# Data upload
st.sidebar.header("1) Upload & parse data")
data_files = st.sidebar.file_uploader("Upload data files (txt/csv)", type=["txt","csv","dat"], accept_multiple_files=True)
if not data_files:
    st.info("Upload your data files to begin.")
    st.stop()
file_blobs: List[Tuple[str, bytes]] = [(f.name, f.getvalue()) for f in data_files]

# Parser config
fmt = st.sidebar.selectbox("Data format", ["auto", "1-column (y)", "2-column (x y)", "custom"], index=0)
if fmt == "1-column (y)":
    mode = "y1"
elif fmt == "2-column (x y)":
    mode = "xy2"
elif fmt == "custom":
    mode = "custom"
else:
    mode = "auto"

delimiter_choice = None
comment_char = "#"
x_col = 0; y_col = 1; skiprows = 0
if mode in ("xy2", "y1", "custom"):
    with st.sidebar.expander("Parsing options", expanded=(mode=="custom")):
        delim_sel = st.selectbox("Delimiter", ["Whitespace", "Comma (,)", "Tab (\\t)", "Semicolon (;)", "Custom"], index=0)
        if delim_sel == "Whitespace": delimiter_choice = None
        elif delim_sel == "Comma (,)": delimiter_choice = ","
        elif delim_sel == "Tab (\\t)": delimiter_choice = "\\t"
        elif delim_sel == "Semicolon (;)": delimiter_choice = ";"
        else: delimiter_choice = st.text_input("Custom delimiter string", value=",")
        comment_char = st.text_input("Comment prefix", value="#")
        skiprows = int(st.number_input("Skip initial rows", 0, 1000, 0, step=1))
        if mode in ("xy2", "custom"):
            x_col = int(st.number_input("x column (0-based)", 0, 99, 0, step=1))
            y_col = int(st.number_input("y column (0-based)", 0, 99, 1, step=1))

parser_cfg = ParserConfig(mode=mode, delimiter=delimiter_choice, comment=comment_char, x_col=x_col, y_col=y_col, skiprows=skiprows)

# Grid selection
st.sidebar.subheader("Interpolation grid")
grid_mode = st.sidebar.selectbox("Grid mode", ["first_file", "span_all"], index=0,
    help="Use x from first file providing x, or span min..max across all files with first-file spacing.")
grid_cfg = GridConfig(mode=grid_mode)

# Build dataset
bundle = assemble_dataset(file_blobs, parser=parser_cfg, grid=grid_cfg)
x = bundle.x
X = bundle.Y

# Preprocessing
st.sidebar.subheader("Preprocessing")
smooth_method = st.sidebar.selectbox("Smoothing", ["none", "savgol", "moving_average", "gaussian"], index=0)
if smooth_method == "savgol":
    win = st.sidebar.number_input("Window length (odd)", 3, 301, 9, step=2)
    poly = st.sidebar.number_input("Polyorder", 1, 9, 2, step=1)
    sigma = 0.0
elif smooth_method == "moving_average":
    win = st.sidebar.number_input("Window (points)", 1, 501, 9, step=1)
    poly = 0; sigma = 0.0
elif smooth_method == "gaussian":
    win = 0; poly = 0
    sigma = st.sidebar.number_input("Sigma (points)", 0.1, 100.0, 2.0, step=0.1)
else:
    win = 0; poly = 0; sigma = 0.0

clip0 = st.sidebar.checkbox("Clip negatives to 0 (after smoothing)", value=True, help="Applied after smoothing to remove small negative artifacts.")
norm_method = st.sidebar.selectbox("Normalize (row-wise)", ["none", "max", "l2", "area"], index=0)

X_proc = X.copy()
X_proc = apply_smoothing(X_proc, method=smooth_method, window=int(win), poly=int(poly), sigma=float(sigma))
if clip0:
    X_proc = clip_nonneg(X_proc)
if norm_method != "none":
    X_proc = normalize_rows(X_proc, method=norm_method)

st.write(f"**Loaded {X.shape[0]} files**, each with **{X.shape[1]} points** (after interpolation).")
with st.expander("Preview first 3 samples"):
    df = pd.DataFrame(X_proc[:3].T, columns=[os.path.basename(n) for n in bundle.file_names[:3]])
    st.dataframe(df.head(20))

# Analysis selection
st.sidebar.header("2) Choose analysis")
method = st.sidebar.radio("Method", ["NMF", "PCA", "NNLS", "LSQ"], horizontal=True)

if method == "NMF":
    st.sidebar.subheader("NMF parameters")
    k = st.sidebar.number_input("n_components", 1, min(8, X_proc.shape[1]), min(3, X_proc.shape[1]), step=1)
    init_choice = st.sidebar.selectbox("Initialization", ["nndsvd", "random", "custom (from basis files)"])
    max_iter = st.sidebar.number_input("max_iter", 100, 50000, 5000, step=100)
    l1_ratio = st.sidebar.slider("l1_ratio", 0.0, 1.0, 0.0, 0.05)
    alpha_W = st.sidebar.number_input("alpha_W", 0.0, 10.0, 0.0, step=0.1)
    alpha_H = st.sidebar.number_input("alpha_H", 0.0, 10.0, 0.0, step=0.1)
    random_state = st.sidebar.number_input("random_state", 0, 10, 0, step=1)

    # NMF custom initialization
    H_init = None
    W_init = None
    if "custom" in init_choice:
        st.sidebar.markdown("**Upload initial basis components (H)**")
        H_files = st.sidebar.file_uploader("H components (each file is one component)", type=["txt","csv","dat"], accept_multiple_files=True, key="nmf_h")
        with st.sidebar.expander("Preprocess H init (optional)", expanded=False):
            use_same = st.checkbox("Use same smoothing as data", value=True, key="nmf_h_use_same")
            if not use_same:
                h_smooth_method = st.selectbox("H smoothing", ["none", "savgol", "moving_average", "gaussian"], index=0, key="nmf_h_smooth_method")
                if h_smooth_method == "savgol":
                    h_win = st.number_input("H window length (odd)", 3, 301, 9, step=2, key="nmf_h_win")
                    h_poly = st.number_input("H polyorder", 1, 9, 2, step=1, key="nmf_h_poly")
                    h_sigma = 0.0
                elif h_smooth_method == "moving_average":
                    h_win = st.number_input("H window (points)", 1, 501, 9, step=1, key="nmf_h_win_ma")
                    h_poly = 0; h_sigma = 0.0
                elif h_smooth_method == "gaussian":
                    h_win = 0; h_poly = 0
                    h_sigma = st.number_input("H sigma (points)", 0.1, 100.0, 2.0, step=0.1, key="nmf_h_sigma")
                else:
                    h_win = 0; h_poly = 0; h_sigma = 0.0
            else:
                h_smooth_method = smooth_method
                h_win, h_poly, h_sigma = int(win), int(poly), float(sigma)
            h_clip0 = st.checkbox("Clip negatives in H after smoothing", value=True, key="nmf_h_clip0")

        if H_files:
            B, names = read_basis_vectors([(f.name, f.getvalue()) for f in H_files], x_target=x, parser=parser_cfg)
            if B.shape[0] != k:
                st.sidebar.warning(
                    f"Uploaded {B.shape[0]} components, but n_components is {k}. "
                    f"Using the first {min(B.shape[0], k)} and random for the rest."
                )
                B = B[:min(B.shape[0], k), :]
            # Preprocess H_init
            B = apply_smoothing(B, method=h_smooth_method, window=int(h_win), poly=int(h_poly), sigma=float(h_sigma))
            if h_clip0:
                B = clip_nonneg(B)
            H_init = B  # (<=k, n_features)

        st.sidebar.markdown("**Upload initial coefficients (W)** (CSV)")
        W_file = st.sidebar.file_uploader("W coefficients", type=["csv"], accept_multiple_files=False, key="nmf_w")
        if W_file is not None:
            try:
                W_df = pd.read_csv(W_file)
                W_arr = W_df.to_numpy()
                if W_arr.shape[0] != X_proc.shape[0]:
                    st.sidebar.warning(
                        f"W file has {W_arr.shape[0]} rows but data has {X_proc.shape[0]} samples. Ignoring W init."
                    )
                else:
                    if W_arr.shape[1] != k:
                        st.sidebar.warning(
                            f"W file has {W_arr.shape[1]} components, expected {k}. "
                            "Missing columns will be filled with 0.5 or extra columns truncated."
                        )
                    W_init = W_arr
            except Exception as e:
                st.sidebar.warning(f"Failed to parse W file: {e}")

    try:
        res = run_nmf(
            X=X_proc, n_components=int(k),
            init="custom" if "custom" in init_choice and H_init is not None else init_choice,
            H_init=H_init, W_init=W_init, max_iter=int(max_iter), l1_ratio=float(l1_ratio),
            alpha_W=float(alpha_W), alpha_H=float(alpha_H), random_state=int(random_state),
        )
    except Exception as e:
        st.error(f"NMF failed: {e}")
        st.stop()

    st.subheader("Results — NMF")
    st.write(f"Relative reconstruction error: **{res.recon_error:.4g}**")
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(pltmod.line_components(x, res.H, "NMF components (rows of H)"), use_container_width=True)
    with col2:
        st.plotly_chart(pltmod.line_coeffs(res.W, "NMF coefficients (W) vs sample", sample_names=[os.path.basename(n) for n in bundle.file_names]), use_container_width=True)
    st.plotly_chart(pltmod.heat_residuals(res.residuals, "Residuals (X - W @ H)"), use_container_width=True)
    st.markdown("**Per-sample fit viewer**")
    idx = st.slider("Sample index", 0, X_proc.shape[0]-1, 0, key="nmf_idx")
    st.plotly_chart(pltmod.line_fit_residual(x, X_proc[idx], res.X_hat[idx], f"Sample {idx} — data/fit/residual"), use_container_width=True)
    st.download_button("Download NMF components (H) CSV", data=pd.DataFrame(res.H, columns=x).to_csv(index=False).encode("utf-8"), file_name="nmf_components_H.csv")
    st.download_button("Download NMF coefficients (W) CSV", data=pd.DataFrame(res.W).to_csv(index=False).encode("utf-8"), file_name="nmf_coeffs_W.csv")

elif method == "PCA":
    st.sidebar.subheader("PCA parameters")
    k = st.sidebar.number_input("n_components", 1, min(12, X_proc.shape[1]), min(3, X_proc.shape[1]), step=1)
    whiten = st.sidebar.checkbox("whiten", value=False)
    random_state = st.sidebar.number_input("random_state", 0, 10, 0, step=1)
    try:
        res = run_pca(X_proc, n_components=int(k), whiten=bool(whiten), random_state=int(random_state))
    except Exception as e:
        st.error(f"PCA failed: {e}")
        st.stop()
    st.subheader("Results — PCA")
    st.write(f"Relative reconstruction error: **{res.recon_error:.4g}**")
    st.write("Explained variance ratio per component:", np.round(res.explained_variance_ratio, 4))
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(pltmod.line_components(x, res.components, "PCA loadings (components)"), use_container_width=True)
    with col2:
        st.plotly_chart(pltmod.line_coeffs(res.scores, "PCA scores vs sample", sample_names=[os.path.basename(n) for n in bundle.file_names]), use_container_width=True)
    st.plotly_chart(pltmod.heat_residuals(res.residuals, "Residuals (X - scores @ components - mean)"), use_container_width=True)
    st.markdown("**Per-sample fit viewer**")
    idx = st.slider("Sample index", 0, X_proc.shape[0]-1, 0, key="pca_idx")
    st.plotly_chart(pltmod.line_fit_residual(x, X_proc[idx], res.X_hat[idx], f"Sample {idx} — data/fit/residual"), use_container_width=True)
    st.download_button("Download PCA components CSV", data=pd.DataFrame(res.components, columns=x).to_csv(index=False).encode("utf-8"), file_name="pca_components.csv")
    st.download_button("Download PCA scores CSV", data=pd.DataFrame(res.scores).to_csv(index=False).encode("utf-8"), file_name="pca_scores.csv")

elif method == "NNLS":
    st.sidebar.subheader("NNLS basis")
    basis_files = st.sidebar.file_uploader("Upload basis files (each file is one basis vector)", type=["txt","csv","dat"], accept_multiple_files=True, key="nnls_basis")
    if not basis_files:
        st.info("Upload basis files for NNLS.")
        st.stop()
    B, names = read_basis_vectors([(f.name, f.getvalue()) for f in basis_files], x_target=x, parser=parser_cfg)
    st.write(f"Loaded **{B.shape[0]} basis vectors**, each length **{B.shape[1]}**.")
    try:
        res = run_nnls(X_proc, B)
    except Exception as e:
        st.error(f"NNLS failed: {e}")
        st.stop()
    st.subheader("Results — NNLS")
    st.write(f"Relative reconstruction error: **{res.recon_error:.4g}**")
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(pltmod.line_components(x, B, "NNLS basis components", names), use_container_width=True)
    with col2:
        st.plotly_chart(pltmod.line_coeffs(res.coeffs, "NNLS coefficients vs sample", sample_names=[os.path.basename(n) for n in bundle.file_names]), use_container_width=True)
    st.plotly_chart(pltmod.heat_residuals(res.residuals, "Residuals (X - C @ B)"), use_container_width=True)
    st.markdown("**Per-sample fit viewer**")
    idx = st.slider("Sample index", 0, X_proc.shape[0]-1, 0, key="nnls_idx")
    st.plotly_chart(pltmod.line_fit_residual(x, X_proc[idx], res.X_hat[idx], f"Sample {idx} — data/fit/residual"), use_container_width=True)
    st.download_button("Download NNLS coefficients CSV", data=pd.DataFrame(res.coeffs).to_csv(index=False).encode("utf-8"), file_name="nnls_coeffs.csv")
    st.download_button("Download NNLS basis (interpolated) CSV", data=pd.DataFrame(B, columns=x).to_csv(index=False).encode("utf-8"), file_name="nnls_basis_interpolated.csv")

else:  # LSQ
    st.sidebar.subheader("LSQ basis (unconstrained least squares)")
    basis_files = st.sidebar.file_uploader("Upload basis files (each file is one basis vector)", type=["txt","csv","dat"], accept_multiple_files=True, key="lsq_basis")
    if not basis_files:
        st.info("Upload basis files for LSQ.")
        st.stop()
    B, names = read_basis_vectors([(f.name, f.getvalue()) for f in basis_files], x_target=x, parser=parser_cfg)
    st.write(f"Loaded **{B.shape[0]} basis vectors**, each length **{B.shape[1]}**.")
    try:
        res = run_lstsq(X_proc, B)
    except Exception as e:
        st.error(f"LSQ failed: {e}")
        st.stop()
    st.subheader("Results — LSQ")
    st.write(f"Relative reconstruction error: **{res.recon_error:.4g}**")
    col1, col2 = st.columns([1,1])
    with col1:
        st.plotly_chart(pltmod.line_components(x, B, "LSQ basis components", names), use_container_width=True)
    with col2:
        st.plotly_chart(pltmod.line_coeffs(res.coeffs, "LSQ coefficients vs sample", sample_names=[os.path.basename(n) for n in bundle.file_names]), use_container_width=True)
    st.plotly_chart(pltmod.heat_residuals(res.residuals, "Residuals (X - C @ B)"), use_container_width=True)
    st.markdown("**Per-sample fit viewer**")
    idx = st.slider("Sample index", 0, X_proc.shape[0]-1, 0, key="lsq_idx")
    st.plotly_chart(pltmod.line_fit_residual(x, X_proc[idx], res.X_hat[idx], f"Sample {idx} — data/fit/residual"), use_container_width=True)
    st.download_button("Download LSQ coefficients CSV", data=pd.DataFrame(res.coeffs).to_csv(index=False).encode("utf-8"), file_name="lsq_coeffs.csv")
    st.download_button("Download LSQ basis (interpolated) CSV", data=pd.DataFrame(B, columns=x).to_csv(index=False).encode("utf-8"), file_name="lsq_basis_interpolated.csv")
