# Spectral Components Analyzer (Streamlit)

A program to decompose a series of spectra into basis components using PCA, NMF, LSQ (or NNLSQ) fitting

- Select **data format**: auto, 1-column (y), 2-column (x y), or custom.
- Select **interpolation grid**: use first file’s x, or span all files (min..max with first-file spacing).
- **Preprocessing**: optionally smooth corresponding values between adjacent uploaded samples after interpolation (in upload order), then smooth each spectrum along its x-axis using Savitzky–Golay, moving average, or Gaussian filters. Pipeline is **interpolate → between-sample smooth → within-spectrum smooth → clip negatives → normalize**.
- Analyses: **NMF** (Non-negative Matrix Factorization, fits both components and coefficients, optional custom init for H and W from files — missing H rows are random and missing W columns filled with 0.5), **PCA** (Principal Components Analysis, fits components and coefficients, components are orthogonal), **NNLS** (Non-Negative Least Squares, fits only coefficients, uses datapoints as basis set), **LSQ** (unconstrained Least SQuares).
- Plots: components, line-plotted coefficients/scores, reconstruction, residuals.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Docker
```bash
docker build -t spectral-analyzer .
docker run --rm -p 8501:8501 spectral-analyzer   # open http://localhost:8501
```
Change container port if needed:
```bash
docker run --rm -e STREAMLIT_SERVER_PORT=9000 -p 9000:9000 spectral-analyzer
```

With Compose:
```bash
docker compose up --build
```

## Example
Use .dat files from example folder to try various options of fitting.


# Additional files
drift_analysis_clean.ipynb is a Jupyter notebook for correcting time-drift in NRVS spectra. Notebook is optimized for the SPring-8 NRVS spectra measured at BL35XU. The data is available at https://doi.org/10.5281/zenodo.22704071
