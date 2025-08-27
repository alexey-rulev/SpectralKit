# Spectral Components Analyzer (Streamlit)

A program to decompose a series of spectra into basis components using PCA, NNLS, LSQ fitting

- Select **data format**: auto, 1-column (y), 2-column (x y), or custom.
- Select **interpolation grid**: use first file’s x, or span all files (min..max with first-file spacing).
- **Smoothing**: none, Savitzky–Golay, moving average, Gaussian. Pipeline is **smooth → clip negatives → normalize**.
- Analyses: **NMF** (optional custom init, partial components allowed), **PCA**, **NNLS**, **LSQ** (unconstrained least squares).
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
Use .dat files from example folder to try various options.
