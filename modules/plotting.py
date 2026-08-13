from __future__ import annotations
from typing import Optional, List
import numpy as np
import plotly.graph_objects as go

def line_components(
    x: np.ndarray,
    M: np.ndarray,
    title: str,
    errors: Optional[np.ndarray] = None,
    names: Optional[List[str]] = None,
) -> go.Figure:
    fig = go.Figure()
    k = M.shape[0]
    for i in range(k):
        name = names[i] if names and i < len(names) else f"C{i+1}"
        fig.add_trace(go.Scatter(
            x=x, y=M[i], mode="lines", name=name,
            error_y=dict(type="data", array=errors[i], visible=True) if errors is not None else None,
        ))
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="amplitude", legend_title="Components")
    return fig

def line_coeffs(
    C: np.ndarray,
    title: str,
    errors: Optional[np.ndarray] = None,
    sample_names: Optional[List[str]] = None,
) -> go.Figure:
    n_samples, k = C.shape
    x = list(range(n_samples))
    hover_text = sample_names if sample_names is not None else None
    fig = go.Figure()
    for j in range(k):
        fig.add_trace(go.Scatter(
            x=x, y=C[:, j], mode="lines+markers", name=f"C{j+1}",
            text=hover_text,
            error_y=dict(type="data", array=errors[:, j], visible=True) if errors is not None else None,
            hovertemplate=("sample=%{text}<br>idx=%{x}<br>value=%{y}<extra>%{fullData.name}</extra>") if hover_text else None,
        ))
    fig.update_layout(title=title, xaxis_title="sample index", yaxis_title="value", legend_title="Component")
    if sample_names:
        step = max(1, len(sample_names)//20)
        tickvals = list(range(0, len(sample_names), step))
        ticktext = [sample_names[i] for i in tickvals]
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    return fig

def line_fit_residual(x: np.ndarray, y_true: np.ndarray, y_fit: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_true, mode="lines", name="data"))
    fig.add_trace(go.Scatter(x=x, y=y_fit, mode="lines", name="fit"))
    fig.add_trace(go.Scatter(x=x, y=y_true - y_fit, mode="lines", name="residual"))
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="amplitude")
    return fig

def heat_residuals(R: np.ndarray, title: str):
    fig = go.Figure(data=go.Heatmap(z=R, colorbar=dict(title="residual"), colorscale="RdBu"))
    fig.update_layout(title=title, xaxis_title="feature", yaxis_title="sample")
    return fig
