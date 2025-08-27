from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class ParserConfig:
    mode: str = "auto"             # "auto", "y1", "xy2", "custom"
    delimiter: Optional[str] = None
    comment: str = "#"
    x_col: int = 0
    y_col: int = 1
    skiprows: int = 0


@dataclass
class GridConfig:
    mode: str = "first_file"       # "first_file", "span_all"


@dataclass
class DataBundle:
    x: np.ndarray
    Y: np.ndarray
    file_names: List[str]


def _load_array(bytes_or_str: bytes, delimiter: Optional[str], comment: str, skiprows: int) -> np.ndarray:
    bio = io.BytesIO(bytes_or_str if isinstance(bytes_or_str, (bytes, bytearray)) else bytes(bytes_or_str, 'utf-8'))
    arr = np.loadtxt(bio, comments=comment, delimiter=delimiter, skiprows=skiprows)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _read_txt_like_auto(bytes_or_str: bytes, comment: str, skiprows: int) -> np.ndarray:
    bio = io.BytesIO(bytes_or_str if isinstance(bytes_or_str, (bytes, bytearray)) else bytes(bytes_or_str, 'utf-8'))
    try:
        arr = np.loadtxt(bio, comments=comment, delimiter=None, skiprows=skiprows)
    except Exception:
        bio.seek(0)
        arr = np.loadtxt(bio, comments=comment, delimiter=",", skiprows=skiprows)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def parse_xy(bytes_or_str: bytes, filename: str, cfg: ParserConfig) -> Tuple[Optional[np.ndarray], np.ndarray]:
    if cfg.mode == "auto":
        arr = _read_txt_like_auto(bytes_or_str, comment=cfg.comment, skiprows=cfg.skiprows)
        if arr.shape[1] >= 2:
            x = arr[:, 0]
            y = arr[:, 1]
        else:
            x = None
            y = arr[:, 0]
        return x, y

    if cfg.mode == "y1":
        arr = _load_array(bytes_or_str, delimiter=cfg.delimiter, comment=cfg.comment, skiprows=cfg.skiprows)
        y_col = min(cfg.y_col, arr.shape[1]-1)
        return None, arr[:, y_col]

    if cfg.mode == "xy2":
        arr = _load_array(bytes_or_str, delimiter=cfg.delimiter, comment=cfg.comment, skiprows=cfg.skiprows)
        x_col = min(cfg.x_col, arr.shape[1]-1)
        y_col = min(cfg.y_col, arr.shape[1]-1)
        return arr[:, x_col], arr[:, y_col]

    if cfg.mode == "custom":
        arr = _load_array(bytes_or_str, delimiter=cfg.delimiter, comment=cfg.comment, skiprows=cfg.skiprows)
        x_col = min(cfg.x_col, arr.shape[1]-1)
        y_col = min(cfg.y_col, arr.shape[1]-1)
        if arr.shape[1] == 1:
            return None, arr[:, 0]
        return arr[:, x_col], arr[:, y_col]

    raise ValueError(f"Unknown parser mode: {cfg.mode}")


def assemble_dataset(file_blobs: List[Tuple[str, bytes]], parser: ParserConfig = ParserConfig(), grid: GridConfig = GridConfig()) -> DataBundle:
    xs: List[Optional[np.ndarray]] = []
    ys: List[np.ndarray] = []
    names: List[str] = []

    for name, blob in file_blobs:
        x, y = parse_xy(blob, name, parser)
        xs.append(None if x is None else np.asarray(x, dtype=float))
        ys.append(np.asarray(y, dtype=float))
        names.append(name)

    if grid.mode == "first_file":
        x_ref = None
        for x in xs:
            if x is not None:
                x_ref = x
                break
        if x_ref is None:
            x_ref = np.arange(len(ys[0]), dtype=float)
    elif grid.mode == "span_all":
        x_ref = None
        first_x = None
        for x in xs:
            if x is not None:
                first_x = x
                break
        if first_x is not None and len(first_x) > 1:
            dx = np.median(np.diff(first_x))
            xmin = min(x[0] if x is not None else first_x[0] for x in xs)
            xmax = max(x[-1] if x is not None else first_x[-1] for x in xs)
            n = int(max(2, round((xmax - xmin) / dx) + 1))
            x_ref = np.linspace(xmin, xmax, n)
        else:
            Lmax = max(len(y) for y in ys)
            x_ref = np.arange(Lmax, dtype=float)
    else:
        raise ValueError(f"Unknown grid mode: {grid.mode}")

    Y_list = []
    for x, y in zip(xs, ys):
        if x is None:
            if len(y) == len(x_ref):
                Y_list.append(y)
            else:
                xi = np.linspace(0, len(y)-1, num=len(y))
                refi = np.linspace(0, len(y)-1, num=len(x_ref))
                f = interp1d(xi, y, kind="linear", bounds_error=False, fill_value="extrapolate")
                Y_list.append(f(refi))
        else:
            f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
            Y_list.append(f(x_ref))

    Y = np.vstack([row[None, :] for row in Y_list])
    return DataBundle(x=x_ref, Y=Y, file_names=names)


def read_basis_vectors(file_blobs: List[Tuple[str, bytes]], x_target: np.ndarray, parser: ParserConfig = ParserConfig()):
    basis_rows = []
    names = []
    for name, blob in file_blobs:
        x, y = parse_xy(blob, name, parser)
        y = np.asarray(y, dtype=float)
        if x is None:
            if len(y) == len(x_target):
                yy = y
            else:
                xi = np.linspace(0, len(y)-1, num=len(y))
                refi = np.linspace(0, len(y)-1, num=len(x_target))
                f = interp1d(xi, y, kind="linear", bounds_error=False, fill_value="extrapolate")
                yy = f(refi).astype(float)
        else:
            f = interp1d(np.asarray(x, dtype=float), y, kind="linear", bounds_error=False, fill_value="extrapolate")
            yy = f(x_target).astype(float)
        basis_rows.append(yy)
        names.append(name)

    B = np.vstack([row[None, :] for row in basis_rows])
    return B, names
