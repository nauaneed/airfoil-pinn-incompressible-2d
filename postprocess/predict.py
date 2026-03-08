"""
Grid construction and model-prediction helpers.

Field operators
---------------
  get_u  – horizontal velocity  u = y[:, 0]  (direct network output)
  get_v  – vertical   velocity  v = y[:, 1]  (direct network output)
  get_p  – pressure             p = y[:, 2]  (direct network output)
"""
import numpy as np
import deepxde as dde

from config import DomainConfig


# ---------------------------------------------------------------------------
# Operator lambdas (passed to model.predict)
# ---------------------------------------------------------------------------

def get_u(x, y):
    return y[:, 0:1]

def get_v(x, y):
    return y[:, 1:2]

def get_p(x, y):
    return y[:, 2:3]


# ---------------------------------------------------------------------------
# Grid and prediction
# ---------------------------------------------------------------------------

def build_grid(domain_cfg: DomainConfig, dx: float = 0.01, dy: float = 0.01):
    """
    Build a regular (x, y) evaluation grid over the fluid domain.

    Returns
    -------
    x, y : 1-D arrays (the axes)
    X    : (N, 2) array of all grid points, row-major (y-outer, x-inner)
    """
    x = np.arange(domain_cfg.xmin, domain_cfg.xmax + dx, dx)
    y = np.arange(domain_cfg.ymin, domain_cfg.ymax + dy, dy)
    xx, yy = np.meshgrid(x, y)      # shape (ny, nx)
    X = np.column_stack([xx.ravel(), yy.ravel()])
    return x, y, X


def predict_fields(model: dde.Model, X: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Run model inference on the grid `X` and reshape each field into a 2-D
    (ny × nx) array aligned with (y, x) axes.

    Returns dict with keys 'u', 'v', 'p'.
    """
    nx, ny = len(x), len(y)
    fields = {}
    for name, op in [("u", get_u), ("v", get_v), ("p", get_p)]:
        raw = model.predict(X, operator=op)
        fields[name] = raw.reshape(ny, nx)
    return fields
