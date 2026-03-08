"""
NACA 4-digit airfoil boundary coordinate generator.

Reference convention: NACA MPSS  (e.g. NACA 0012 → M=0, P=0, SS=12)
Returns a closed polygon of (x, y) coordinates ordered from lower-trailing-edge
around the lower surface, over the leading edge, and back along the upper surface.
"""
import numpy as np
from config import AirfoilConfig


def naca4_boundary(cfg: AirfoilConfig) -> np.ndarray:
    """
    Return the (2*n + 1, 2) array of boundary coordinates for a NACA 4-digit
    airfoil, translated by (offset_x, offset_y).

    Parameters are taken from an AirfoilConfig instance so callers only need to
    pass one object.
    """
    m = cfg.M  / 100.0
    p = cfg.P  / 10.0
    t = cfg.SS / 100.0
    c = cfg.chord
    n = cfg.n_panels

    # Cosine-clustered chord discretisation (finer near LE/TE)
    beta = np.linspace(0.0, np.pi, n + 1)
    xv   = c / 2.0 * (1.0 - np.cos(beta))

    # Thickness distribution (NACA 4-digit formula)
    xi = xv / c
    yt = 5 * t * c * (
          0.2969 * xi**0.5
        - 0.1260 * xi
        - 0.3516 * xi**2
        + 0.2843 * xi**3
        - 0.1015 * xi**4
    )

    # Mean camber line and its slope
    # Symmetric airfoil (m == 0): no camber at all – skip to avoid /0.
    if m == 0.0:
        yc  = np.zeros_like(xv)
        dyc = np.zeros_like(xv)
    else:
        yc  = np.where(xv <= p * c,
                       c * m / p**2      * xi * (2*p - xi),
                       c * m / (1-p)**2  * (1 + (2*p - xi) * xi - 2*p))
        dyc = np.where(xv <= p * c,
                       m / p**2      * 2 * (p - xi),
                       m / (1-p)**2  * 2 * (p - xi))

    th = np.arctan2(dyc, 1.0)
    xU = xv - yt * np.sin(th)
    yU = yc + yt * np.cos(th)
    xL = xv + yt * np.sin(th)
    yL = yc - yt * np.cos(th)

    # Assemble: lower surface (tip→LE) + upper surface (LE→tip) = closed loop
    x = np.empty(2 * n + 1)
    y = np.empty(2 * n + 1)
    x[:n]       = xL[n:0:-1]   # lower, trailing edge → leading edge
    y[:n]       = yL[n:0:-1]
    x[n:]       = xU            # upper, leading edge → trailing edge
    y[n:]       = yU

    return np.column_stack((x + cfg.offset_x, y + cfg.offset_y))
