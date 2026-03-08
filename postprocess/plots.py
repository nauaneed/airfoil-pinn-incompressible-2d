"""
Post-processing plots for the NACA airfoil PINN results.

Each function saves a PNG and optionally returns the Figure.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import AirfoilConfig
from geometry.airfoil import naca4_boundary


_FIELD_META = {
    "p": dict(label="Pressure  p",           cmap="coolwarm"),
    "u": dict(label="Velocity  u  (x-comp)", cmap="viridis"),
    "v": dict(label="Velocity  v  (y-comp)", cmap="RdBu"),
}


def _add_airfoil_patch(ax, airfoil_cfg: AirfoilConfig, **kwargs):
    """Fill the airfoil solid on an existing axis."""
    pts = naca4_boundary(airfoil_cfg)
    ax.fill(pts[:, 0], pts[:, 1], color="white", zorder=5, **kwargs)
    ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=0.8, zorder=6)


def plot_field(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    airfoil_cfg: AirfoilConfig,
    out_path: str | Path = ".",
    n_levels: int = 200,
    figsize=(16, 9),
):
    """
    Filled-contour plot for a single field (u, v, or p).

    Parameters
    ----------
    field     : (ny, nx) array
    name      : one of 'u', 'v', 'p'
    out_path  : directory where the PNG is saved
    """
    meta = _FIELD_META.get(name, dict(label=name, cmap="viridis"))
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    levels = np.linspace(field.min(), field.max(), n_levels)

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(x, y, field, levels=levels, cmap=meta["cmap"], extend="both")
    fig.colorbar(cf, ax=ax, label=meta["label"])
    _add_airfoil_patch(ax, airfoil_cfg)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(meta["label"])

    save_file = out_path / f"field_{name}.png"
    fig.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {save_file}")
    return fig


def plot_streamlines(
    fields: dict,
    x: np.ndarray,
    y: np.ndarray,
    airfoil_cfg: AirfoilConfig,
    out_path: str | Path = ".",
    figsize=(16, 9),
):
    """Streamline plot overlaid on speed magnitude."""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    u, v  = fields["u"], fields["v"]
    speed = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(x, y, speed, levels=100, cmap="viridis", extend="both")
    fig.colorbar(cf, ax=ax, label="|U|")
    ax.streamplot(x, y, u, v, density=1.5, color="white", linewidth=0.6, arrowsize=0.8)
    _add_airfoil_patch(ax, airfoil_cfg)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Streamlines coloured by speed magnitude")

    save_file = out_path / "streamlines.png"
    fig.savefig(save_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {save_file}")
    return fig


def plot_all_fields(
    fields: dict,
    x: np.ndarray,
    y: np.ndarray,
    airfoil_cfg: AirfoilConfig,
    out_path: str | Path = "results",
):
    """Convenience wrapper: plot u, v, p + streamlines."""
    for name in ("u", "v", "p"):
        plot_field(fields[name], x, y, name, airfoil_cfg, out_path=out_path)
    plot_streamlines(fields, x, y, airfoil_cfg, out_path=out_path)
