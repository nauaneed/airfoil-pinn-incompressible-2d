"""
scripts/plot_domain.py
======================
Observability script: visualise the computational domain without running
any training.

Produces three figures:

  1. domain_overview.png   – farfield box, inner/outer sub-domains,
                             airfoil polygon, and all sampling points
                             colour-coded by region.
  2. domain_selections.png – named boundary selections (inlet, outlet,
                             top_bottom, airfoil surface) in distinct colours.
  3. sampling_density.png  – 2-D histogram of collocation-point density.

Run from the project root::

    python scripts/plot_domain.py
"""
import sys
import pathlib
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import deepxde as dde

from config import cfg
from geometry.airfoil import naca4_boundary
from geometry.domain  import build_geometry, build_sampling_points, sample_farfield_boundary

OUT = ROOT / "outputs" / "domain"
OUT.mkdir(parents=True, exist_ok=True)


def _airfoil_patch(airfoil_cfg, color="lightgrey", zorder=4, **kw):
    pts = naca4_boundary(airfoil_cfg)
    return plt.Polygon(pts, closed=True, facecolor=color, edgecolor="black",
                       linewidth=0.8, zorder=zorder, **kw)


# ---------------------------------------------------------------------------
# Figure 1 – domain overview with sampling points
# ---------------------------------------------------------------------------
def plot_domain_overview():
    dc = cfg.domain
    ac = cfg.airfoil
    sc = cfg.sampling

    geom, farfield = build_geometry(dc, ac)
    points = build_sampling_points(dc, ac, sc)

    n_outer = sc.n_outer
    n_inner = sc.n_inner
    n_le    = sc.n_le
    n_te    = sc.n_te
    n_ff    = sc.n_inlet + sc.n_outlet + sc.n_top_bottom
    n_af    = 2 * ac.n_panels + 1
    # Order matches build_sampling_points: outer | inner | le | te | ff | airfoil
    splits  = np.cumsum([n_outer, n_inner, n_le, n_te, n_ff])

    outer_pts   = points[:splits[0]]
    inner_pts   = points[splits[0]:splits[1]]
    le_pts      = points[splits[1]:splits[2]]
    te_pts      = points[splits[2]:splits[3]]
    ff_pts      = points[splits[3]:splits[4]]   # all farfield sides combined
    airfoil_pts = points[splits[4]:]

    # Split farfield into per-side for colour-coded display
    ff_splits   = np.cumsum([sc.n_inlet, sc.n_outlet, sc.n_top_bottom // 2])
    inlet_pts   = ff_pts[:ff_splits[0]]
    outlet_pts  = ff_pts[ff_splits[0]:ff_splits[1]]
    tb_pts      = ff_pts[ff_splits[1]:]

    # Derived airfoil positions (for drawing region outlines)
    mid_x = ac.offset_x + ac.chord / 2
    mid_y = ac.offset_y
    le_x, le_y = ac.offset_x, ac.offset_y
    te_x, te_y = ac.offset_x + ac.chord, ac.offset_y

    fig, ax = plt.subplots(figsize=(18, 11))

    # Farfield rectangle
    rect = mpatches.FancyBboxPatch(
        (dc.xmin, dc.ymin), dc.xmax - dc.xmin, dc.ymax - dc.ymin,
        boxstyle="square,pad=0", linewidth=2, edgecolor="#333333",
        facecolor="#f7f9fc", zorder=0, label="Farfield domain"
    )
    ax.add_patch(rect)

    # Inner oval (ellipse)
    inner_ellipse = mpatches.Ellipse(
        (mid_x, mid_y), 2 * dc.inner_semi_x, 2 * dc.inner_semi_y,
        linewidth=1.5, edgecolor="#e07b00", facecolor="#fff4e0",
        zorder=1, label=f"Inner oval ({n_inner:,} pts)"
    )
    ax.add_patch(inner_ellipse)

    # LE circle
    le_circle_patch = mpatches.Circle(
        (le_x, le_y), dc.le_radius,
        linewidth=1.2, edgecolor="#9467bd", facecolor="#ede0f7",
        zorder=2, label=f"LE circle  r={dc.le_radius} ({n_le:,} pts)"
    )
    ax.add_patch(le_circle_patch)

    # TE rectangle
    te_rect_patch = mpatches.FancyBboxPatch(
        (te_x - dc.te_half_x, te_y - dc.te_half_y),
        2 * dc.te_half_x, 2 * dc.te_half_y,
        boxstyle="square,pad=0", linewidth=1.2,
        edgecolor="#17becf", facecolor="#dff5f8",
        zorder=2, label=f"TE rect ({n_te:,} pts)"
    )
    ax.add_patch(te_rect_patch)

    # Sampling points
    ax.scatter(outer_pts[:, 0], outer_pts[:, 1], s=0.4, c="#4c78a8",
               alpha=0.4, label=f"Outer collocation ({n_outer:,})", rasterized=True)
    ax.scatter(inner_pts[:, 0], inner_pts[:, 1], s=0.6, c="#f58518",
               alpha=0.4, label=f"Inner collocation ({n_inner:,})", rasterized=True)
    ax.scatter(le_pts[:,    0], le_pts[:,    1], s=1.5, c="#9467bd",
               alpha=0.4, label=f"LE fine ({n_le:,})", zorder=4)
    ax.scatter(te_pts[:,    0], te_pts[:,    1], s=1.5, c="#17becf",
               alpha=0.4, label=f"TE fine ({n_te:,})", zorder=4)
    ax.scatter(inlet_pts[:, 0], inlet_pts[:, 1], s=6, c="#1f77b4",
               alpha=1.0, label=f"Inlet  Gaussian ({sc.n_inlet:,})", zorder=5)
    ax.scatter(outlet_pts[:,0], outlet_pts[:,1], s=6, c="#d62728",
               alpha=1.0, label=f"Outlet Gaussian ({sc.n_outlet:,})", zorder=5)
    ax.scatter(tb_pts[:,    0], tb_pts[:,    1], s=4, c="#2ca02c",
               alpha=0.8, label=f"Top/bottom uniform ({sc.n_top_bottom:,})", zorder=5)
    ax.plot(airfoil_pts[:, 0], airfoil_pts[:, 1], "r-", linewidth=1.5,
            label=f"Airfoil surface ({n_af} pts)", zorder=6)

    ax.add_patch(_airfoil_patch(ac))
    ax.set_xlim(dc.xmin - 0.02, dc.xmax + 0.02)
    ax.set_ylim(dc.ymin - 0.02, dc.ymax + 0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Computational domain – sampling points overview")
    ax.legend(loc="upper right", markerscale=6, fontsize=9)

    out = OUT / "domain_overview.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[domain] saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2 – named boundary selections
# ---------------------------------------------------------------------------
def plot_boundary_selections():
    dc = cfg.domain
    ac = cfg.airfoil
    sc = cfg.sampling

    rng = np.random.default_rng(cfg.training.random_seed)
    ff_pts = sample_farfield_boundary(dc, sc, ac, rng=rng)

    # Split farfield into named selections (order from sample_farfield_boundary)
    ff_splits  = np.cumsum([sc.n_inlet, sc.n_outlet, sc.n_top_bottom // 2])
    inlet_pts  = ff_pts[:ff_splits[0]]
    outlet_pts = ff_pts[ff_splits[0]:ff_splits[1]]
    tb_pts     = ff_pts[ff_splits[1]:]    # top + bottom combined
    airfoil_pts = naca4_boundary(ac)

    named = [
        (inlet_pts,   "#1f77b4", f"Inlet  Gaussian  (u=u∞, v=0)  [{sc.n_inlet:,}]"),
        (outlet_pts,  "#d62728", f"Outlet Gaussian  (p=0)         [{sc.n_outlet:,}]"),
        (tb_pts,      "#2ca02c", f"Top / bottom  uniform          [{sc.n_top_bottom:,}]"),
        (airfoil_pts, "#ff7f0e", f"Airfoil surface  (no-slip)     [{len(airfoil_pts):,}]"),
    ]

    fig, ax = plt.subplots(figsize=(18, 11))
    for pts, col, lbl in named:
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c=col, label=lbl, zorder=5)
    ax.add_patch(_airfoil_patch(ac))
    ax.set_xlim(dc.xmin - 0.02, dc.xmax + 0.02)
    ax.set_ylim(dc.ymin - 0.02, dc.ymax + 0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Named boundary selections")
    ax.legend(loc="upper right", fontsize=10)

    out = OUT / "domain_selections.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[domain] saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3 – sampling density heatmap
# ---------------------------------------------------------------------------
def plot_sampling_density():
    dc = cfg.domain
    ac = cfg.airfoil
    sc = cfg.sampling

    points = build_sampling_points(dc, ac, sc)

    fig, ax = plt.subplots(figsize=(14, 9))
    h = ax.hist2d(
        points[:, 0], points[:, 1],
        bins=[120, 80],
        range=[[dc.xmin, dc.xmax], [dc.ymin, dc.ymax]],
        cmap="hot_r",
    )
    fig.colorbar(h[3], ax=ax, label="Points per bin")
    ax.add_patch(_airfoil_patch(ac, color="white"))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Collocation-point density")

    out = OUT / "sampling_density.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[domain] saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dde.config.set_random_seed(cfg.training.random_seed)
    print("Generating domain observability figures …")
    plot_domain_overview()
    plot_boundary_selections()
    plot_sampling_density()
    print("Done – outputs in", OUT)
