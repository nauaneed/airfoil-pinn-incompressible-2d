"""
DeepXDE geometry construction and collocation-point sampling.

Named selections (boundary tags) are kept here so every other module
can reference them by name rather than repeating lambda predicates.
"""
import numpy as np
import deepxde as dde

from config import DomainConfig, AirfoilConfig, SamplingConfig
from geometry.airfoil import naca4_boundary


# ---------------------------------------------------------------------------
# Named boundary predicates
# ---------------------------------------------------------------------------

def is_inlet(x, on_boundary, domain_cfg: DomainConfig):
    return on_boundary and np.isclose(x[0], domain_cfg.xmin)

def is_outlet(x, on_boundary, domain_cfg: DomainConfig):
    return on_boundary and np.isclose(x[0], domain_cfg.xmax)

def is_top_bottom(x, on_boundary, domain_cfg: DomainConfig):
    return on_boundary and (np.isclose(x[1], domain_cfg.ymax) or
                            np.isclose(x[1], domain_cfg.ymin))

def is_airfoil(x, on_boundary, farfield: dde.geometry.Geometry):
    """Any boundary point that does NOT lie on the farfield rectangle."""
    return on_boundary and not farfield.on_boundary(x)


# ---------------------------------------------------------------------------
# Farfield boundary sampling (per-side, with Gaussian on inlet/outlet)
# ---------------------------------------------------------------------------

def _gaussian_edge(x_fixed: float, n: int,
                   y_center: float, y_sigma: float,
                   ymin: float, ymax: float,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Sample n points on the vertical edge x = x_fixed with y drawn from a
    Gaussian N(y_center, y_sigma), clipped to [ymin, ymax].
    """
    y = rng.normal(loc=y_center, scale=y_sigma, size=n)
    y = np.clip(y, ymin, ymax)
    return np.column_stack([np.full(n, x_fixed), y])


def _uniform_edge_h(y_fixed: float, n: int,
                    xmin: float, xmax: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Sample n points on the horizontal edge y = y_fixed, uniform in x."""
    x = rng.uniform(xmin, xmax, size=n)
    return np.column_stack([x, np.full(n, y_fixed)])


def sample_farfield_boundary(
    domain_cfg: DomainConfig,
    sampling_cfg: SamplingConfig,
    airfoil_cfg,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample farfield boundary points with per-side strategies:

      Inlet  (x = xmin) : Gaussian in y, centred at airfoil offset_y
      Outlet (x = xmax) : Gaussian in y, centred at airfoil offset_y
      Top    (y = ymax) : uniform in x
      Bottom (y = ymin) : uniform in x

    Returns (N, 2) array ordered [inlet | outlet | top | bottom].
    """
    if rng is None:
        rng = np.random.default_rng()

    dc = domain_cfg
    sc = sampling_cfg
    height = dc.ymax - dc.ymin
    y_center = airfoil_cfg.offset_y

    inlet_pts  = _gaussian_edge(
        dc.xmin, sc.n_inlet,
        y_center, sc.inlet_sigma_frac  * height,
        dc.ymin, dc.ymax, rng,
    )
    outlet_pts = _gaussian_edge(
        dc.xmax, sc.n_outlet,
        y_center, sc.outlet_sigma_frac * height,
        dc.ymin, dc.ymax, rng,
    )
    n_top = sc.n_top_bottom // 2
    n_bot = sc.n_top_bottom - n_top
    top_pts    = _uniform_edge_h(dc.ymax, n_top, dc.xmin, dc.xmax, rng)
    bottom_pts = _uniform_edge_h(dc.ymin, n_bot, dc.xmin, dc.xmax, rng)

    return np.concatenate([inlet_pts, outlet_pts, top_pts, bottom_pts], axis=0)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_geometry(domain_cfg: DomainConfig, airfoil_cfg: AirfoilConfig):
    """
    Build the fluid domain = farfield rectangle  MINUS  airfoil polygon.

    Returns
    -------
    geom     : CSGDifference  – the full fluid domain
    farfield : Rectangle      – kept for boundary-predicate queries
    """
    farfield = dde.geometry.Rectangle(
        [domain_cfg.xmin, domain_cfg.ymin],
        [domain_cfg.xmax, domain_cfg.ymax],
    )
    airfoil_poly = dde.geometry.Polygon(naca4_boundary(airfoil_cfg))
    geom = dde.geometry.CSGDifference(farfield, airfoil_poly)
    return geom, farfield


def build_sampling_points(
    domain_cfg: DomainConfig,
    airfoil_cfg: AirfoilConfig,
    sampling_cfg: SamplingConfig,
) -> np.ndarray:
    """
    Generate the full set of anchor (collocation) points used in training.

    Sampling hierarchy (coarse → fine):
      - Outer region   : farfield minus inner oval                 (coarse)
      - Inner oval     : ellipse around the airfoil               (medium-fine)
      - LE circle      : disk at the leading edge                 (extra-fine)
      - TE rectangle   : rectangle waking from the trailing edge  (extra-fine)
      - Farfield BCs   : uniform random on the farfield boundary
      - Airfoil BCs    : cosine-clustered airfoil surface points

    The LE and TE sub-regions are *inside* the inner oval, so they add extra
    density there without being subtracted from it — intentional overlap.

    Returns a single (N, 2) array.  Order:
      [outer | inner | le | te | farfield_bnd | airfoil_bnd]
    """
    ac = airfoil_cfg
    dc = domain_cfg
    sc = sampling_cfg

    # Key airfoil positions
    le_center  = [ac.offset_x,               ac.offset_y]
    te_center  = [ac.offset_x + ac.chord,     ac.offset_y]
    mid_center = [ac.offset_x + ac.chord / 2, ac.offset_y]

    farfield     = dde.geometry.Rectangle([dc.xmin, dc.ymin], [dc.xmax, dc.ymax])
    airfoil_poly = dde.geometry.Polygon(naca4_boundary(ac))


    # Extra-fine circle at the leading edge
    le_circle = dde.geometry.Disk(le_center, dc.le_radius)
    le_dom    = dde.geometry.CSGDifference(le_circle, airfoil_poly)

    # Extra-fine rectangle at the trailing edge / wake attachment point
    te_rect = dde.geometry.Rectangle(
        [te_center[0] - dc.te_half_x, te_center[1] - dc.te_half_y],
        [te_center[0] + dc.te_half_x, te_center[1] + dc.te_half_y],
    )
    te_dom = dde.geometry.CSGDifference(te_rect, airfoil_poly)

    # Inner oval shaped around the airfoil (medium-fine density)
    inner_oval = dde.geometry.Ellipse(mid_center, dc.inner_semi_x, dc.inner_semi_y)
    inner_dom  = dde.geometry.CSGDifference(inner_oval, airfoil_poly)
    inner_dom  = dde.geometry.CSGDifference(inner_dom, le_dom)
    inner_dom  = dde.geometry.CSGDifference(inner_dom, te_dom)

    # Outer domain: farfield minus the entire inner oval (and airfoil)
    outer_dom = dde.geometry.CSGDifference(farfield,   inner_oval)
    outer_dom = dde.geometry.CSGDifference(outer_dom,  airfoil_poly)
    outer_dom = dde.geometry.CSGDifference(outer_dom,  le_dom)
    outer_dom = dde.geometry.CSGDifference(outer_dom,  te_dom)

    outer_pts    = outer_dom.random_points(sc.n_outer)
    inner_pts    = inner_dom.random_points(sc.n_inner)
    le_pts       = le_dom.random_points(sc.n_le)
    te_pts       = te_dom.random_points(sc.n_te)
    farfield_pts = sample_farfield_boundary(dc, sc, ac)
    airfoil_pts  = naca4_boundary(ac)

    return np.concatenate([outer_pts, inner_pts, le_pts, te_pts, farfield_pts, airfoil_pts], axis=0)
