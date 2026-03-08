"""
Boundary condition assembly for the NACA airfoil PINN.

Primitive-variable formulation: network outputs are (u, v, p) directly,
so all BCs are plain DirichletBCs — no stream function, no gauge issues,
no OperatorBCs needed.

Named selections
----------------
  inlet      (x = xmin)       →  u = u_inf,  v = 0
  top_bottom (y = ymin/ymax)  →  u = u_inf,  v = 0
  outlet     (x = xmax)       →  p = 0
  airfoil    (surface)         →  u = 0,      v = 0  (no-slip)
"""
import deepxde as dde
from functools import partial

from config import DomainConfig, FluidConfig
from geometry.domain import is_inlet, is_outlet, is_top_bottom, is_airfoil


def build_bcs(geom, farfield,
              domain_cfg: DomainConfig,
              fluid_cfg: FluidConfig,
              airfoil_cfg=None):  # kept for API compatibility, unused
    """
    Assemble and return the list of boundary conditions.

    BC order  (must match loss_weights in TrainingConfig):
      [0] inlet      u = u_inf
      [1] inlet      v = 0
      [2] top/bottom u = u_inf
      [3] top/bottom v = 0
      [4] outlet     p = 0
      [5] airfoil    u = 0
      [6] airfoil    v = 0
    """
    u_inf = fluid_cfg.u_inf

    inlet_pred      = partial(is_inlet,      domain_cfg=domain_cfg)
    outlet_pred     = partial(is_outlet,     domain_cfg=domain_cfg)
    top_bottom_pred = partial(is_top_bottom, domain_cfg=domain_cfg)
    airfoil_pred    = partial(is_airfoil,    farfield=farfield)

    bcs = [
        # Inlet
        dde.DirichletBC(geom, lambda x: u_inf, inlet_pred,      component=0),
        dde.DirichletBC(geom, lambda x: 0.0,   inlet_pred,      component=1),

        # Top / bottom
        dde.DirichletBC(geom, lambda x: u_inf, top_bottom_pred, component=0),
        dde.DirichletBC(geom, lambda x: 0.0,   top_bottom_pred, component=1),

        # Outlet: p = 0
        dde.DirichletBC(geom, lambda x: 0.0,   outlet_pred,     component=2),

        # Airfoil no-slip
        dde.DirichletBC(geom, lambda x: 0.0,   airfoil_pred,    component=0),
        dde.DirichletBC(geom, lambda x: 0.0,   airfoil_pred,    component=1),
    ]
    return bcs