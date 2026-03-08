"""
Incompressible Navier-Stokes residuals in primitive-variable formulation.

Network output layout
---------------------
  y[:, 0]  – u  (x-velocity)
  y[:, 1]  – v  (y-velocity)
  y[:, 2]  – p  (pressure)

Returns 3 scalar residuals per point:
  [continuity, momentum_x, momentum_y]
"""
import deepxde as dde
from config import FluidConfig


def make_navier_stokes(fluid: FluidConfig):
    """
    Return a closure with the fluid parameters baked in, ready to pass
    to dde.data.PDE.
    """
    rho = fluid.rho
    mu  = fluid.mu

    def navier_stokes(x, y):
        u = y[:, 0:1]
        v = y[:, 1:2]
        # p = y[:, 2:3]  (only its gradient appears in momentum)

        u_x  = dde.grad.jacobian(y, x, i=0, j=0)
        u_y  = dde.grad.jacobian(y, x, i=0, j=1)
        v_x  = dde.grad.jacobian(y, x, i=1, j=0)
        v_y  = dde.grad.jacobian(y, x, i=1, j=1)
        p_x  = dde.grad.jacobian(y, x, i=2, j=0)
        p_y  = dde.grad.jacobian(y, x, i=2, j=1)

        u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

        continuity = u_x + v_y
        momentum_x = rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)
        momentum_y = rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)

        return continuity, momentum_x, momentum_y

    return navier_stokes
