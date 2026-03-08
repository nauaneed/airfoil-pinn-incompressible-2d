"""
Global configuration for the NACA airfoil PINN simulation.

All physical parameters, domain bounds, airfoil geometry parameters,
sampling counts, and training hyper-parameters live here so every other
module can import from a single source of truth.
"""
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Domain / physical parameters
# ---------------------------------------------------------------------------
@dataclass
class DomainConfig:
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 0.7

    # Inner oval (ellipse) centred on the chord midpoint
    # Centre is derived from AirfoilConfig at runtime; these are the semi-axes.
    inner_semi_x: float = 0.25   # semi-axis along x  (extends ~0.75c upstream & downstream)
    inner_semi_y: float = 0.09   # semi-axis along y  (covers near-wake + surface layer)

    # Extra-fine region: circle around the leading edge
    le_radius: float = 0.025

    # Extra-fine region: rectangle around the trailing edge
    te_half_x: float = 0.020   # half-width  in x
    te_half_y: float = 0.020   # half-height in y


@dataclass
class FluidConfig:
    rho: float = 1.0   # density
    mu: float  = 0.002  # dynamic viscosity
    u_inf: float = 1.0  # free-stream velocity


# ---------------------------------------------------------------------------
# NACA 4-digit airfoil parameters
# ---------------------------------------------------------------------------
@dataclass
class AirfoilConfig:
    M: int   = 0     # max camber  (* 100)
    P: int   = 0     # max camber position (* 10)
    SS: int  = 12    # max thickness (* 100)
    chord: float = 0.2
    n_panels: int = 250    # half the total boundary points (2*n + 1 total)
    offset_x: float = 0.40  # LE position: 2c upstream (was 0.20 = 1c — too near-field)
    offset_y: float = 0.35


# ---------------------------------------------------------------------------
# Sampling / collocation counts
# ---------------------------------------------------------------------------
@dataclass
class SamplingConfig:
    n_inner: int    =  8_000   # collocation points in inner oval (medium-fine)
    n_outer: int    = 40_000   # collocation points in outer region (coarse)
    n_le: int       =  2_000   # extra-fine points near leading edge
    n_te: int       =  1_500   # extra-fine points near trailing edge
    n_test: int     =  5_000   # test / validation points
    # Airfoil boundary points are determined by AirfoilConfig.n_panels

    # Per-side farfield boundary counts
    n_inlet: int      = 400   # inlet  (x = xmin) – Gaussian in y
    n_outlet: int     = 400   # outlet (x = xmax) – Gaussian in y
    n_top_bottom: int = 480   # top + bottom combined – uniform in x

    # Gaussian parameters for inlet / outlet (centred at airfoil midline)
    # sigma is expressed as a fraction of the domain height
    inlet_sigma_frac: float  = 0.25   # std = sigma_frac * (ymax - ymin)
    outlet_sigma_frac: float = 0.25


# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
@dataclass
class NetworkConfig:
    input_dim: int          = 2
    output_dim: int         = 3        # (u, v, p)  — primitive variables
    hidden_layers: int      = 8
    hidden_units: int       = 40
    activation: str         = "swish"       # swish = x·σ(x): C∞, non-saturating, better
                                            # gradient flow than tanh in deep NS-PINNs
    initializer: str        = "Glorot uniform"  # correct variance for swish near origin

    @property
    def layer_sizes(self) -> List[int]:
        return [self.input_dim] + [self.hidden_units] * self.hidden_layers + [self.output_dim]


# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    # Adam stage
    adam_lr: float              = 5e-4
    adam_epochs: int            = 10_000
    adam_display_every: int     = 100

    # L-BFGS-B stage  (backend-agnostic via dde.optimizers.config.set_LBFGS_options)
    # NOTE: with PyTorch backend, compile with "L-BFGS" not "L-BFGS-B" —
    # the latter silently skips the training loop in deepxde's dispatch.
    lbfgs_maxcor: int        = 50      # history size / num correction pairs
    lbfgs_ftol: float        = 0.0     # tolerance on relative loss change
    lbfgs_gtol: float        = 1e-8    # tolerance on gradient norm
    lbfgs_maxls: int         = 50      # max line-search steps per iteration
    lbfgs_maxiter: int       = 15_000  # outer loop limit in _train_pytorch_lbfgs
    lbfgs_display_every: int = 100

    # Loss weights — must match the order returned by navier_stokes() + build_bcs()
    # PDE residuals (6):                BC terms (7):

    loss_weights: List[float] = field(
        default_factory=lambda: [1,  # [0]  PDE: continuity
                                 1,  # [1]  PDE: momentum x
                                 1,  # [2]  PDE: momentum y
                                 10, # [3]  BC:  inlet      u = u∞
                                 10, # [4]  BC:  inlet      v = 0
                                 10, # [5]  BC:  top/bottom u = u∞
                                 10, # [6]  BC:  top/bottom v = 0
                                 10, # [7]  BC:  outlet     p = 0
                                 10, # [8]  BC:  airfoil    u = 0
                                 10] # [9]  BC:  airfoil    v = 0
    )

    model_save_path: str = "./checkpoints/prim_vars"
    run_dir: str         = "./runs/prim_vars"
    random_seed: int     = 48


# ---------------------------------------------------------------------------
# Convenience: one object that bundles everything
# ---------------------------------------------------------------------------
@dataclass
class Config:
    domain:   DomainConfig   = field(default_factory=DomainConfig)
    fluid:    FluidConfig    = field(default_factory=FluidConfig)
    airfoil:  AirfoilConfig  = field(default_factory=AirfoilConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    network:  NetworkConfig  = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Default global config instance
cfg = Config()
