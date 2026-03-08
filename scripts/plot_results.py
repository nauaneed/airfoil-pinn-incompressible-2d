"""
scripts/plot_results.py
=======================
Observability script: load a saved PINN checkpoint and generate result plots.

Outputs (written to outputs/results/):
  field_u.png     – horizontal velocity
  field_v.png     – vertical velocity
  field_p.png     – pressure
  streamlines.png – streamlines over speed magnitude

Run from the project root::

    python scripts/plot_results.py                       # uses default config
    python scripts/plot_results.py --ckpt path/to/model  # custom checkpoint
"""
import sys
import argparse
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import deepxde as dde

from config import cfg
from geometry.domain import build_geometry
from physics.equations import make_navier_stokes
from model.network import build_model
from postprocess.predict import build_grid, predict_fields
from postprocess.plots import plot_all_fields


def parse_args():
    p = argparse.ArgumentParser(description="Plot PINN results from a saved checkpoint.")
    p.add_argument("--ckpt",  type=str, default=None,
                   help="Path prefix of the saved model (without .pt / -best.pt).")
    p.add_argument("--dx",    type=float, default=0.005,
                   help="Grid spacing in x (default 0.005).")
    p.add_argument("--dy",    type=float, default=0.005,
                   help="Grid spacing in y (default 0.005).")
    p.add_argument("--out",   type=str, default="outputs/results",
                   help="Output directory for PNG files.")
    return p.parse_args()


def main():
    args = parse_args()

    dde.config.set_random_seed(cfg.training.random_seed)
    dde.config.set_default_float("float64")

    # --- Rebuild data / model skeleton (needed to restore weights) ---
    # BCs are not needed for inference — pass empty list to avoid the
    # DeepXDE filter/predicate machinery which requires boundary sampling.
    geom, _ = build_geometry(cfg.domain, cfg.airfoil)
    pde     = make_navier_stokes(cfg.fluid)

    data  = dde.data.PDE(geom, pde, [], num_domain=0, num_boundary=0,
                         num_test=0)
    model = build_model(data, cfg.network, cfg.training)

    # --- Restore weights ---
    ckpt = args.ckpt or str(pathlib.Path(cfg.training.model_save_path) / "model")
    print(f"[results] restoring weights from: {ckpt}")
    model.restore(ckpt, verbose=1)

    # --- Predict on grid ---
    x, y, X = build_grid(cfg.domain, dx=args.dx, dy=args.dy)
    print(f"[results] predicting on {len(X):,} grid points …")
    fields = predict_fields(model, X, x, y)

    # --- Plot ---
    plot_all_fields(fields, x, y, cfg.airfoil, out_path=args.out)
    print(f"[results] all figures saved to {args.out}/")


if __name__ == "__main__":
    main()
