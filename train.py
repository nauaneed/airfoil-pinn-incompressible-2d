"""
train.py – main entry point for the NACA airfoil PINN
======================================================

Training pipeline
-----------------
  1.  Build geometry (farfield rectangle minus NACA airfoil polygon).
  2.  Generate collocation / anchor points.
  3.  Assemble the PDE and boundary conditions.
  4.  Construct the FNN and compile.
  5.  Adam stage   (fast global convergence).
  6.  L-BFGS-B stage (local refinement).
  7.  Save results and generate field plots.

Usage::

    python train.py
"""
import argparse
import pathlib
import deepxde as dde

from pathlib import Path
from config import cfg
from geometry.domain import build_geometry, build_sampling_points
from physics.equations import make_navier_stokes
from physics.boundaries import build_bcs
from model.network import build_model, train_adam, train_lbfgs
from postprocess.predict import build_grid, predict_fields
from postprocess.plots import plot_all_fields


def parse_args():
    p = argparse.ArgumentParser(description="Train the NACA airfoil PINN.")
    p.add_argument(
        "--resume", type=str, default=None, metavar="CKPT",
        help=(
            "Path to a saved checkpoint (without extension) to restore before "
            "training.  The Adam stage is skipped and training resumes directly "
            "with L-BFGS-B.  Example: --resume ./-10000"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    dde.config.set_random_seed(cfg.training.random_seed)
    dde.config.set_default_float("float64")

    # ------------------------------------------------------------------
    # Geometry & sampling points
    # ------------------------------------------------------------------
    print("[train] Building geometry …")
    geom, farfield = build_geometry(cfg.domain, cfg.airfoil)

    print("[train] Generating collocation points …")
    anchors = build_sampling_points(cfg.domain, cfg.airfoil, cfg.sampling)
    print(f"[train] Total anchor points: {len(anchors):,}")

    # ------------------------------------------------------------------
    # PDE & BCs
    # ------------------------------------------------------------------
    pde = make_navier_stokes(cfg.fluid)
    bcs = build_bcs(geom, farfield, cfg.domain, cfg.fluid, cfg.airfoil)

    data = dde.data.PDE(
        geom,
        pde,
        bcs,
        num_domain   = 0,
        num_boundary = 0,
        num_test     = cfg.sampling.n_test,
        anchors      = anchors,
    )
    print(f"[train] Training points: {len(data.train_x_all):,}  "
          f"| Test points: {cfg.sampling.n_test:,}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("[train] Building model …")
    model = build_model(data, cfg.network, cfg.training)

    # ------------------------------------------------------------------
    # Stage 1 – Adam  (skipped when resuming from checkpoint)
    # ------------------------------------------------------------------
    if args.resume:
        ckpt = args.resume
        if not ckpt.endswith(".pt"):
            ckpt = ckpt + ".pt"
        print(f"[train] Restoring checkpoint: {ckpt}")
        model.restore(ckpt, verbose=1)
        print("[train] Skipping Adam stage.")
    else:
        print(f"\n[train] === Adam stage  ({cfg.training.adam_epochs:,} epochs) ===")
        lh_adam, ts_adam = train_adam(model, cfg.training)
        dde.saveplot(lh_adam, ts_adam, issave=True, isplot=False,
                     output_dir=cfg.training.run_dir)

    # ------------------------------------------------------------------
    # Stage 2 – L-BFGS-B
    # ------------------------------------------------------------------
    print("\n[train] === L-BFGS-B stage ===")
    lh_lbfgs, ts_lbfgs = train_lbfgs(model, cfg.training)
    dde.saveplot(lh_lbfgs, ts_lbfgs, issave=True, isplot=False,
                 output_dir=cfg.training.run_dir)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    print("\n[train] Generating result plots …")
    x, y, X   = build_grid(cfg.domain, dx=0.005, dy=0.005)
    fields     = predict_fields(model, X, x, y)
    out_dir    = pathlib.Path(cfg.training.run_dir) / "results"
    plot_all_fields(fields, x, y, cfg.airfoil, out_path=out_dir)

    print(f"\n[train] Done.  Figures in {out_dir}")


if __name__ == "__main__":
    Path(cfg.training.model_save_path).mkdir(parents=True, exist_ok=True)
    Path(cfg.training.run_dir).mkdir(parents=True, exist_ok=True)
    main()
