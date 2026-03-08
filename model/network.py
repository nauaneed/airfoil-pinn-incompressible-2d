"""
Neural network construction and training stages.

Two-stage training strategy
----------------------------
1. Adam   – fast global optimisation
2. L-BFGS-B – local refinement to machine precision
"""
import deepxde as dde

from config import NetworkConfig, TrainingConfig


def build_model(data, net_cfg: NetworkConfig, train_cfg: TrainingConfig) -> dde.Model:
    """Construct the FNN and compile for the Adam stage."""
    net   = dde.nn.FNN(net_cfg.layer_sizes, net_cfg.activation, net_cfg.initializer)
    model = dde.Model(data, net)
    model.compile(
        optimizer    = "adam",
        lr           = train_cfg.adam_lr,
        loss_weights = train_cfg.loss_weights,
    )
    return model


def train_adam(model: dde.Model, train_cfg: TrainingConfig):
    """Run the Adam stage. Returns (losshistory, train_state)."""
    return model.train(
        epochs         = train_cfg.adam_epochs,
        display_every  = train_cfg.adam_display_every,
        model_save_path= train_cfg.model_save_path,
    )


def train_lbfgs(model: dde.Model, train_cfg: TrainingConfig):
    """
    Re-compile for L-BFGS-B and run to convergence.
    Returns (losshistory, train_state).

    Uses dde.optimizers.config.set_LBFGS_options which is backend-agnostic:
    it maps the same keyword arguments to the correct underlying parameters for
    scipy (TF1), tfp (TF2), torch.optim.LBFGS (PyTorch), and paddle.LBFGS.

    For PyTorch specifically:
      - maxcor  → history_size   (memory depth)
      - ftol    → tolerance_change
      - gtol    → tolerance_grad
      - maxls>0 → line_search_fn='strong_wolfe'
      - iter_per_step defaults to 1000 (not exposed by set_LBFGS_options);
        total inner budget = outer_iters x iter_per_step = 1000 x 1000 = 1M.
    """
    dde.optimizers.config.set_LBFGS_options(
        maxcor  = train_cfg.lbfgs_maxcor,
        ftol    = train_cfg.lbfgs_ftol,
        gtol    = train_cfg.lbfgs_gtol,
        maxls   = train_cfg.lbfgs_maxls,
        maxiter = train_cfg.lbfgs_maxiter,
    )

    model.compile(
        optimizer    = "L-BFGS",   # PyTorch: "L-BFGS-B" silently skips training loop; use "L-BFGS"
        loss_weights = train_cfg.loss_weights,
    )

    return model.train(
        display_every  = train_cfg.lbfgs_display_every,
        model_save_path= train_cfg.model_save_path,
    )
