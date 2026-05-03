import os
import sys
import datetime
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add the parent directory to Python path so we can import pinn_core, pinn_functions, pinn_viz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pinn_core import PINNConfig, PINN, Trainer, Analyzer, make_dataset, ic_bc_points, residual_advection, residual_burgers
import pinn_viz as viz
from pinn_functions import catalog, exact_advection, gaussian
from .database import SessionLocal
from .models import Experiment

def run_simulation(experiment_id: int):
    db = SessionLocal()
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        db.close()
        return

    try:
        # Prepare directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("..", "results", f"exp_{exp.id}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        exp.results_dir = os.path.abspath(results_dir)
        db.commit()

        # Load IC Function
        if exp.u0_name == "gaussian":
            u0_params = exp.u0_params or {"center": 0.0, "sigma": 0.2, "amplitude": 1.0}
            u0_fn = gaussian(**u0_params)
        else:
            u0_fn = catalog.get(exp.u0_name)
            if not u0_fn:
                u0_fn = catalog["gaussian"] # Fallback

        cfg = PINNConfig(
            n_layers=exp.n_layers,
            n_neurons=exp.n_neurons,
            activation=exp.activation,
            lr_adam=exp.lr_adam,
            epochs_adam=exp.epochs_adam,
            epochs_lbfgs=exp.epochs_lbfgs,
            x_min=exp.x_min, x_max=exp.x_max, t_min=exp.t_min, t_max=exp.t_max,
            seed=42
        )

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        model = PINN(cfg)
        
        x_d, t_d, u_d = ic_bc_points(
            cfg, u0_fn=u0_fn, N_ic=200, N_bc=0, bc_left=None, bc_right=None,
        )
        ds = make_dataset(x_d, t_d, u_d, N_f=8000,
                          x_min=cfg.x_min, x_max=cfg.x_max, t_min=cfg.t_min, t_max=cfg.t_max, seed=cfg.seed)

        if exp.model_type == "advection_linear":
            a_vel = exp.a_velocity if exp.a_velocity is not None else 1.0
            res_fn = lambda m, x, t: residual_advection(m, x, t, a=a_vel)
            exact_fn = exact_advection(u0_fn, a=a_vel)
            eq_title = f"Advecção Linear a={a_vel}"
        elif exp.model_type == "burgers_viscous":
            nu = exp.nu_viscosity if exp.nu_viscosity is not None else 0.01 / np.pi
            res_fn = lambda m, x, t: residual_burgers(m, x, t, nu=nu)
            exact_fn = None
            eq_title = f"Burgers Viscosa ν={nu:.4f}"
        elif exp.model_type == "burgers_inviscid":
            res_fn = lambda m, x, t: residual_burgers(m, x, t, nu=0.0)
            exact_fn = None
            eq_title = "Burgers Invíscida"
        else:
            raise ValueError(f"Unknown model_type: {exp.model_type}")

        trainer = Trainer(cfg, model, ds, res_fn)
        trainer.train(verbose=False)
        
        analyzer = Analyzer(cfg, model, res_fn)
        
        # Save plots
        import matplotlib.pyplot as plt
        
        # Slices
        fig = viz.plot_slices(
            model, cfg.x_min, cfg.x_max, t_vals=[0.0, 0.25, 0.5, 0.75, 1.0],
            exact_fn=exact_fn,
            title=eq_title,
            savefig=os.path.join(results_dir, "slices.png"),
        )
        plt.close(fig)

        # 3D Surface
        fig = viz.plot_surface_3d(
            model, cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max,
            title="Superfície 3D",
            savefig=os.path.join(results_dir, "surface_3d.png"),
        )
        plt.close(fig)

        # Animation (GIF)
        anim = viz.animate_solution(
            model, cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max,
            exact_fn=exact_fn,
            title=eq_title,
            interval_ms=50,
            N_frames=120,
            savefig=os.path.join(results_dir, "animation.gif")
        )

        # 3D Animation (GIF)
        anim_3d = viz.animate_surface_3d(
            model, cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max,
            title="Superfície 3D Rotativa",
            savefig=os.path.join(results_dir, "animation_3d.gif")
        )

        # Dashboard
        fig = viz.dashboard(
            model, analyzer, trainer.history,
            t_fourier=0.5,
            exact_fn=exact_fn,
            title=f"Dashboard — Exp {exp.id} ({eq_title})",
            savefig=os.path.join(results_dir, "dashboard.png"),
        )
        plt.close(fig)

        # Update DB
        exp.loss_final = trainer.history["loss"][-1]
        exp.nu_numerical = analyzer.nu_numerical(0.5)
        exp.diagnostico = analyzer.diagnose(0.5)
        exp.status = "COMPLETED"
        db.commit()

    except Exception as e:
        exp.status = "FAILED"
        exp.diagnostico = str(e)
        db.commit()
        print(f"Experiment {experiment_id} failed: {e}")
    finally:
        db.close()
