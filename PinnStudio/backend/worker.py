import os
import datetime
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pinn_core import PINNConfig, PINN, Trainer, Analyzer, make_dataset, ic_bc_points, residual_advection, residual_burgers
from . import pinn_viz as viz
from .pinn_functions import catalog, gaussian, get_exact_fn
from .database import SessionLocal
from .models import Experiment


def run_simulation(experiment_id: int, progress_dict: dict = None):
    db = SessionLocal()
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        db.close()
        return

    try:
        exp.status = "RUNNING"
        db.commit()

        # Diretório de resultados — path absoluto relativo à raiz do projeto
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(project_root, "results", f"exp_{exp.id}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        exp.results_dir = results_dir
        db.commit()

        # ── Condição inicial ──────────────────────────────────────────────────
        if exp.u0_name == "gaussian":
            u0_params = exp.u0_params or {"center": 0.0, "sigma": 0.2, "amplitude": 1.0}
            u0_fn = gaussian(**u0_params)
        else:
            u0_fn = catalog.get(exp.u0_name)
            if u0_fn is None:
                u0_fn = catalog["gaussian"]

        # ── Configuração e modelo ─────────────────────────────────────────────
        cfg = PINNConfig(
            n_layers=exp.n_layers,
            n_neurons=exp.n_neurons,
            activation=exp.activation,
            lr_adam=exp.lr_adam,
            epochs_adam=exp.epochs_adam,
            epochs_lbfgs=exp.epochs_lbfgs,
            x_min=exp.x_min, x_max=exp.x_max,
            t_min=exp.t_min, t_max=exp.t_max,
            seed=exp.seed if exp.seed is not None else 42,
        )

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        model = PINN(cfg)

        x_d, t_d, u_d = ic_bc_points(cfg, u0_fn=u0_fn, N_ic=200, N_bc=0,
                                      bc_left=None, bc_right=None)
        n_f = exp.n_collocation if exp.n_collocation is not None else 8000
        ds = make_dataset(x_d, t_d, u_d, N_f=n_f,
                          x_min=cfg.x_min, x_max=cfg.x_max,
                          t_min=cfg.t_min, t_max=cfg.t_max,
                          seed=cfg.seed)

        # ── Resíduo e equação título ──────────────────────────────────────────
        if exp.model_type == "advection_linear":
            a_vel = exp.a_velocity if exp.a_velocity is not None else 1.0
            res_fn  = lambda m, x, t: residual_advection(m, x, t, a=a_vel)
            eq_title = f"Advecção Linear a={a_vel}"
        elif exp.model_type == "burgers_viscous":
            nu = exp.nu_viscosity if exp.nu_viscosity is not None else 0.01 / np.pi
            res_fn  = lambda m, x, t: residual_burgers(m, x, t, nu=nu)
            eq_title = f"Burgers Viscosa ν={nu:.4f}"
        elif exp.model_type == "burgers_inviscid":
            res_fn  = lambda m, x, t: residual_burgers(m, x, t, nu=0.0)
            eq_title = "Burgers Invíscida"
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {exp.model_type}")

        # ── Callback de progresso ─────────────────────────────────────────────
        last_adam_loss = [None]  # mutable closure

        def on_epoch_end(epoch, total, loss_val):
            if progress_dict is not None:
                if epoch == -1:
                    # L-BFGS started — keep Adam's final loss visible
                    progress_dict[experiment_id] = {
                        "epoch": 0, "total_epochs": 0,
                        "loss": last_adam_loss[0], "phase": "L-BFGS"
                    }
                else:
                    last_adam_loss[0] = loss_val
                    progress_dict[experiment_id] = {
                        "epoch": epoch + 1, "total_epochs": total,
                        "loss": loss_val, "phase": "Adam"
                    }

        # ── Treinamento ───────────────────────────────────────────────────────
        trainer = Trainer(cfg, model, ds, res_fn)
        start_time = time.time()
        trainer.train(verbose=False, on_epoch_end=on_epoch_end)
        time_taken = time.time() - start_time

        # ── Fase pós-treino: geração de gráficos ─────────────────────────────
        if progress_dict is not None:
            progress_dict[experiment_id] = {
                "epoch": 0, "total_epochs": 0,
                "loss": trainer.history["loss"][-1] if trainer.history["loss"] else None,
                "phase": "Gerando graficos"
            }

        # ── Métricas de loss ──────────────────────────────────────────────────
        final_loss   = trainer.history["loss"][-1]      if trainer.history["loss"]       else None
        final_loss_f = trainer.history["loss_phys"][-1] if trainer.history["loss_phys"]  else None
        final_loss_u = trainer.history["loss_data"][-1] if trainer.history["loss_data"]  else None

        # ── Solução exata (se disponível) ─────────────────────────────────────
        # u0_fn é passada explicitamente para garantir consistência quando o usuário
        # usa parâmetros customizados (e.g. gaussiana deslocada)
        exact_fn, has_exact = get_exact_fn(
            model_type=exp.model_type,
            u0_name=exp.u0_name,
            a_velocity=exp.a_velocity if exp.a_velocity is not None else 1.0,
            nu_viscosity=exp.nu_viscosity if exp.nu_viscosity is not None else 0.01 / np.pi,
            u0_fn=u0_fn,
        )

        # ── Análise ───────────────────────────────────────────────────────────
        analyzer = Analyzer(cfg, model, res_fn)

        # ── Métricas de erro (quando existe solução exata) ────────────────────
        l2_err   = None
        linf_err = None
        if has_exact:
            metrics = analyzer.compute_error_metrics(exact_fn, N=512)
            l2_err   = float(metrics["l2_max"])
            linf_err = float(metrics["linf_max"])

        # ── Persistência do histórico ─────────────────────────────────────────
        np.save(os.path.join(results_dir, "history.npy"), trainer.history)

        # ── Gráficos padrão ───────────────────────────────────────────────────
        fig = viz.plot_slices(
            model, cfg.x_min, cfg.x_max,
            t_vals=[0.0, 0.25, 0.5, 0.75, 1.0],
            exact_fn=exact_fn,
            title=eq_title,
            savefig=os.path.join(results_dir, "slices.png"),
        )
        plt.close(fig)

        fig = viz.plot_surface_3d(
            model, cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max,
            title="Superfície 3D",
            savefig=os.path.join(results_dir, "surface_3d.png"),
        )
        plt.close(fig)

        anim = viz.animate_solution(
            model, cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max,
            exact_fn=exact_fn,
            title=eq_title,
            interval_ms=50,
            N_frames=120,
            savefig=os.path.join(results_dir, "animation.gif"),
        )

        anim_3d = viz.animate_surface_3d(
            model, cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max,
            title="Superfície 3D Rotativa",
            savefig=os.path.join(results_dir, "animation_3d.gif"),
        )

        fig = viz.dashboard(
            model, analyzer, trainer.history,
            t_fourier=0.5,
            exact_fn=exact_fn,
            title=f"Dashboard — Exp {exp.id} ({eq_title})",
            savefig=os.path.join(results_dir, "dashboard.png"),
        )
        plt.close(fig)

        fig_spec = viz.plot_spectral_analysis(
            analyzer, t_val=0.5,
            savefig=os.path.join(results_dir, "spectral.png"),
        )
        plt.close(fig_spec)

        # ── Gráficos de comparação (apenas quando há solução exata) ───────────
        if has_exact:
            fig = viz.plot_error_heatmap(
                model, exact_fn,
                cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max,
                title=f"Comparação PINN vs Exata — {eq_title}",
                savefig=os.path.join(results_dir, "error_heatmap.png"),
            )
            plt.close(fig)

            fig = viz.plot_error_curve(
                metrics,
                title=f"Evolução do Erro — {eq_title}",
                savefig=os.path.join(results_dir, "error_curve.png"),
            )
            plt.close(fig)

            fig = viz.plot_comparison_panel(
                model, exact_fn, analyzer, metrics,
                cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max,
                title=f"Painel de Comparação — Exp {exp.id} ({eq_title})",
                savefig=os.path.join(results_dir, "comparison.png"),
            )
            plt.close(fig)

            fig = viz.plot_surface_3d_trio(
                model, exact_fn,
                cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max,
                title=f"Superfícies 3D — Exp {exp.id} ({eq_title})",
                savefig=os.path.join(results_dir, "surface_3d_trio.png"),
            )
            plt.close(fig)

        # ── Gráficos interativos 3D (Plotly HTML) ────────────────────────────
        viz.plot_surface_3d_interactive(
            model,
            cfg.x_min, cfg.x_max, cfg.t_min, cfg.t_max,
            exact_fn=exact_fn if has_exact else None,
            title=f"Superfície 3D Interativa — Exp {exp.id} ({eq_title})",
            savefig=os.path.join(results_dir, "surface_3d_interactive.html"),
        )

        # ── Atualiza banco de dados ───────────────────────────────────────────
        exp.status = "COMPLETED"
        exp.loss_final      = final_loss
        exp.loss_f_final    = final_loss_f
        exp.loss_u_final    = final_loss_u
        exp.time_taken_sec  = time_taken
        exp.has_exact_solution = has_exact
        if l2_err is not None:
            exp.l2_error = l2_err
        if linf_err is not None:
            exp.linf_error = linf_err
        exp.nu_numerical = analyzer.nu_numerical(0.5)
        exp.mu_numerical = analyzer.mu_numerical(0.5)
        exp.diagnostico  = analyzer.diagnose(0.5)
        db.commit()

    except Exception as e:
        exp.status = "FAILED"
        exp.diagnostico = str(e)
        db.commit()
        print(f"Experiment {experiment_id} failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        db.close()
