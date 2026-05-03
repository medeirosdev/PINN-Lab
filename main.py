"""
main.py
=======
Teste da PINN para Advecção Linear:  U_t + a·U_x = 0

Treina a PINN, gera gráficos e salva tudo em results/advection_<timestamp>/
"""

import os
import json
import datetime
import numpy as np

import torch

from pinn_core import (
    PINNConfig, PINN, Trainer, Analyzer,
    make_dataset, ic_bc_points, residual_advection,
)
import pinn_viz as viz
from pinn_functions import gaussian, sine, exact_advection, torch_fn, catalog


# ═══════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES DO EXPERIMENTO
# ═══════════════════════════════════════════════════════════════

# --- Física ---
A_VELOCITY = 1.0                      # velocidade de advecção
U0_NAME    = "sine"                   # condição inicial (do catálogo)
U0_FN      = sine(k=1.0, amplitude=1.0)

# --- Domínio ---
X_MIN, X_MAX = 0.0, 2.0
T_MIN, T_MAX = 0.0, 1.0

# --- Dados ---
N_IC = 200                            # pontos na condição inicial
N_F  = 10_000                         # pontos de colocação

# --- Rede ---
N_LAYERS   = 4
N_NEURONS  = 10
ACTIVATION = "tanh"                   # tanh | sin | relu | swish

# --- Treino ---
LR_ADAM      = 1e-3
EPOCHS_ADAM  = 10_000
EPOCHS_LBFGS = 500                    # 0 = desligado
LAMBDA_PHYS  = 1.0
SEED         = 42


# ═══════════════════════════════════════════════════════════════
#  PASTA DE RESULTADOS
# ═══════════════════════════════════════════════════════════════

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("results", f"advection_{timestamp}")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Resultados serão salvos em: {RESULTS_DIR}/\n")


# ═══════════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════════

cfg = PINNConfig(
    n_layers=N_LAYERS,
    n_neurons=N_NEURONS,
    activation=ACTIVATION,
    lr_adam=LR_ADAM,
    epochs_adam=EPOCHS_ADAM,
    epochs_lbfgs=EPOCHS_LBFGS,
    lambda_phys=LAMBDA_PHYS,
    seed=SEED,
    x_min=X_MIN, x_max=X_MAX,
    t_min=T_MIN, t_max=T_MAX,
)

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

# ── modelo ──
model = PINN(cfg)
print(f"PINN | camadas={cfg.n_layers} | neurônios={cfg.n_neurons} | "
      f"ativação={cfg.activation} | params={model.count_params():,}")

# ── dataset ──
x_d, t_d, u_d = ic_bc_points(
    cfg, u0_fn=U0_FN, N_ic=N_IC, N_bc=0, bc_left=None, bc_right=None,
)
ds = make_dataset(x_d, t_d, u_d, N_f=N_F,
                  x_min=X_MIN, x_max=X_MAX, t_min=T_MIN, t_max=T_MAX, seed=SEED)

# ── função de resíduo ──
res_fn = lambda m, x, t: residual_advection(m, x, t, a=A_VELOCITY)

# ── solução exata ──
exact_fn = exact_advection(U0_FN, a=A_VELOCITY)


# ═══════════════════════════════════════════════════════════════
#  TREINO
# ═══════════════════════════════════════════════════════════════

trainer = Trainer(cfg, model, ds, res_fn)
trainer.train(verbose=True)


# ═══════════════════════════════════════════════════════════════
#  ANÁLISE
# ═══════════════════════════════════════════════════════════════

analyzer = Analyzer(cfg, model, res_fn)


# ═══════════════════════════════════════════════════════════════
#  GRÁFICOS → salvos em RESULTS_DIR
# ═══════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")     # backend sem janela (salva direto)
import matplotlib.pyplot as plt

T_SLICES = [0.0, 0.25, 0.5, 0.75, 1.0]

# 1. Condição inicial
from pinn_functions import plot_function
fig = plot_function(U0_FN, x_min=X_MIN, x_max=X_MAX,
                    title=f"Condição Inicial: {U0_NAME}")
fig.savefig(os.path.join(RESULTS_DIR, "01_condicao_inicial.png"), dpi=150)
plt.close(fig)

# 2. Histórico de loss
fig = viz.plot_loss(trainer.history,
                    title="Histórico de Loss — Advecção Linear",
                    savefig=os.path.join(RESULTS_DIR, "02_loss.png"))
plt.close(fig)

# 3. Solução em fatias (PINN vs Exata)
fig = viz.plot_slices(
    model, X_MIN, X_MAX, t_vals=T_SLICES,
    exact_fn=exact_fn,
    title=f"Advecção Linear  U_t + {A_VELOCITY}·U_x = 0\nPINN vs Solução Exata",
    savefig=os.path.join(RESULTS_DIR, "03_solucao_fatias.png"),
)
plt.close(fig)

# 4. Heatmap u(x, t)
fig = viz.plot_heatmap(
    model, X_MIN, X_MAX, T_MIN, T_MAX,
    title="Advecção Linear — u(x, t)",
    savefig=os.path.join(RESULTS_DIR, "04_heatmap.png"),
)
plt.close(fig)

# 5. Superfície 3D
fig = viz.plot_surface_3d(
    model, X_MIN, X_MAX, T_MIN, T_MAX,
    title="Advecção Linear — Superfície 3D",
    savefig=os.path.join(RESULTS_DIR, "05_superficie_3d.png"),
)
plt.close(fig)

# 6. Erro pontual
fig = viz.plot_error(
    model, exact_fn, X_MIN, X_MAX, t_vals=T_SLICES,
    title="Erro Pontual |u_PINN − u_exata|",
    savefig=os.path.join(RESULTS_DIR, "06_erro_pontual.png"),
)
plt.close(fig)

# 7. Resíduo no espaço
fig = viz.plot_residual(
    analyzer, t_vals=[0.25, 0.5, 0.75],
    title="Resíduo r(x) = U_t + a·U_x",
    savefig=os.path.join(RESULTS_DIR, "07_residuo.png"),
)
plt.close(fig)

# 8. Fourier do resíduo (painel diagnóstico)
for tv in [0.25, 0.5, 0.75]:
    fig = viz.plot_fourier_panel(
        analyzer, t_val=tv,
        savefig=os.path.join(RESULTS_DIR, f"08_fourier_t{tv:.2f}.png"),
    )
    plt.close(fig)

# 9. Energia L² vs Gronwall
fig = viz.plot_energy(
    analyzer, norm_a_C1=abs(A_VELOCITY),
    title="Energia ||u||²_L2 vs Cota de Gronwall",
    savefig=os.path.join(RESULTS_DIR, "09_energia_gronwall.png"),
)
plt.close(fig)

# 10. Dashboard completo
fig = viz.dashboard(
    model, analyzer, trainer.history,
    t_fourier=0.5,
    exact_fn=exact_fn,
    title=f"Dashboard — Advecção Linear (a={A_VELOCITY})",
    savefig=os.path.join(RESULTS_DIR, "10_dashboard.png"),
)
plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  SALVAR METADADOS DO EXPERIMENTO
# ═══════════════════════════════════════════════════════════════

meta = {
    "equacao": "advecção linear  U_t + a·U_x = 0",
    "a": A_VELOCITY,
    "u0": U0_NAME,
    "dominio": {"x": [X_MIN, X_MAX], "t": [T_MIN, T_MAX]},
    "rede": {
        "camadas": N_LAYERS,
        "neuronios": N_NEURONS,
        "ativacao": ACTIVATION,
        "parametros": model.count_params(),
    },
    "treino": {
        "epocas_adam": EPOCHS_ADAM,
        "epocas_lbfgs": EPOCHS_LBFGS,
        "lr_adam": LR_ADAM,
        "lambda_phys": LAMBDA_PHYS,
        "seed": SEED,
    },
    "resultados": {
        "loss_final": trainer.history["loss"][-1],
        "loss_data_final": trainer.history["loss_data"][-1],
        "loss_phys_final": trainer.history["loss_phys"][-1],
        "diagnostico_t05": analyzer.diagnose(0.5),
        "nu_num_t05": analyzer.nu_numerical(0.5),
    },
    "timestamp": timestamp,
}

meta_path = os.path.join(RESULTS_DIR, "experiment.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

# Salvar histórico completo
hist_path = os.path.join(RESULTS_DIR, "loss_history.npz")
np.savez(hist_path, **{k: np.array(v) for k, v in trainer.history.items()})

# Salvar pesos do modelo
model_path = os.path.join(RESULTS_DIR, "model.pt")
torch.save(model.state_dict(), model_path)


# ═══════════════════════════════════════════════════════════════
#  RESUMO
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EXPERIMENTO CONCLUÍDO")
print("=" * 60)
print(f"  Pasta:        {RESULTS_DIR}/")
print(f"  Loss final:   {meta['resultados']['loss_final']:.4e}")
print(f"  ν_num (t=0.5): {meta['resultados']['nu_num_t05']:.4e}")
print(f"  Diagnóstico:  {meta['resultados']['diagnostico_t05']}")
print()
print("Arquivos gerados:")
for f in sorted(os.listdir(RESULTS_DIR)):
    size = os.path.getsize(os.path.join(RESULTS_DIR, f))
    print(f"  {f:<35s} {size / 1024:.1f} KB")
print("=" * 60)
