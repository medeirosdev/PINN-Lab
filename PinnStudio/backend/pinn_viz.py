"""
pinn_viz.py
===========
Biblioteca de visualização para PINNs — MS901/MT861.

Funções:
  plot_loss()             — histórico de loss (escala log)
  plot_slices()           — u(x, t_i) em vários instantes
  plot_heatmap()          — mapa de calor u(x, t) 2D
  plot_surface_3d()       — superfície 3D u(x, t)
  animate_solution()      — animação da evolução u(x, t)
  plot_residual()         — resíduo no espaço
  plot_fourier()          — espectro de Fourier do resíduo
  plot_fourier_panel()    — resíduo + FFT lado a lado
  plot_energy()           — norma L² vs cota de Gronwall
  plot_error()            — erro pontual PINN vs solução exata
  dashboard()             — painel completo (solução + loss + Fourier)
  plot_spectral_analysis()  — difusão e dispersão numérica via Fourier
  plot_error_heatmap()      — mapas de calor: PINN | exata | |erro|
  plot_error_curve()        — curvas L²(t) e L∞(t) vs tempo
  plot_comparison_panel()   — painel 2×2 de comparação PINN vs exata
  plot_surface_3d_trio()    — três superfícies 3D: PINN | exata | |erro|
  plot_surface_3d_interactive() — superfície(s) 3D interativas via Plotly (HTML)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import torch

# Estilo global
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

_CMAP_SOL = "RdBu_r"
_CMAP_RES = "coolwarm"


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários internos
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  → salvo em {path}")


def _grid_xt(
    model,
    x_min: float, x_max: float,
    t_min: float, t_max: float,
    Nx: int = 200, Nt: int = 200,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Avalia o modelo em grade (x, t) e retorna X, T, U como numpy."""
    xv = torch.linspace(x_min, x_max, Nx)
    tv = torch.linspace(t_min, t_max, Nt)
    X, T = torch.meshgrid(xv, tv, indexing="ij")
    dev = torch.device(device)
    with torch.no_grad():
        U = model(X.reshape(-1, 1).to(dev), T.reshape(-1, 1).to(dev))
    return X.numpy(), T.numpy(), U.cpu().reshape(Nx, Nt).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 1. HISTÓRICO DE LOSS
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss(
    history: Dict[str, List[float]],
    title: str = "Histórico de Treinamento",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Plota loss total, de dados e de física em escala logarítmica.

    Parâmetros
    ----------
    history : dict com chaves 'loss', 'loss_data', 'loss_phys'
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    eps = range(len(history["loss"]))

    ax.semilogy(eps, history["loss"],      lw=2,   label="Total")
    ax.semilogy(eps, history["loss_data"], lw=1.5, ls="--", label="Dados (CI+CC)")
    ax.semilogy(eps, history["loss_phys"], lw=1.5, ls=":",  label="Física (resíduo)")

    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    _save(fig, savefig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. SOLUÇÃO EM FATIAS DE TEMPO
# ─────────────────────────────────────────────────────────────────────────────

def plot_slices(
    model,
    x_min: float, x_max: float,
    t_vals: Optional[List[float]] = None,
    exact_fn: Optional[Callable] = None,
    N: int = 512,
    title: str = "PINN — u(x, t)",
    device: str = "cpu",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Plota u(x) para cada instante em t_vals.

    Parâmetros
    ----------
    exact_fn : callable (x_np, t_val) → u_np  (opcional, traça solução exata)
    """
    if t_vals is None:
        t_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

    n = len(t_vals)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13)
    dev = torch.device(device)
    x_np = np.linspace(x_min, x_max, N)

    for ax, t_val in zip(axes, t_vals):
        x_t = torch.tensor(x_np, dtype=torch.float32).reshape(-1, 1).to(dev)
        t_t = torch.full_like(x_t, t_val)
        with torch.no_grad():
            u_pinn = model(x_t, t_t).cpu().numpy().ravel()

        ax.plot(x_np, u_pinn, "tab:red", lw=2, label="PINN")

        if exact_fn is not None:
            u_ex = exact_fn(x_np, t_val)
            ax.plot(x_np, u_ex, "k--", lw=1.5, label="Exata")

        ax.set_title(f"t = {t_val:.2f}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("u(x, t)")
    plt.tight_layout()
    _save(fig, savefig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAPA DE CALOR 2D
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(
    model,
    x_min: float, x_max: float,
    t_min: float, t_max: float,
    Nx: int = 200, Nt: int = 200,
    title: str = "PINN — u(x, t)",
    device: str = "cpu",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """Mapa de calor u(x, t) no domínio completo."""
    X, T, U = _grid_xt(model, x_min, x_max, t_min, t_max, Nx, Nt, device)

    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.contourf(T, X, U, levels=120, cmap=_CMAP_SOL)
    fig.colorbar(c, ax=ax, label="u(x, t)", shrink=0.85)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)
    plt.tight_layout()
    _save(fig, savefig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. SUPERFÍCIE 3D
# ─────────────────────────────────────────────────────────────────────────────

def plot_surface_3d(
    model,
    x_min: float, x_max: float,
    t_min: float, t_max: float,
    Nx: int = 100, Nt: int = 100,
    title: str = "PINN — Superfície u(x, t)",
    elev: float = 28.0,
    azim: float = -60.0,
    device: str = "cpu",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Superfície 3D interativa (matplotlib) de u(x, t).

    Parâmetros
    ----------
    elev, azim : ângulo de visão inicial (graus)
    """
    X, T, U = _grid_xt(model, x_min, x_max, t_min, t_max, Nx, Nt, device)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T, X, U, cmap=_CMAP_SOL, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="u(x, t)")

    ax.set_xlabel("t", labelpad=8)
    ax.set_ylabel("x", labelpad=8)
    ax.set_zlabel("u", labelpad=8)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    _save(fig, savefig)
    return fig

def animate_surface_3d(
    model,
    x_min: float, x_max: float,
    t_min: float, t_max: float,
    Nx: int = 100, Nt: int = 100,
    title: str = "PINN — Superfície 3D Rotativa",
    device: str = "cpu",
    savefig: Optional[str] = None,
) -> animation.FuncAnimation:
    """Anima a superfície 3D rotacionando a câmera."""
    X, T, U = _grid_xt(model, x_min, x_max, t_min, t_max, Nx, Nt, device)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T, X, U, cmap=_CMAP_SOL, linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="u(x, t)")

    ax.set_xlabel("t", labelpad=8)
    ax.set_ylabel("x", labelpad=8)
    ax.set_zlabel("u", labelpad=8)
    ax.set_title(title)

    def init():
        return fig,

    def animate(i):
        # Rotaciona de azim=-60 até azim=300 (360 graus)
        ax.view_init(elev=28.0, azim=-60 + i * 4)
        return fig,

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=90, interval=50, blit=False
    )

    if savefig:
        if savefig.endswith(".gif"):
            anim.save(savefig, writer="pillow", fps=20)
        else:
            anim.save(savefig, writer="ffmpeg", fps=20)
        print(f"  → animação 3D salva em {savefig}")

    return anim
# ─────────────────────────────────────────────────────────────────────────────
# 5. ANIMAÇÃO DA EVOLUÇÃO TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────

def animate_solution(
    model,
    x_min: float, x_max: float,
    t_min: float, t_max: float,
    N_frames: int = 60,
    N_x: int = 400,
    exact_fn: Optional[Callable] = None,
    title: str = "PINN — Evolução u(x, t)",
    interval_ms: int = 50,
    device: str = "cpu",
    savefig: Optional[str] = None,
) -> animation.FuncAnimation:
    """
    Animação da solução u(x, t) ao longo do tempo.

    Parâmetros
    ----------
    exact_fn    : callable (x_np, t_val) → u_np  (solução exata, opcional)
    interval_ms : delay entre frames em ms
    savefig     : se terminar em '.gif', salva em GIF; se '.mp4', salva vídeo
    """
    x_np = np.linspace(x_min, x_max, N_x)
    t_arr = np.linspace(t_min, t_max, N_frames)
    dev = torch.device(device)
    x_t = torch.tensor(x_np, dtype=torch.float32).reshape(-1, 1).to(dev)

    # Pré-computa todos os frames
    U_frames = []
    U_ex_frames = []
    with torch.no_grad():
        for tv in t_arr:
            t_t = torch.full_like(x_t, tv)
            U_frames.append(model(x_t, t_t).cpu().numpy().ravel())
            if exact_fn is not None:
                U_ex_frames.append(exact_fn(x_np, tv))

    all_u = np.concatenate(U_frames)
    u_min, u_max = all_u.min(), all_u.max()
    margin = 0.1 * (u_max - u_min + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 4))
    line_pinn, = ax.plot([], [], "tab:red", lw=2, label="PINN")
    line_ex = None
    if exact_fn is not None:
        line_ex, = ax.plot([], [], "k--", lw=1.5, label="Exata")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(u_min - margin, u_max + margin)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    time_text = ax.text(0.02, 0.94, "", transform=ax.transAxes, fontsize=10)
    ax.set_title(title)

    def init():
        line_pinn.set_data([], [])
        time_text.set_text("")
        return [line_pinn, time_text]

    def update(i):
        line_pinn.set_data(x_np, U_frames[i])
        time_text.set_text(f"t = {t_arr[i]:.3f}")
        artists = [line_pinn, time_text]
        if line_ex is not None:
            line_ex.set_data(x_np, U_ex_frames[i])
            artists.append(line_ex)
        return artists

    anim = animation.FuncAnimation(
        fig, update, frames=N_frames, init_func=init,
        interval=interval_ms, blit=True
    )

    if savefig:
        if savefig.endswith(".gif"):
            anim.save(savefig, writer="pillow", fps=1000 // interval_ms)
        else:
            anim.save(savefig, writer="ffmpeg", fps=1000 // interval_ms)
        print(f"  → animação salva em {savefig}")

    return anim


# ─────────────────────────────────────────────────────────────────────────────
# 6. RESÍDUO NO ESPAÇO
# ─────────────────────────────────────────────────────────────────────────────

def plot_residual(
    analyzer,
    t_vals: List[float],
    N: int = 512,
    title: str = "Resíduo da EDP",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Plota r(x, t_val) para cada instante.

    Parâmetros
    ----------
    analyzer : objeto com método residual_grid(t_val, N)
    """
    n = len(t_vals)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    cfg = analyzer.cfg
    x_np = np.linspace(cfg.x_min, cfg.x_max, N)

    for ax, tv in zip(axes, t_vals):
        r = analyzer.residual_grid(tv, N)
        ax.plot(x_np, r, "tab:blue", lw=1.5)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"t = {tv:.2f}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("r(x)")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    _save(fig, savefig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. ESPECTRO DE FOURIER DO RESÍDUO
# ─────────────────────────────────────────────────────────────────────────────

def plot_fourier(
    analyzer,
    t_val: float,
    N: int = 512,
    log_scale: bool = True,
    savefig: Optional[str] = None,
) -> plt.Figure:
    """Espectro |R(k)| do resíduo em t_val."""
    freqs, amps = analyzer.fft_residual(t_val, N)
    diag = analyzer.diagnose(t_val, N)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_fn = ax.semilogy if log_scale else ax.plot
    plot_fn(freqs[1:], amps[1:] + 1e-16, "tab:red", lw=1.5)
    ax.set_xlabel("Frequência espacial k")
    ax.set_ylabel("|R(k)|")
    ax.set_title(f"Espectro de Fourier do Resíduo  (t={t_val:.2f})")
    ax.grid(True, which="both", alpha=0.25)
    fig.text(0.5, -0.04, diag, ha="center", fontsize=9, color="navy")
    plt.tight_layout()
    _save(fig, savefig)
    print(diag)
    return fig


def plot_fourier_panel(
    analyzer,
    t_val: float,
    N: int = 512,
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Painel lado a lado: resíduo no espaço + espectro de Fourier.
    Ideal para o 'Diagnóstico Físico do Resíduo' do MS901.
    """
    cfg = analyzer.cfg
    x_np = np.linspace(cfg.x_min, cfg.x_max, N)
    r = analyzer.residual_grid(t_val, N)
    freqs, amps = analyzer.fft_residual(t_val, N)
    diag = analyzer.diagnose(t_val, N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # ── esquerda: resíduo ──
    ax1.plot(x_np, r, "tab:blue", lw=1.5)
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.set_title(f"Resíduo r(x)  [t = {t_val:.2f}]")
    ax1.set_xlabel("x")
    ax1.set_ylabel("r(x) = EDP[u_PINN]")
    ax1.grid(True, alpha=0.25)

    # ── direita: espectro ──
    ax2.semilogy(freqs[1:], amps[1:] + 1e-16, "tab:red", lw=1.5)
    ax2.set_title("Espectro |R(k)|  (Fourier do Resíduo)")
    ax2.set_xlabel("Frequência espacial k")
    ax2.set_ylabel("|R(k)|")
    ax2.grid(True, which="both", alpha=0.25)

    fig.suptitle(diag, fontsize=10, color="navy", y=1.03)
    plt.tight_layout()
    _save(fig, savefig)
    print(diag)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. MONITOR DE ENERGIA (GRONWALL)
# ─────────────────────────────────────────────────────────────────────────────

def plot_energy(
    analyzer,
    norm_a_C1: float,
    N_times: int = 30,
    N_grid: int = 256,
    title: str = "Monitor de Energia — Cota de Gronwall",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Evolução de ||u||²_L2 vs cota de Gronwall.

    Se a curva PINN ultrapassar a cota, a rede injeta energia artificial.
    """
    t_vals, energias, cotas = analyzer.scan_energy(norm_a_C1, N_times, N_grid)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_vals, energias, "tab:blue", lw=2, marker="o", ms=4, label=r"$\|u\|^2_{L^2}$ (PINN)")
    ax.plot(t_vals, cotas, "tab:red", lw=2, ls="--", label=r"Gronwall: $e^{\|a\|_{C^1}t}E(0)$")
    ax.fill_between(t_vals, energias, cotas, alpha=0.12, color="red", label="Margem")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\int u^2\,dx$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, savefig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. ERRO PONTUAL (PINN vs EXATA)
# ─────────────────────────────────────────────────────────────────────────────

def plot_error(
    model,
    exact_fn: Callable,
    x_min: float, x_max: float,
    t_vals: List[float],
    N: int = 512,
    device: str = "cpu",
    title: str = "Erro Pontual  |u_PINN − u_exata|",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Plota |u_PINN(x, t) − u_exata(x, t)| para cada instante.

    Parâmetros
    ----------
    exact_fn : callable (x_np, t_val) → u_np
    """
    n = len(t_vals)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    dev = torch.device(device)
    x_np = np.linspace(x_min, x_max, N)
    x_t = torch.tensor(x_np, dtype=torch.float32).reshape(-1, 1).to(dev)

    for ax, tv in zip(axes, t_vals):
        t_t = torch.full_like(x_t, tv)
        with torch.no_grad():
            u_pinn = model(x_t, t_t).cpu().numpy().ravel()
        u_ex = exact_fn(x_np, tv)
        err = np.abs(u_pinn - u_ex)

        ax.plot(x_np, err, "tab:orange", lw=1.8)
        ax.fill_between(x_np, 0, err, alpha=0.25, color="tab:orange")
        ax.set_title(f"t = {tv:.2f}  |  L∞={err.max():.2e}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("|erro|")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    _save(fig, savefig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. DASHBOARD COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def dashboard(
    model,
    analyzer,
    history: Dict[str, List[float]],
    t_fourier: float = 0.5,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    exact_fn: Optional[Callable] = None,
    device: str = "cpu",
    title: str = "Dashboard PINN",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Painel completo 2×2:
      [0,0] Heatmap u(x,t)   |  [0,1] Loss
      [1,0] Resíduo no espaço |  [1,1] Espectro de Fourier

    Parâmetros
    ----------
    analyzer : objeto Analyzer de pinn_core
    history  : dict de histórico de loss do Trainer
    """
    cfg = analyzer.cfg
    xmin = x_min or cfg.x_min
    xmax = x_max or cfg.x_max
    tmin = t_min or cfg.t_min
    tmax = t_max or cfg.t_max

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ── [0,0] heatmap ──
    ax = axes[0, 0]
    X, T, U = _grid_xt(model, xmin, xmax, tmin, tmax, device=device)
    c = ax.contourf(T, X, U, levels=100, cmap=_CMAP_SOL)
    fig.colorbar(c, ax=ax, label="u", shrink=0.9)
    ax.set_xlabel("t"); ax.set_ylabel("x")
    ax.set_title("Solução u(x, t)")
    if exact_fn is not None:
        n_lines = 5
        for tv in np.linspace(tmin, tmax, n_lines):
            x_np = np.linspace(xmin, xmax, 200)
            u_ex = exact_fn(x_np, tv)
            ax.plot(np.full_like(x_np, tv), x_np, "w--", lw=0.8, alpha=0.5)

    # ── [0,1] loss ──
    ax = axes[0, 1]
    eps = range(len(history["loss"]))
    ax.semilogy(eps, history["loss"],      lw=2,   label="Total")
    ax.semilogy(eps, history["loss_data"], lw=1.5, ls="--", label="Dados")
    ax.semilogy(eps, history["loss_phys"], lw=1.5, ls=":",  label="Física")
    ax.set_xlabel("Época"); ax.set_ylabel("Loss")
    ax.set_title("Histórico de Loss"); ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)

    # ── [1,0] resíduo no espaço ──
    ax = axes[1, 0]
    x_np = np.linspace(xmin, xmax, 512)
    r = analyzer.residual_grid(t_fourier, 512)
    ax.plot(x_np, r, "tab:blue", lw=1.5)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("x"); ax.set_ylabel("r(x)")
    ax.set_title(f"Resíduo r(x, t={t_fourier:.2f})")
    ax.grid(True, alpha=0.2)

    # ── [1,1] Fourier ──
    ax = axes[1, 1]
    freqs, amps = analyzer.fft_residual(t_fourier, 512)
    ax.semilogy(freqs[1:], amps[1:] + 1e-16, "tab:red", lw=1.5)
    ax.set_xlabel("k"); ax.set_ylabel("|R(k)|")
    ax.set_title("Espectro de Fourier do Resíduo")
    ax.grid(True, which="both", alpha=0.2)
    diag = analyzer.diagnose(t_fourier)
    ax.text(0.02, 0.04, diag.split("\n")[0], transform=ax.transAxes,
            fontsize=7.5, color="navy")

    plt.tight_layout()
    _save(fig, savefig)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 11. ANÁLISE ESPECTRAL PROFUNDA (DIAGNÓSTICO FÍSICO)
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectral_analysis(
    analyzer,
    t_val: float = 0.5,
    title: str = "Diagnóstico Físico do Resíduo (Fourier)",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Plota a análise espectral profunda mostrando Difusão e Dispersão Numérica.
    """
    res = analyzer.deep_spectral_analysis(t_val, N=512)
    k_v = res['k_valid']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{title} - t={t_val}", fontsize=14, fontweight="bold")

    # Gráfico 1: Parte Real (Difusão) vs k^2
    ax1 = axes[0]
    ax1.plot(k_v**2, res['H_real'], 'o', color='tab:blue', alpha=0.5, label='Dados Re(H)')
    # Linha de ajuste (Difusão)
    k2_line = np.linspace(0, np.max(k_v**2), 50)
    ax1.plot(k2_line, -res['nu_num'] * k2_line, 'r-', lw=2, label=f'Ajuste (ν={res["nu_num"]:.3e})')
    ax1.set_xlabel(r"$k^2$")
    ax1.set_ylabel(r"$Re[H(k)]$")
    ax1.set_title("Diagnóstico de Difusão Numérica")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Gráfico 2: Parte Imaginária (Dispersão) vs k^3
    ax2 = axes[1]
    ax2.plot(k_v**3, res['H_imag'], 'o', color='tab:orange', alpha=0.5, label='Dados Im(H)')
    # Linha de ajuste (Dispersão)
    k3_line = np.linspace(0, np.max(k_v**3), 50)
    ax2.plot(k3_line, -res['mu_num'] * k3_line, 'g-', lw=2, label=f'Ajuste (μ={res["mu_num"]:.3e})')
    ax2.set_xlabel(r"$k^3$")
    ax2.set_ylabel(r"$Im[H(k)]$")
    ax2.set_title("Diagnóstico de Dispersão Numérica")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    _save(fig, savefig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 12. MAPA DE CALOR DO ERRO PINN vs EXATA
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_heatmap(
    model,
    exact_fn: Callable,
    x_min: float, x_max: float,
    t_min: float, t_max: float,
    Nx: int = 256,
    Nt: int = 200,
    title: str = "Comparação PINN vs Exata",
    device: str = "cpu",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Painel de 3 mapas de calor no domínio (x, t):
      esquerda  — u_PINN(x, t)
      centro    — u_exata(x, t)
      direita   — |u_PINN − u_exata|(x, t)  em escala logarítmica

    Parâmetros
    ----------
    exact_fn : callable (x_np, t_val) → u_np
    """
    # grade espaciotemporal
    xv = np.linspace(x_min, x_max, Nx)
    tv = np.linspace(t_min, t_max, Nt)
    X_mg, T_mg = np.meshgrid(xv, tv, indexing="ij")  # shape (Nx, Nt)

    # predição PINN em toda a grade
    dev = torch.device(device)
    x_flat = torch.tensor(X_mg.ravel(), dtype=torch.float32).unsqueeze(1).to(dev)
    t_flat = torch.tensor(T_mg.ravel(), dtype=torch.float32).unsqueeze(1).to(dev)
    with torch.no_grad():
        U_pinn = model(x_flat, t_flat).cpu().numpy().reshape(Nx, Nt)

    # solução exata coluna a coluna (uma slice por instante)
    U_exact = np.stack([exact_fn(xv, t) for t in tv], axis=1)  # (Nx, Nt)

    err_abs = np.abs(U_pinn - U_exact)

    # limites de cor simétricos para soluções
    vmax_sol = max(np.abs(U_pinn).max(), np.abs(U_exact).max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    extent = [t_min, t_max, x_min, x_max]

    def _imshow(ax, data, cmap, label, vmin=None, vmax=None, log=False):
        if log:
            data_plot = np.log10(np.maximum(data, 1e-16))
            im = ax.imshow(data_plot, origin="lower", aspect="auto",
                           extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("log₁₀|erro|")
        else:
            im = ax.imshow(data, origin="lower", aspect="auto",
                           extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(label)

    _imshow(axes[0], U_pinn,  _CMAP_SOL, "u — PINN",       vmin=-vmax_sol, vmax=vmax_sol)
    _imshow(axes[1], U_exact, _CMAP_SOL, "u — Exata",      vmin=-vmax_sol, vmax=vmax_sol)
    _imshow(axes[2], err_abs, "inferno",  "|erro| (log₁₀)", log=True)

    plt.tight_layout()
    _save(fig, savefig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 13. CURVAS DE ERRO L²(t) E L∞(t)
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_curve(
    metrics: Dict,
    title: str = "Evolução do Erro PINN vs Exata",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Plota L²_rel(t) e L∞(t) usando o dicionário retornado por
    ``Analyzer.compute_error_metrics()``.

    Parâmetros
    ----------
    metrics : dict com chaves 't_vals', 'l2_rel', 'linf'
    """
    t     = np.asarray(metrics["t_vals"])
    l2    = np.asarray(metrics["l2_rel"])
    linf  = np.asarray(metrics["linf"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax_l2, ax_linf = axes

    ax_l2.semilogy(t, l2, lw=2, color="tab:blue")
    ax_l2.set_xlabel("t")
    ax_l2.set_ylabel(r"$\|u_{PINN} - u_{ex}\|_{L^2} \,/\, \|u_{ex}\|_{L^2}$")
    ax_l2.set_title(f"Erro L² relativo  (máx = {l2.max():.2e})")
    ax_l2.grid(True, which="both", alpha=0.25)
    ax_l2.axhline(l2.max(), color="tab:blue", ls=":", alpha=0.5)

    ax_linf.semilogy(t, linf, lw=2, color="tab:orange")
    ax_linf.set_xlabel("t")
    ax_linf.set_ylabel(r"$\|u_{PINN} - u_{ex}\|_{L^\infty}$")
    ax_linf.set_title(f"Erro L∞ pontual  (máx = {linf.max():.2e})")
    ax_linf.grid(True, which="both", alpha=0.25)
    ax_linf.axhline(linf.max(), color="tab:orange", ls=":", alpha=0.5)

    plt.tight_layout()
    _save(fig, savefig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 14. PAINEL DE COMPARAÇÃO 2×2  (PINN vs EXATA)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_panel(
    model,
    exact_fn: Callable,
    analyzer,
    metrics: Dict,
    x_min: float, x_max: float,
    t_min: float, t_max: float,
    t_slices: Optional[List[float]] = None,
    title: str = "Painel de Comparação PINN vs Exata",
    device: str = "cpu",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Painel 2×2 reunindo os principais diagnósticos de comparação:

      [0,0] — fatias u(x, t_i): PINN (sólido) vs Exata (tracejado)
      [0,1] — mapa de calor do erro absoluto |u_PINN − u_exata| (log)
      [1,0] — curvas L²_rel(t) e L∞(t) vs tempo
      [1,1] — espectro de Fourier do resíduo PINN no instante central

    Parâmetros
    ----------
    metrics   : dict retornado por ``Analyzer.compute_error_metrics()``
    t_slices  : instantes usados no subplot de fatias; padrão [0, 0.5, 1.0]
    """
    if t_slices is None:
        t_slices = [0.0, 0.5, 1.0]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)
    ax_slices  = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[0, 1])
    ax_err     = fig.add_subplot(gs[1, 0])
    ax_fourier = fig.add_subplot(gs[1, 1])

    dev  = torch.device(device)
    x_np = np.linspace(x_min, x_max, 512)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(t_slices)))

    # ── [0,0] Fatias de solução ──────────────────────────────────────────────
    for color, t_val in zip(colors, t_slices):
        x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1).to(dev)
        t_t = torch.full_like(x_t, t_val)
        with torch.no_grad():
            u_pinn = model(x_t, t_t).cpu().numpy().ravel()
        u_ex = exact_fn(x_np, t_val)
        ax_slices.plot(x_np, u_pinn, color=color, lw=2,   label=f"PINN  t={t_val:.1f}")
        ax_slices.plot(x_np, u_ex,   color=color, lw=1.5, ls="--", label=f"Exata t={t_val:.1f}")

    ax_slices.set_xlabel("x")
    ax_slices.set_ylabel("u(x, t)")
    ax_slices.set_title("Fatias: PINN (sólido) vs Exata (---)")
    ax_slices.legend(fontsize=7, ncol=2)
    ax_slices.grid(True, alpha=0.2)

    # ── [0,1] Mapa de calor do erro absoluto ─────────────────────────────────
    Nx, Nt = 200, 150
    xv = np.linspace(x_min, x_max, Nx)
    tv = np.linspace(t_min, t_max, Nt)
    X_mg, T_mg = np.meshgrid(xv, tv, indexing="ij")
    x_flat = torch.tensor(X_mg.ravel(), dtype=torch.float32).unsqueeze(1).to(dev)
    t_flat = torch.tensor(T_mg.ravel(), dtype=torch.float32).unsqueeze(1).to(dev)
    with torch.no_grad():
        U_pinn_grid = model(x_flat, t_flat).cpu().numpy().reshape(Nx, Nt)
    U_exact_grid = np.stack([exact_fn(xv, t) for t in tv], axis=1)
    err_log = np.log10(np.maximum(np.abs(U_pinn_grid - U_exact_grid), 1e-16))

    extent = [t_min, t_max, x_min, x_max]
    im = ax_heatmap.imshow(err_log, origin="lower", aspect="auto",
                           extent=extent, cmap="inferno")
    fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04).set_label("log₁₀|erro|")
    ax_heatmap.set_xlabel("t")
    ax_heatmap.set_ylabel("x")
    ax_heatmap.set_title("|u_PINN − u_exata|  (escala log₁₀)")

    # ── [1,0] Curvas de erro L² e L∞ ─────────────────────────────────────────
    t_arr  = np.asarray(metrics["t_vals"])
    l2_arr = np.asarray(metrics["l2_rel"])
    li_arr = np.asarray(metrics["linf"])

    ax_err.semilogy(t_arr, l2_arr, lw=2, color="tab:blue",   label=r"$L^2$ rel")
    ax_err.semilogy(t_arr, li_arr, lw=2, color="tab:orange", label=r"$L^\infty$")
    ax_err.set_xlabel("t")
    ax_err.set_ylabel("Erro")
    ax_err.set_title("Evolução do erro vs tempo")
    ax_err.legend()
    ax_err.grid(True, which="both", alpha=0.25)

    # ── [1,1] Espectro de Fourier do resíduo ──────────────────────────────────
    t_mid  = 0.5 * (t_min + t_max)
    freqs, spec = analyzer.fft_residual(t_mid, N=len(x_np))

    ax_fourier.semilogy(freqs[1:], spec[1:], lw=1.5, color="tab:purple")
    ax_fourier.set_xlabel("Frequência k")
    ax_fourier.set_ylabel("|FFT(resíduo)|")
    ax_fourier.set_title(f"Espectro do resíduo PINN  (t = {t_mid:.2f})")
    ax_fourier.grid(True, which="both", alpha=0.25)

    _save(fig, savefig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 15. TRIO DE SUPERFÍCIES 3D  (PINN | EXATA | |ERRO|)
# ─────────────────────────────────────────────────────────────────────────────

def plot_surface_3d_trio(
    model,
    exact_fn: Callable,
    x_min: float, x_max: float,
    t_min: float, t_max: float,
    Nx: int = 120,
    Nt: int = 120,
    title: str = "Superfícies 3D — PINN | Exata | |Erro|",
    elev: float = 28.0,
    azim: float = -55.0,
    device: str = "cpu",
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Três superfícies 3D lado a lado:
      esquerda — u_PINN(x, t)
      centro   — u_exata(x, t)
      direita  — |u_PINN − u_exata|(x, t)  em escala logarítmica

    Parâmetros
    ----------
    exact_fn : callable (x_np, t_val) → u_np
    elev, azim : ângulo de visão inicial (graus)
    """
    # ── grade e predições ─────────────────────────────────────────────────────
    xv = np.linspace(x_min, x_max, Nx)
    tv = np.linspace(t_min, t_max, Nt)
    X_mg, T_mg = np.meshgrid(xv, tv, indexing="ij")   # (Nx, Nt)

    dev = torch.device(device)
    x_flat = torch.tensor(X_mg.ravel(), dtype=torch.float32).unsqueeze(1).to(dev)
    t_flat = torch.tensor(T_mg.ravel(), dtype=torch.float32).unsqueeze(1).to(dev)
    with torch.no_grad():
        U_pinn = model(x_flat, t_flat).cpu().numpy().reshape(Nx, Nt)

    U_exact = np.stack([exact_fn(xv, t) for t in tv], axis=1)   # (Nx, Nt)
    E_abs   = np.abs(U_pinn - U_exact)
    E_log   = np.log10(np.maximum(E_abs, 1e-16))

    # grade meshgrid para plot_surface (eixos: T horizontal, X profundidade)
    T_plot = T_mg   # (Nx, Nt)
    X_plot = X_mg   # (Nx, Nt)

    # limites de cor simétricos para soluções
    vmax = max(np.abs(U_pinn).max(), np.abs(U_exact).max())

    # ── figura com 3 eixos 3D ────────────────────────────────────────────────
    fig = plt.figure(figsize=(21, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    specs = [
        (U_pinn, _CMAP_SOL, "u — PINN",         -vmax, vmax,  False),
        (U_exact, _CMAP_SOL, "u — Exata",        -vmax, vmax,  False),
        (E_log,  "inferno",  "|erro| (log₁₀)",   None,  None,  True),
    ]

    for idx, (Z, cmap, label, zmin, zmax, is_log) in enumerate(specs, 1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")

        surf = ax.plot_surface(
            T_plot, X_plot, Z,
            cmap=cmap,
            vmin=zmin, vmax=zmax,
            linewidth=0,
            antialiased=True,
            alpha=0.92,
        )
        cbar = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08)
        if is_log:
            cbar.set_label("log₁₀|u_PINN − u_exata|", fontsize=8)

        ax.set_xlabel("t", labelpad=6)
        ax.set_ylabel("x", labelpad=6)
        ax.set_zlabel("u" if not is_log else "log₁₀|e|", labelpad=6)
        ax.set_title(label, fontsize=11)
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    _save(fig, savefig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Superfícies 3D Interativas (Plotly)
# ─────────────────────────────────────────────────────────────────────────────

def plot_surface_3d_interactive(
    model,
    x_min: float, x_max: float,
    t_min: float, t_max: float,
    exact_fn: Optional[Callable] = None,
    title: str = "Superfície 3D Interativa",
    Nx: int = 120,
    Nt: int = 120,
    device: str = "cpu",
    savefig: Optional[str] = None,
) -> None:
    """Gera superfície(s) 3D interativas via Plotly e salva como HTML autocontido.

    Se ``exact_fn`` for fornecida, gera três subplots lado a lado:
    PINN | Exata | log₁₀|erro|. Caso contrário, apenas a superfície da PINN.

    O arquivo HTML é autocontido (inclui plotly.js) e pode ser aberto
    diretamente no browser ou embutido via <iframe>.

    Parâmetros
    ----------
    savefig : caminho do arquivo .html de saída (obrigatório na prática).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    xv = np.linspace(x_min, x_max, Nx)
    tv = np.linspace(t_min, t_max, Nt)
    X_mg, T_mg = np.meshgrid(xv, tv, indexing="ij")   # (Nx, Nt)

    dev = torch.device(device)
    x_flat = torch.tensor(X_mg.ravel(), dtype=torch.float32).unsqueeze(1).to(dev)
    t_flat = torch.tensor(T_mg.ravel(), dtype=torch.float32).unsqueeze(1).to(dev)
    with torch.no_grad():
        U_pinn = model(x_flat, t_flat).cpu().numpy().reshape(Nx, Nt)

    # eixos: plotly usa (y=linhas, x=colunas) → passamos tv como "x" e xv como "y"
    # para que t fique no eixo horizontal e x no vertical (convencional em PDE).
    common_surface_kw = dict(x=tv, y=xv, colorscale="RdBu_r")

    if exact_fn is None:
        fig = go.Figure()
        vmax = float(np.abs(U_pinn).max()) or 1.0
        fig.add_trace(go.Surface(
            **common_surface_kw,
            z=U_pinn,
            cmin=-vmax, cmax=vmax,
            name="PINN",
            colorbar=dict(title="u"),
        ))
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="t",
                yaxis_title="x",
                zaxis_title="u",
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=600,
        )
    else:
        U_exact = np.stack([exact_fn(xv, t) for t in tv], axis=1)   # (Nx, Nt)
        E_log   = np.log10(np.maximum(np.abs(U_pinn - U_exact), 1e-16))
        vmax    = float(max(np.abs(U_pinn).max(), np.abs(U_exact).max())) or 1.0

        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
            subplot_titles=["PINN", "Solução Exata", "log₁₀|erro|"],
            horizontal_spacing=0.02,
        )

        fig.add_trace(go.Surface(
            **common_surface_kw, z=U_pinn,
            cmin=-vmax, cmax=vmax,
            colorbar=dict(title="u", x=0.30, len=0.8),
        ), row=1, col=1)

        fig.add_trace(go.Surface(
            **common_surface_kw, z=U_exact,
            cmin=-vmax, cmax=vmax,
            colorbar=dict(title="u", x=0.63, len=0.8),
        ), row=1, col=2)

        fig.add_trace(go.Surface(
            x=tv, y=xv, z=E_log,
            colorscale="inferno",
            colorbar=dict(title="log₁₀|e|", x=0.99, len=0.8),
            name="Erro",
        ), row=1, col=3)

        scene_kw = dict(xaxis_title="t", yaxis_title="x")
        fig.update_layout(
            title=title,
            scene=dict(**scene_kw, zaxis_title="u"),
            scene2=dict(**scene_kw, zaxis_title="u"),
            scene3=dict(**scene_kw, zaxis_title="log₁₀|e|"),
            margin=dict(l=0, r=0, t=60, b=0),
            height=600,
        )

    if savefig:
        fig.write_html(savefig, include_plotlyjs=True, full_html=True)
