"""
pinn_viz.py
===========
Biblioteca de visualização para PINNs — MS901/MT861.

Funções:
  plot_loss()           — histórico de loss (escala log)
  plot_slices()         — u(x, t_i) em vários instantes
  plot_heatmap()        — mapa de calor u(x, t) 2D
  plot_surface_3d()     — superfície 3D u(x, t)
  animate_solution()    — animação da evolução u(x, t)
  plot_residual()       — resíduo no espaço
  plot_fourier()        — espectro de Fourier do resíduo
  plot_fourier_panel()  — resíduo + FFT lado a lado
  plot_energy()         — norma L² vs cota de Gronwall
  plot_error()          — erro pontual PINN vs solução exata
  dashboard()           — painel completo (solução + loss + Fourier)
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
