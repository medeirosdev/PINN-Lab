"""
pinn_burgers_lib.py
===================
Biblioteca PINN para a Equação de Burgers (viscosa e invíscida).

Equação viscosa:  ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
Equação invíscida: ∂u/∂t + ∂(u²/2)/∂x = 0

Uso básico:
    from pinn_burgers_lib import BurgersConfig, BurgersDataset, BurgersTrainer, BurgersAnalyzer

Referência metodológica: MS901/MT861 — "Diagnóstico Físico do Resíduo via Equação Modificada"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utilitário: device padrão
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÃO
# ---------------------------------------------------------------------------

@dataclass
class BurgersConfig:
    """Todos os hiperparâmetros do experimento."""

    # Física
    nu: float = 0.01 / math.pi   # viscosidade (0 = invíscida)
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # Condição inicial padrão: -sin(π x)
    u0: Callable[[torch.Tensor], torch.Tensor] = field(
        default=None, repr=False
    )
    # Condições de contorno (valor em x_min e x_max); None = periódicas
    bc_left: Optional[float] = 0.0
    bc_right: Optional[float] = 0.0

    # Pontos de amostragem
    N_ic: int = 200    # condição inicial
    N_bc: int = 100    # contorno (cada lado)
    N_f: int = 10_000  # colocação (física)

    # Arquitetura MLP
    n_layers: int = 4
    n_neurons: int = 40
    activation: str = "tanh"   # "tanh" | "sin" | "relu"

    # Treino
    lr_adam: float = 1e-3
    epochs_adam: int = 10_000
    epochs_lbfgs: int = 500
    lambda_phys: float = 1.0   # peso do termo de física na loss

    # Reprodutibilidade
    seed: int = 42

    def __post_init__(self):
        if self.u0 is None:
            self.u0 = lambda x: -torch.sin(math.pi * x)


# ---------------------------------------------------------------------------
# 2. REDE NEURAL (MLP)
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """
    Multi-Layer Perceptron para aproximar u(x, t).
    Entrada: (x, t) ∈ ℝ²  →  Saída: u(x, t) ∈ ℝ
    """

    _activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sin": lambda: _Sin(),
    }

    def __init__(self, cfg: BurgersConfig):
        super().__init__()
        act_cls = self._activations.get(cfg.activation, nn.Tanh)
        layers: List[nn.Module] = [nn.Linear(2, cfg.n_neurons), act_cls()]
        for _ in range(cfg.n_layers - 1):
            layers += [nn.Linear(cfg.n_neurons, cfg.n_neurons), act_cls()]
        layers.append(nn.Linear(cfg.n_neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))


class _Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


# ---------------------------------------------------------------------------
# 3. RESÍDUO DA EDP (autodiferenciação)
# ---------------------------------------------------------------------------

class BurgersResidual:
    """
    Calcula o resíduo da equação de Burgers via autograd.

    Viscosa:   r = u_t + u·u_x − ν·u_xx
    Invíscida: r = u_t + u·u_x          (nu=0)
    """

    def __init__(self, cfg: BurgersConfig):
        self.nu = cfg.nu

    def __call__(
        self, model: PINN, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        u = model(x, t)

        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        residual = u_t + u * u_x

        if self.nu != 0.0:
            u_xx = torch.autograd.grad(
                u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
            )[0]
            residual = residual - self.nu * u_xx

        return residual


# ---------------------------------------------------------------------------
# 4. DATASET
# ---------------------------------------------------------------------------

class BurgersDataset:
    """
    Gera pontos de CI, CC e colocação (física).
    Todos os tensores ficam no DEVICE configurado.
    """

    def __init__(self, cfg: BurgersConfig):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.cfg = cfg
        self._build()

    def _build(self):
        cfg = self.cfg

        # --- Condição inicial u(x, 0) = u0(x) ---
        x_ic = torch.linspace(cfg.x_min, cfg.x_max, cfg.N_ic).reshape(-1, 1)
        t_ic = torch.zeros(cfg.N_ic, 1)
        u_ic = cfg.u0(x_ic)

        data_x = [x_ic]
        data_t = [t_ic]
        data_u = [u_ic]

        # --- Condições de contorno ---
        t_bc = torch.linspace(cfg.t_min, cfg.t_max, cfg.N_bc).reshape(-1, 1)
        if cfg.bc_left is not None:
            data_x.append(torch.full((cfg.N_bc, 1), cfg.x_min))
            data_t.append(t_bc)
            data_u.append(torch.full((cfg.N_bc, 1), cfg.bc_left))
        if cfg.bc_right is not None:
            data_x.append(torch.full((cfg.N_bc, 1), cfg.x_max))
            data_t.append(t_bc)
            data_u.append(torch.full((cfg.N_bc, 1), cfg.bc_right))

        self.x_data = torch.cat(data_x).to(DEVICE)
        self.t_data = torch.cat(data_t).to(DEVICE)
        self.u_data = torch.cat(data_u).to(DEVICE)

        # --- Pontos de colocação (aleatórios no domínio) ---
        lx = cfg.x_max - cfg.x_min
        lt = cfg.t_max - cfg.t_min
        self.x_f = (torch.rand(cfg.N_f, 1) * lx + cfg.x_min).to(DEVICE)
        self.t_f = (torch.rand(cfg.N_f, 1) * lt + cfg.t_min).to(DEVICE)


# ---------------------------------------------------------------------------
# 5. TRAINER
# ---------------------------------------------------------------------------

class BurgersTrainer:
    """
    Loop de treino: Adam (fase 1) + L-BFGS opcional (fase 2).
    Histórico completo de loss armazenado em self.history.
    """

    def __init__(self, cfg: BurgersConfig, model: PINN, dataset: BurgersDataset):
        self.cfg = cfg
        self.model = model.to(DEVICE)
        self.ds = dataset
        self.residual_fn = BurgersResidual(cfg)
        self.history = {"loss": [], "loss_data": [], "loss_phys": []}

    def _compute_losses(self):
        # Loss dados
        u_pred = self.model(self.ds.x_data, self.ds.t_data)
        loss_data = torch.mean((u_pred - self.ds.u_data) ** 2)
        # Loss física
        res = self.residual_fn(self.model, self.ds.x_f, self.ds.t_f)
        loss_phys = torch.mean(res ** 2)
        loss = loss_data + self.cfg.lambda_phys * loss_phys
        return loss, loss_data, loss_phys

    def train_adam(self, verbose: bool = True):
        cfg = self.cfg
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr_adam)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=cfg.epochs_adam // 2, gamma=0.1
        )

        if verbose:
            print(f"{'Época':<8} {'Total':<14} {'Dados':<14} {'Física':<14}")
            print("-" * 52)

        for epoch in range(cfg.epochs_adam):
            opt.zero_grad()
            loss, ld, lp = self._compute_losses()
            loss.backward()
            opt.step()
            scheduler.step()

            self.history["loss"].append(loss.item())
            self.history["loss_data"].append(ld.item())
            self.history["loss_phys"].append(lp.item())

            if verbose and epoch % (cfg.epochs_adam // 10) == 0:
                print(f"{epoch:<8} {loss.item():<14.6f} {ld.item():<14.6f} {lp.item():<14.6f}")

        if verbose:
            print(f"\n[Adam] Loss final: {self.history['loss'][-1]:.6e}")

    def train_lbfgs(self, verbose: bool = True):
        """Refinamento com L-BFGS após Adam."""
        opt = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=self.cfg.epochs_lbfgs,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            loss, ld, lp = self._compute_losses()
            loss.backward()
            self.history["loss"].append(loss.item())
            self.history["loss_data"].append(ld.item())
            self.history["loss_phys"].append(lp.item())
            return loss

        opt.step(closure)
        if verbose:
            print(f"[L-BFGS] Loss final: {self.history['loss'][-1]:.6e}")

    def train(self, verbose: bool = True):
        """Treino completo: Adam + L-BFGS."""
        self.train_adam(verbose=verbose)
        if self.cfg.epochs_lbfgs > 0:
            self.train_lbfgs(verbose=verbose)


# ---------------------------------------------------------------------------
# 6. ANALYZER — Fourier do resíduo
# ---------------------------------------------------------------------------

class BurgersAnalyzer:
    """
    Análise de Fourier do resíduo espacial r(x, t_eval).

    Diagnóstico físico (Teoria da Equação Modificada):
      - Espectro concentrado em baixas freq  → difusão numérica (amortecimento)
      - Picos em frequências intermediárias  → dispersão numérica (oscilações)
    """

    def __init__(self, cfg: BurgersConfig, model: PINN):
        self.cfg = cfg
        self.model = model
        self.residual_fn = BurgersResidual(cfg)

    @torch.no_grad()
    def predict(self, t_val: float, N: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna (x_np, u_np) para u(x, t_val)."""
        cfg = self.cfg
        x = torch.linspace(cfg.x_min, cfg.x_max, N).reshape(-1, 1).to(DEVICE)
        t = torch.full_like(x, t_val)
        u = self.model(x, t)
        return x.cpu().numpy().ravel(), u.cpu().numpy().ravel()

    def fft_residual(
        self, t_eval: float, N: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computa o resíduo r(x, t_eval) e retorna seu espectro de Fourier.

        Retorna
        -------
        freqs : np.ndarray  — frequências espaciais (positivas)
        amps  : np.ndarray  — |R(k)| normalizado
        """
        cfg = self.cfg
        x = torch.linspace(cfg.x_min, cfg.x_max, N).reshape(-1, 1).to(DEVICE)
        t = torch.full_like(x, t_eval)

        self.model.train()  # garante que autograd funciona
        res = self.residual_fn(self.model, x, t).detach().cpu().numpy().ravel()
        self.model.eval()

        dx = (cfg.x_max - cfg.x_min) / N
        R = np.fft.rfft(res)
        freqs = np.fft.rfftfreq(N, d=dx)
        amps = np.abs(R) / N
        return freqs, amps

    def diagnose(self, t_eval: float, N: int = 512) -> str:
        """
        Diagnóstico textual: difusão vs dispersão numérica.
        Baseia-se na assimetria do espectro de potência.
        """
        freqs, amps = self.fft_residual(t_eval, N)
        total_energy = np.sum(amps ** 2)
        if total_energy < 1e-12:
            return "Resíduo ~ 0: PINN convergiu bem neste instante."

        mid = len(freqs) // 2
        low_energy = np.sum(amps[:mid] ** 2)
        high_energy = np.sum(amps[mid:] ** 2)

        if low_energy > 5 * high_energy:
            return (
                "DIAGNÓSTICO: Difusão Numérica dominante — erro concentrado em "
                "baixas frequências (amortecimento espúrio)."
            )
        else:
            return (
                "DIAGNÓSTICO: Dispersão Numérica dominante — energia espalhada em "
                "altas frequências (oscilações espúrias)."
            )


# ---------------------------------------------------------------------------
# 7. FUNÇÕES DE VISUALIZAÇÃO
# ---------------------------------------------------------------------------

def plot_solution(
    cfg: BurgersConfig,
    analyzer: BurgersAnalyzer,
    t_vals: Optional[List[float]] = None,
    savefig: Optional[str] = None,
):
    """Plota u(x, t) para cada instante em t_vals."""
    if t_vals is None:
        t_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(1, len(t_vals), figsize=(4 * len(t_vals), 4), sharey=True)
    fig.suptitle(
        r"PINN — Eq. de Burgers   $u_t + u\,u_x = \nu\,u_{xx}$"
        f"\n  ν = {cfg.nu:.5f}",
        fontsize=13,
    )

    for ax, t_val in zip(axes, t_vals):
        x_np, u_np = analyzer.predict(t_val)
        ax.plot(x_np, u_np, "r-", lw=2, label="PINN")
        # CI como referência
        x_ic = np.linspace(cfg.x_min, cfg.x_max, 256)
        u_ic = cfg.u0(torch.tensor(x_ic, dtype=torch.float32)).numpy()
        if t_val == 0.0:
            ax.plot(x_ic, u_ic, "k--", lw=1, label="CI exata")
        ax.set_title(f"t = {t_val:.2f}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("u(x, t)")
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150)
    plt.show()


def plot_loss_history(trainer: BurgersTrainer, savefig: Optional[str] = None):
    """Plota histórico de loss (total, dados, física)."""
    h = trainer.history
    epochs = range(len(h["loss"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(epochs, h["loss"], label="Total", lw=2)
    ax.semilogy(epochs, h["loss_data"], label="Dados (CI+CC)", lw=1.5, ls="--")
    ax.semilogy(epochs, h["loss_phys"], label="Física (resíduo EDP)", lw=1.5, ls=":")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (escala log)")
    ax.set_title("Histórico de Treinamento — PINN Burgers")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150)
    plt.show()


def plot_residual_fourier(
    analyzer: BurgersAnalyzer,
    t_eval: float,
    N: int = 512,
    savefig: Optional[str] = None,
):
    """
    Espectro de Fourier do resíduo r(x, t_eval).
    Diagnóstico visual: difusão vs dispersão numérica.
    """
    freqs, amps = analyzer.fft_residual(t_eval, N)
    diag = analyzer.diagnose(t_eval, N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- painel esquerdo: resíduo no espaço ---
    cfg = analyzer.cfg
    x = np.linspace(cfg.x_min, cfg.x_max, N)
    # recomputa resíduo para plot
    x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    t_t = torch.full_like(x_t, t_eval)
    analyzer.model.train()
    with torch.enable_grad():
        res_np = analyzer.residual_fn(analyzer.model, x_t, t_t).detach().cpu().numpy().ravel()
    analyzer.model.eval()

    axes[0].plot(x, res_np, "b-", lw=1.5)
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].set_title(f"Resíduo r(x, t={t_eval:.2f})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("r(x)")
    axes[0].grid(True, alpha=0.3)

    # --- painel direito: espectro |R(k)| ---
    axes[1].semilogy(freqs[1:], amps[1:], "r-", lw=1.5)
    axes[1].set_title("Espectro de Fourier do Resíduo |R(k)|")
    axes[1].set_xlabel("Frequência espacial k")
    axes[1].set_ylabel("|R(k)|")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle(diag, fontsize=11, color="navy", y=1.02)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()
    print(diag)


def plot_heatmap(
    cfg: BurgersConfig,
    analyzer: BurgersAnalyzer,
    Nx: int = 200,
    Nt: int = 200,
    savefig: Optional[str] = None,
):
    """Mapa de calor u(x, t) no domínio completo."""
    x_grid = torch.linspace(cfg.x_min, cfg.x_max, Nx)
    t_grid = torch.linspace(cfg.t_min, cfg.t_max, Nt)
    X, T = torch.meshgrid(x_grid, t_grid, indexing="ij")

    with torch.no_grad():
        U = analyzer.model(
            X.reshape(-1, 1).to(DEVICE), T.reshape(-1, 1).to(DEVICE)
        ).cpu().reshape(Nx, Nt)

    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.contourf(T.numpy(), X.numpy(), U.numpy(), levels=100, cmap="RdBu_r")
    fig.colorbar(c, ax=ax, label="u(x, t)")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("PINN — Solução u(x, t) da Eq. de Burgers")
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# 8. FUNÇÃO DE ALTO NÍVEL (conveniência)
# ---------------------------------------------------------------------------

def run_burgers(cfg: Optional[BurgersConfig] = None, verbose: bool = True) -> Tuple[PINN, BurgersTrainer, BurgersAnalyzer]:
    """
    Executa o pipeline completo: build → treino → análise.

    Retorna
    -------
    model, trainer, analyzer
    """
    if cfg is None:
        cfg = BurgersConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = PINN(cfg)
    dataset = BurgersDataset(cfg)
    trainer = BurgersTrainer(cfg, model, dataset)
    analyzer = BurgersAnalyzer(cfg, model)

    if verbose:
        total = sum(p.numel() for p in model.parameters())
        print(f"PINN Burgers | parâmetros: {total:,} | device: {DEVICE}")
        print(f"ν = {cfg.nu:.6f} | domínio: [{cfg.x_min},{cfg.x_max}] × [{cfg.t_min},{cfg.t_max}]")
        print(f"N_ic={cfg.N_ic}, N_bc={cfg.N_bc}, N_f={cfg.N_f}\n")

    trainer.train(verbose=verbose)
    model.eval()
    return model, trainer, analyzer
