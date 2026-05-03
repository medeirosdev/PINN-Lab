"""
pinn_advection_lib.py
=====================
Biblioteca PINN para a Equação de Advecção-Transporte Linear.

Equação: ∂U/∂t + a(x,t) ∂U/∂x = 0

Base teórica (LinearAdvectionEnergy.pdf — MS901/MT861):
  • Solução exata via método das características: U(x,t) = U₀(x − at)  (a constante)
  • Estimativa de energia (Lema 2.1 + Gronwall):
        ∫ U²(x,t) dx  ≤  exp(||a||_C¹ · t) · ∫ U₀²(x) dx
  • Loss residual não nula → viscosidade artificial ν_num u_xx na equação modificada

Uso básico:
    from pinn_advection_lib import AdvectionConfig, AdvectionTrainer, AdvectionAnalyzer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Device padrão
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÃO
# ---------------------------------------------------------------------------

@dataclass
class AdvectionConfig:
    """Hiperparâmetros do experimento de advecção."""

    # Velocidade de advecção: callable a(x, t) → tensor, ou constante float
    a: float = 1.0          # caso mais simples: a constante

    # Domínio
    x_min: float = 0.0
    x_max: float = 2.0
    t_min: float = 0.0
    t_max: float = 1.0

    # Condição inicial padrão: sin(π x)
    u0: Callable[[torch.Tensor], torch.Tensor] = field(default=None, repr=False)

    # Pontos de amostragem
    N_ic: int = 200
    N_f: int = 8_000

    # Arquitetura
    n_layers: int = 4
    n_neurons: int = 40
    activation: str = "tanh"

    # Treino
    lr_adam: float = 1e-3
    epochs_adam: int = 8_000
    epochs_lbfgs: int = 500
    lambda_phys: float = 1.0

    # Reprodutibilidade
    seed: int = 42

    def __post_init__(self):
        if self.u0 is None:
            import math
            self.u0 = lambda x: torch.sin(math.pi * x)

    def a_value(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Retorna o campo de velocidade como tensor.
        Se self.a for float/int, retorna escalar broadcast.
        Se self.a for callable, chama a(x, t).
        """
        if callable(self.a):
            return self.a(x, t)
        return torch.full_like(x, float(self.a))

    def a_norm_C1(self) -> float:
        """
        Estima ||a||_C¹ para a cota de Gronwall.
        Para a constante, ||a||_C¹ = |a| (derivada zero).
        Para a variável, usa valor heurístico (usuário pode sobrescrever).
        """
        if not callable(self.a):
            return abs(float(self.a))
        return 1.0  # heurístico; usuário deve refinar


# ---------------------------------------------------------------------------
# 2. REDE NEURAL
# ---------------------------------------------------------------------------

class _Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class PINN(nn.Module):
    """MLP para aproximar U(x, t)."""

    _activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "sin": lambda: _Sin()}

    def __init__(self, cfg: AdvectionConfig):
        super().__init__()
        act_cls = self._activations.get(cfg.activation, nn.Tanh)
        layers: List[nn.Module] = [nn.Linear(2, cfg.n_neurons), act_cls()]
        for _ in range(cfg.n_layers - 1):
            layers += [nn.Linear(cfg.n_neurons, cfg.n_neurons), act_cls()]
        layers.append(nn.Linear(cfg.n_neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))


# ---------------------------------------------------------------------------
# 3. RESÍDUO DA EDP
# ---------------------------------------------------------------------------

class AdvectionResidual:
    """
    Resíduo: r(x,t) = ∂U/∂t + a(x,t) ∂U/∂x

    Se a PINN convergiu perfeitamente, r ≡ 0.
    Um resíduo não nulo indica equação modificada:
        U_t + a U_x = ν_num U_xx + O(h³)
    onde ν_num é a viscosidade numérica artificial.
    """

    def __init__(self, cfg: AdvectionConfig):
        self.cfg = cfg

    def __call__(
        self, model: PINN, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        U = model(x, t)

        U_t = torch.autograd.grad(
            U, t, grad_outputs=torch.ones_like(U), create_graph=True
        )[0]
        U_x = torch.autograd.grad(
            U, x, grad_outputs=torch.ones_like(U), create_graph=True
        )[0]

        a_val = self.cfg.a_value(x, t)
        return U_t + a_val * U_x


# ---------------------------------------------------------------------------
# 4. DATASET
# ---------------------------------------------------------------------------

class AdvectionDataset:
    """Pontos de CI e colocação."""

    def __init__(self, cfg: AdvectionConfig):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.cfg = cfg
        self._build()

    def _build(self):
        cfg = self.cfg

        # Condição inicial U(x, 0) = u0(x)
        x_ic = torch.linspace(cfg.x_min, cfg.x_max, cfg.N_ic).reshape(-1, 1)
        t_ic = torch.zeros(cfg.N_ic, 1)
        u_ic = cfg.u0(x_ic)

        self.x_ic = x_ic.to(DEVICE)
        self.t_ic = t_ic.to(DEVICE)
        self.u_ic = u_ic.to(DEVICE)

        # Pontos de colocação
        lx = cfg.x_max - cfg.x_min
        lt = cfg.t_max - cfg.t_min
        self.x_f = (torch.rand(cfg.N_f, 1) * lx + cfg.x_min).to(DEVICE)
        self.t_f = (torch.rand(cfg.N_f, 1) * lt + cfg.t_min).to(DEVICE)


# ---------------------------------------------------------------------------
# 5. SOLUÇÃO EXATA (método das características)
# ---------------------------------------------------------------------------

class ExactSolution:
    """
    Solução analítica para a constante: U(x,t) = U₀(x − a·t).

    Só válida quando a é constante e o domínio é periódico ou
    o perfil inicial não chega nas bordas no intervalo de tempo.
    """

    def __init__(self, cfg: AdvectionConfig):
        self.cfg = cfg
        if callable(cfg.a):
            raise ValueError(
                "ExactSolution só suporta a constante. "
                "Para a(x,t), use solução numérica de referência."
            )
        self.a_const = float(cfg.a)

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """U_exact(x, t) = U₀(x − a·t)."""
        x_char = x - self.a_const * t
        return self.cfg.u0(x_char)

    def numpy(self, x_np: np.ndarray, t_val: float) -> np.ndarray:
        x_t = torch.tensor(x_np, dtype=torch.float32).reshape(-1, 1)
        t_t = torch.full_like(x_t, t_val)
        with torch.no_grad():
            return self(x_t, t_t).numpy().ravel()


# ---------------------------------------------------------------------------
# 6. MONITOR DE ENERGIA (Gronwall)
# ---------------------------------------------------------------------------

class EnergyMonitor:
    """
    Monitora a norma L² da solução ao longo do tempo e compara com a
    cota da estimativa de energia (Lema 2.1 + Gronwall):

        E(t) = ∫ U²(x,t) dx  ≤  exp(||a||_C¹ · t) · E(0)

    Se a PINN respeitar a física, E(t) deve ficar abaixo da cota.
    Um excesso indica que a rede está injetando energia artificial.
    """

    def __init__(self, cfg: AdvectionConfig, model: PINN, dataset: AdvectionDataset):
        self.cfg = cfg
        self.model = model
        self.ds = dataset
        self._norm_C1 = cfg.a_norm_C1()

    def energy_at(self, t_val: float, N: int = 512) -> float:
        """Aproxima ∫ U²(x,t) dx por regra do trapézio."""
        cfg = self.cfg
        x = torch.linspace(cfg.x_min, cfg.x_max, N).reshape(-1, 1).to(DEVICE)
        t = torch.full_like(x, t_val)
        with torch.no_grad():
            U = self.model(x, t).cpu().numpy().ravel()
        dx = (cfg.x_max - cfg.x_min) / (N - 1)
        return np.trapz(U ** 2, dx=dx)

    def gronwall_bound(self, t_val: float) -> float:
        """Cota superior: exp(||a||_C¹ · t) · E(0)."""
        E0 = self.energy_at(0.0)
        return np.exp(self._norm_C1 * t_val) * E0

    def scan(self, N_times: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Varre t ∈ [t_min, t_max] e retorna (t_vals, energias, cotas).
        """
        cfg = self.cfg
        t_vals = np.linspace(cfg.t_min, cfg.t_max, N_times)
        energias = np.array([self.energy_at(tv) for tv in t_vals])
        cotas = np.array([self.gronwall_bound(tv) for tv in t_vals])
        return t_vals, energias, cotas


# ---------------------------------------------------------------------------
# 7. TRAINER
# ---------------------------------------------------------------------------

class AdvectionTrainer:
    """Loop de treino: Adam + L-BFGS opcional."""

    def __init__(self, cfg: AdvectionConfig, model: PINN, dataset: AdvectionDataset):
        self.cfg = cfg
        self.model = model.to(DEVICE)
        self.ds = dataset
        self.residual_fn = AdvectionResidual(cfg)
        self.history = {"loss": [], "loss_ic": [], "loss_phys": []}

    def _compute_losses(self):
        # Loss condição inicial
        U_pred_ic = self.model(self.ds.x_ic, self.ds.t_ic)
        loss_ic = torch.mean((U_pred_ic - self.ds.u_ic) ** 2)
        # Loss física
        res = self.residual_fn(self.model, self.ds.x_f, self.ds.t_f)
        loss_phys = torch.mean(res ** 2)
        loss = loss_ic + self.cfg.lambda_phys * loss_phys
        return loss, loss_ic, loss_phys

    def train_adam(self, verbose: bool = True):
        cfg = self.cfg
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr_adam)
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=cfg.epochs_adam // 2, gamma=0.1
        )

        if verbose:
            print(f"{'Época':<8} {'Total':<14} {'IC':<14} {'Física':<14}")
            print("-" * 52)

        for epoch in range(cfg.epochs_adam):
            opt.zero_grad()
            loss, li, lp = self._compute_losses()
            loss.backward()
            opt.step()
            scheduler.step()

            self.history["loss"].append(loss.item())
            self.history["loss_ic"].append(li.item())
            self.history["loss_phys"].append(lp.item())

            if verbose and epoch % (cfg.epochs_adam // 10) == 0:
                print(f"{epoch:<8} {loss.item():<14.6f} {li.item():<14.6f} {lp.item():<14.6f}")

        if verbose:
            print(f"\n[Adam] Loss final: {self.history['loss'][-1]:.6e}")

    def train_lbfgs(self, verbose: bool = True):
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
            loss, li, lp = self._compute_losses()
            loss.backward()
            self.history["loss"].append(loss.item())
            self.history["loss_ic"].append(li.item())
            self.history["loss_phys"].append(lp.item())
            return loss

        opt.step(closure)
        if verbose:
            print(f"[L-BFGS] Loss final: {self.history['loss'][-1]:.6e}")

    def train(self, verbose: bool = True):
        self.train_adam(verbose=verbose)
        if self.cfg.epochs_lbfgs > 0:
            self.train_lbfgs(verbose=verbose)


# ---------------------------------------------------------------------------
# 8. ANALYZER — Fourier do resíduo
# ---------------------------------------------------------------------------

class AdvectionAnalyzer:
    """
    Análise de Fourier do resíduo r(x, t_eval) = U_t + a U_x.

    Conecta a análise de von Neumann clássica à equação modificada da PINN:
      - Concentração em baixas frequências  → difusão numérica (ν_num U_xx)
      - Picos em frequências intermediárias → dispersão numérica (β U_xxx)
    """

    def __init__(self, cfg: AdvectionConfig, model: PINN):
        self.cfg = cfg
        self.model = model
        self.residual_fn = AdvectionResidual(cfg)

    @torch.no_grad()
    def predict(self, t_val: float, N: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna (x_np, U_np) para U(x, t_val)."""
        cfg = self.cfg
        x = torch.linspace(cfg.x_min, cfg.x_max, N).reshape(-1, 1).to(DEVICE)
        t = torch.full_like(x, t_val)
        U = self.model(x, t)
        return x.cpu().numpy().ravel(), U.cpu().numpy().ravel()

    def fft_residual(
        self, t_eval: float, N: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Espectro de Fourier do resíduo espacial.

        Retorna
        -------
        freqs : np.ndarray  — frequências espaciais (positivas)
        amps  : np.ndarray  — |R(k)| normalizado
        """
        cfg = self.cfg
        x = torch.linspace(cfg.x_min, cfg.x_max, N).reshape(-1, 1).to(DEVICE)
        t = torch.full_like(x, t_eval)

        self.model.train()
        res = self.residual_fn(self.model, x, t).detach().cpu().numpy().ravel()
        self.model.eval()

        dx = (cfg.x_max - cfg.x_min) / N
        R = np.fft.rfft(res)
        freqs = np.fft.rfftfreq(N, d=dx)
        amps = np.abs(R) / N
        return freqs, amps

    def estimate_numerical_viscosity(self, t_eval: float, N: int = 512) -> float:
        """
        Estima a viscosidade numérica ν_num da equação modificada.

        Metodologia (Teoria da Equação Modificada / Análise de Hirt):
          Para r(x,t) ≈ -ν_num U_xx, no espaço de Fourier:
          R(k) ≈ ν_num k² Û(k)
          → ν_num ≈ mean{ |R(k)| / (k² |Û(k)|) }  para k > 0
        """
        cfg = self.cfg
        x_t = torch.linspace(cfg.x_min, cfg.x_max, N).reshape(-1, 1).to(DEVICE)
        t_t = torch.full_like(x_t, t_eval)

        self.model.train()
        res = self.residual_fn(self.model, x_t, t_t).detach().cpu().numpy().ravel()
        self.model.eval()

        with torch.no_grad():
            U = self.model(x_t, t_t).cpu().numpy().ravel()

        dx = (cfg.x_max - cfg.x_min) / N
        R = np.fft.rfft(res)
        Uhat = np.fft.rfft(U)
        freqs = np.fft.rfftfreq(N, d=dx)
        k = 2 * np.pi * freqs

        mask = (k > 0) & (np.abs(Uhat) > 1e-10)
        if not np.any(mask):
            return 0.0

        nu_estimates = np.abs(R[mask]) / (k[mask] ** 2 * np.abs(Uhat[mask]))
        return float(np.median(nu_estimates))

    def diagnose(self, t_eval: float, N: int = 512) -> str:
        freqs, amps = self.fft_residual(t_eval, N)
        total = np.sum(amps ** 2)
        if total < 1e-12:
            return "Resíduo ~ 0: PINN convergiu bem neste instante."

        mid = len(freqs) // 2
        low = np.sum(amps[:mid] ** 2)
        high = np.sum(amps[mid:] ** 2)

        nu_est = self.estimate_numerical_viscosity(t_eval, N)

        if low > 5 * high:
            kind = "Difusão Numérica dominante"
        else:
            kind = "Dispersão Numérica dominante"

        return (
            f"DIAGNÓSTICO (t={t_eval:.2f}): {kind}\n"
            f"  Viscosidade numérica estimada: ν_num ≈ {nu_est:.4e}\n"
            f"  Energia baixas freq.: {low:.4e}  |  Energia altas freq.: {high:.4e}"
        )


# ---------------------------------------------------------------------------
# 9. VISUALIZAÇÃO
# ---------------------------------------------------------------------------

def plot_solution(
    cfg: AdvectionConfig,
    analyzer: AdvectionAnalyzer,
    t_vals: Optional[List[float]] = None,
    exact: Optional[ExactSolution] = None,
    savefig: Optional[str] = None,
):
    """Plota PINN vs solução exata (características) em vários instantes."""
    if t_vals is None:
        n = 5
        t_vals = list(np.linspace(cfg.t_min, cfg.t_max, n))

    fig, axes = plt.subplots(1, len(t_vals), figsize=(4 * len(t_vals), 4), sharey=True)
    if len(t_vals) == 1:
        axes = [axes]

    a_label = f"a={cfg.a}" if not callable(cfg.a) else "a(x,t)"
    fig.suptitle(
        r"PINN — Advecção Linear  $U_t + a\,U_x = 0$"
        f"\n{a_label}",
        fontsize=13,
    )

    for ax, t_val in zip(axes, t_vals):
        x_np, U_pinn = analyzer.predict(t_val)
        ax.plot(x_np, U_pinn, "r-", lw=2, label="PINN")
        if exact is not None:
            U_ex = exact.numpy(x_np, t_val)
            ax.plot(x_np, U_ex, "k--", lw=1.5, label="Exata")
        ax.set_title(f"t = {t_val:.2f}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("U(x, t)")
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150)
    plt.show()


def plot_energy(
    monitor: EnergyMonitor,
    N_times: int = 30,
    savefig: Optional[str] = None,
):
    """
    Evolução da norma L² vs cota de Gronwall.

    Se a curva PINN ultrapassar a cota, a rede está injetando
    energia artificial (violando a física do problema).
    """
    t_vals, energias, cotas = monitor.scan(N_times)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_vals, energias, "b-o", lw=2, ms=4, label=r"$\|U(\cdot,t)\|^2_{L^2}$ (PINN)")
    ax.plot(t_vals, cotas, "r--", lw=2, label=r"Cota Gronwall: $e^{\|a\|_{C^1}t}\,E(0)$")
    ax.fill_between(t_vals, energias, cotas, alpha=0.15, color="red", label="Margem")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\int U^2\,dx$")
    ax.set_title("Monitor de Energia — PINN Advecção\n(a PINN deve ficar abaixo da cota)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150)
    plt.show()


def plot_loss_history(trainer: AdvectionTrainer, savefig: Optional[str] = None):
    h = trainer.history
    epochs = range(len(h["loss"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(epochs, h["loss"], label="Total", lw=2)
    ax.semilogy(epochs, h["loss_ic"], label="IC", lw=1.5, ls="--")
    ax.semilogy(epochs, h["loss_phys"], label="Física (resíduo EDP)", lw=1.5, ls=":")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (escala log)")
    ax.set_title("Histórico de Treinamento — PINN Advecção")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150)
    plt.show()


def plot_residual_fourier(
    analyzer: AdvectionAnalyzer,
    t_eval: float,
    N: int = 512,
    savefig: Optional[str] = None,
):
    """Espectro de Fourier do resíduo + diagnóstico."""
    freqs, amps = analyzer.fft_residual(t_eval, N)
    diag = analyzer.diagnose(t_eval, N)

    cfg = analyzer.cfg
    x_np = np.linspace(cfg.x_min, cfg.x_max, N)
    x_t = torch.tensor(x_np, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    t_t = torch.full_like(x_t, t_eval)
    analyzer.model.train()
    with torch.enable_grad():
        res_np = analyzer.residual_fn(analyzer.model, x_t, t_t).detach().cpu().numpy().ravel()
    analyzer.model.eval()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x_np, res_np, "b-", lw=1.5)
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].set_title(f"Resíduo r(x, t={t_eval:.2f})  =  U_t + a·U_x")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("r(x)")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(freqs[1:], amps[1:] + 1e-16, "r-", lw=1.5)
    axes[1].set_title("Espectro |R(k)| — Fourier do Resíduo")
    axes[1].set_xlabel("Frequência espacial k")
    axes[1].set_ylabel("|R(k)|")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle(diag, fontsize=10, color="navy", y=1.04)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.show()
    print(diag)


# ---------------------------------------------------------------------------
# 10. PIPELINE DE ALTO NÍVEL
# ---------------------------------------------------------------------------

def run_advection(
    cfg: Optional[AdvectionConfig] = None,
    verbose: bool = True,
) -> Tuple[PINN, AdvectionTrainer, AdvectionAnalyzer, EnergyMonitor]:
    """
    Executa pipeline completo: build → treino → análise.

    Retorna
    -------
    model, trainer, analyzer, monitor
    """
    if cfg is None:
        cfg = AdvectionConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = PINN(cfg)
    dataset = AdvectionDataset(cfg)
    trainer = AdvectionTrainer(cfg, model, dataset)
    analyzer = AdvectionAnalyzer(cfg, model)
    monitor = EnergyMonitor(cfg, model, dataset)

    if verbose:
        total = sum(p.numel() for p in model.parameters())
        a_info = f"a={cfg.a}" if not callable(cfg.a) else "a(x,t) variável"
        print(f"PINN Advecção | parâmetros: {total:,} | device: {DEVICE}")
        print(f"{a_info} | domínio: [{cfg.x_min},{cfg.x_max}] × [{cfg.t_min},{cfg.t_max}]")
        print(f"N_ic={cfg.N_ic}, N_f={cfg.N_f}\n")

    trainer.train(verbose=verbose)
    model.eval()
    return model, trainer, analyzer, monitor
