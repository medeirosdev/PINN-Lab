"""
pinn_core.py
============
Biblioteca genérica de Physics-Informed Neural Networks (PINNs).
MS901/MT861 — Diagnóstico Físico do Resíduo via Equação Modificada.

Uso:
    from pinn_core import PINNConfig, PINN, Trainer, Analyzer, make_dataset

A lib é agnóstica à EDP — você passa a função de resíduo como argumento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# ─────────────────────────────────────────────
# 1. CONFIGURAÇÃO
# ─────────────────────────────────────────────

@dataclass
class PINNConfig:
    """
    Todos os hiperparâmetros em um único objeto.

    Parâmetros da rede
    ------------------
    n_layers   : número de camadas ocultas
    n_neurons  : neurônios por camada oculta
    activation : 'tanh' | 'sin' | 'relu' | 'swish' | 'sigmoid'

    Treino
    ------
    lr_adam      : taxa de aprendizado Adam
    epochs_adam  : épocas Adam
    epochs_lbfgs : iterações L-BFGS (0 = desligado)
    lambda_phys  : peso da loss de física vs dados
    seed         : semente de reprodutibilidade

    Domínio
    -------
    x_min, x_max : limite espacial
    t_min, t_max : limite temporal
    """

    # Rede
    n_layers: int = 4
    n_neurons: int = 40
    activation: str = "tanh"

    # Treino
    lr_adam: float = 1e-3
    epochs_adam: int = 10_000
    epochs_lbfgs: int = 0
    lambda_phys: float = 1.0
    seed: int = 42

    # Domínio (usado para gerar pontos, se necessário)
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0


# ─────────────────────────────────────────────
# 2. REDE NEURAL (MLP)
# ─────────────────────────────────────────────

class _Sin(nn.Module):
    def forward(self, x): return torch.sin(x)

class _Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

_ACT = {
    "tanh":    nn.Tanh,
    "relu":    nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "sin":     lambda: _Sin(),
    "swish":   lambda: _Swish(),
}


class PINN(nn.Module):
    """
    MLP genérica: (x, t) ∈ ℝ² → u ∈ ℝ.

    Parâmetros
    ----------
    n_inputs  : dimensão da entrada (padrão 2 = [x, t])
    n_outputs : dimensão da saída   (padrão 1 = u)
    """

    def __init__(
        self,
        cfg: PINNConfig,
        n_inputs: int = 2,
        n_outputs: int = 1,
    ):
        super().__init__()
        act = _ACT.get(cfg.activation, nn.Tanh)

        layers: List[nn.Module] = [nn.Linear(n_inputs, cfg.n_neurons), act()]
        for _ in range(cfg.n_layers - 1):
            layers += [nn.Linear(cfg.n_neurons, cfg.n_neurons), act()]
        layers.append(nn.Linear(cfg.n_neurons, n_outputs))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────
# 3. GERADOR DE DATASETS
# ─────────────────────────────────────────────

@dataclass
class Dataset:
    """
    Contém pontos de dados supervisionados e pontos de colocação.

    x_data, t_data, u_data : pontos onde u é conhecido (CI + CC)
    x_f, t_f               : pontos de colocação (EDP deve valer)
    """
    x_data: torch.Tensor
    t_data: torch.Tensor
    u_data: torch.Tensor
    x_f:    torch.Tensor
    t_f:    torch.Tensor


def make_dataset(
    x_data: np.ndarray,
    t_data: np.ndarray,
    u_data: np.ndarray,
    N_f: int,
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    seed: int = 42,
) -> Dataset:
    """
    Constrói o Dataset a partir de arrays numpy.

    Os pontos de colocação (x_f, t_f) são amostrados
    uniformemente no domínio [x_min,x_max] × [t_min,t_max].
    """
    rng = np.random.default_rng(seed)

    def _t(a): return torch.tensor(a, dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    x_f = rng.uniform(x_min, x_max, (N_f, 1)).astype(np.float32)
    t_f = rng.uniform(t_min, t_max, (N_f, 1)).astype(np.float32)

    return Dataset(
        x_data=_t(x_data),
        t_data=_t(t_data),
        u_data=_t(u_data),
        x_f=_t(x_f),
        t_f=_t(t_f),
    )


def ic_bc_points(
    cfg: PINNConfig,
    u0_fn: Callable[[np.ndarray], np.ndarray],
    N_ic: int = 200,
    N_bc: int = 100,
    bc_left: Optional[float] = 0.0,
    bc_right: Optional[float] = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera pontos de CI e CC automaticamente.

    Retorna arrays (x, t, u) prontos para passar ao make_dataset.
    """
    x_ic = np.linspace(cfg.x_min, cfg.x_max, N_ic)
    t_ic = np.zeros(N_ic)
    u_ic = u0_fn(x_ic)

    xs, ts, us = [x_ic], [t_ic], [u_ic]

    t_bc = np.linspace(cfg.t_min, cfg.t_max, N_bc)
    if bc_left is not None:
        xs.append(np.full(N_bc, cfg.x_min))
        ts.append(t_bc)
        us.append(np.full(N_bc, bc_left))
    if bc_right is not None:
        xs.append(np.full(N_bc, cfg.x_max))
        ts.append(t_bc)
        us.append(np.full(N_bc, bc_right))

    return np.concatenate(xs), np.concatenate(ts), np.concatenate(us)


# ─────────────────────────────────────────────
# 4. FUNÇÕES DE RESÍDUO PRONTAS
# ─────────────────────────────────────────────

def residual_burgers(
    model: PINN,
    x: torch.Tensor,
    t: torch.Tensor,
    nu: float = 0.01 / np.pi,
) -> torch.Tensor:
    """
    Resíduo da Equação de Burgers viscosa:
        r = u_t + u·u_x − ν·u_xx
    """
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    r = u_t + u * u_x
    if nu != 0.0:
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        r = r - nu * u_xx
    return r


def residual_advection(
    model: PINN,
    x: torch.Tensor,
    t: torch.Tensor,
    a: float = 1.0,
) -> torch.Tensor:
    """
    Resíduo da Advecção Linear:
        r = u_t + a·u_x
    """
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return u_t + a * u_x


# ─────────────────────────────────────────────
# 5. TRAINER
# ─────────────────────────────────────────────

class Trainer:
    """
    Treina uma PINN com qualquer função de resíduo.

    Parâmetros
    ----------
    residual_fn : função (model, x_f, t_f) → tensor de resíduos
    """

    def __init__(
        self,
        cfg: PINNConfig,
        model: PINN,
        dataset: Dataset,
        residual_fn: Callable,
    ):
        self.cfg = cfg
        self.model = model.to(DEVICE)
        self.ds = dataset
        self.residual_fn = residual_fn
        self.history: Dict[str, List[float]] = {
            "loss": [], "loss_data": [], "loss_phys": []
        }

    # ── losses ──────────────────────────────
    def _losses(self):
        u_pred = self.model(self.ds.x_data, self.ds.t_data)
        loss_data = torch.mean((u_pred - self.ds.u_data) ** 2)

        res = self.residual_fn(self.model, self.ds.x_f.clone(), self.ds.t_f.clone())
        loss_phys = torch.mean(res ** 2)

        loss = loss_data + self.cfg.lambda_phys * loss_phys
        return loss, loss_data, loss_phys

    def _record(self, loss, ld, lp):
        self.history["loss"].append(loss.item())
        self.history["loss_data"].append(ld.item())
        self.history["loss_phys"].append(lp.item())

    # ── Adam ────────────────────────────────
    def train_adam(self, verbose: bool = True) -> None:
        cfg = self.cfg
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr_adam)
        sched = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max(1, cfg.epochs_adam // 2), gamma=0.1
        )
        log_every = max(1, cfg.epochs_adam // 10)

        if verbose:
            print(f"[Adam] {cfg.epochs_adam} épocas | device={DEVICE}")
            print(f"{'Época':<8} {'Total':<14} {'Dados':<14} {'Física':<14}")
            print("─" * 52)

        for ep in range(cfg.epochs_adam):
            opt.zero_grad()
            loss, ld, lp = self._losses()
            loss.backward()
            opt.step()
            sched.step()
            self._record(loss, ld, lp)

            if verbose and ep % log_every == 0:
                print(f"{ep:<8} {loss.item():<14.4e} {ld.item():<14.4e} {lp.item():<14.4e}")

        if verbose:
            print(f"[Adam] final → {self.history['loss'][-1]:.4e}\n")

    # ── L-BFGS ──────────────────────────────
    def train_lbfgs(self, verbose: bool = True) -> None:
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
            loss, ld, lp = self._losses()
            loss.backward()
            self._record(loss, ld, lp)
            return loss

        if verbose:
            print(f"[L-BFGS] max_iter={self.cfg.epochs_lbfgs} ...")
        opt.step(closure)
        if verbose:
            print(f"[L-BFGS] final → {self.history['loss'][-1]:.4e}\n")

    # ── pipeline ────────────────────────────
    def train(self, verbose: bool = True) -> None:
        torch.manual_seed(self.cfg.seed)
        self.train_adam(verbose=verbose)
        if self.cfg.epochs_lbfgs > 0:
            self.train_lbfgs(verbose=verbose)
        self.model.eval()


# ─────────────────────────────────────────────
# 6. ANALYZER (Fourier do resíduo)
# ─────────────────────────────────────────────

class Analyzer:
    """
    Diagnóstico físico do resíduo via análise de Fourier.

    Conecta a "Teoria da Equação Modificada" ao treinamento da PINN:
      - Espectro concentrado em baixas freq → difusão numérica
      - Picos em freq intermediárias        → dispersão numérica
    """

    def __init__(self, cfg: PINNConfig, model: PINN, residual_fn: Callable):
        self.cfg = cfg
        self.model = model
        self.residual_fn = residual_fn

    # ── predição ─────────────────────────────
    @torch.no_grad()
    def predict(self, t_val: float, N: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """u(x, t_val) para x em grade uniforme."""
        cfg = self.cfg
        x = torch.linspace(cfg.x_min, cfg.x_max, N).reshape(-1, 1).to(DEVICE)
        t = torch.full_like(x, t_val)
        return x.cpu().numpy().ravel(), self.model(x, t).cpu().numpy().ravel()

    # ── resíduo em grade ─────────────────────
    def residual_grid(self, t_val: float, N: int = 512) -> np.ndarray:
        cfg = self.cfg
        x = torch.linspace(cfg.x_min, cfg.x_max, N).reshape(-1, 1).to(DEVICE)
        t = torch.full_like(x, t_val)
        self.model.train()
        with torch.enable_grad():
            r = self.residual_fn(self.model, x, t).detach().cpu().numpy().ravel()
        self.model.eval()
        return r

    # ── FFT do resíduo ────────────────────────
    def fft_residual(
        self, t_val: float, N: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna (freqs_positivas, amplitudes |R(k)|/N)."""
        cfg = self.cfg
        r = self.residual_grid(t_val, N)
        dx = (cfg.x_max - cfg.x_min) / N
        R = np.fft.rfft(r)
        freqs = np.fft.rfftfreq(N, d=dx)
        amps = np.abs(R) / N
        return freqs, amps

    # ── viscosidade numérica estimada ─────────
    def nu_numerical(self, t_val: float, N: int = 512) -> float:
        """
        Estima ν_num da equação modificada neural via:
            ν_num ≈ median{ |R(k)| / (k² |Û(k)|) }
        """
        cfg = self.cfg
        x = torch.linspace(cfg.x_min, cfg.x_max, N).reshape(-1, 1).to(DEVICE)
        t = torch.full_like(x, t_val)

        r = self.residual_grid(t_val, N)
        with torch.no_grad():
            u = self.model(x, t).cpu().numpy().ravel()

        dx = (cfg.x_max - cfg.x_min) / N
        R = np.fft.rfft(r)
        Uhat = np.fft.rfft(u)
        k = 2 * np.pi * np.fft.rfftfreq(N, d=dx)

        mask = (k > 0) & (np.abs(Uhat) > 1e-10)
        if not mask.any():
            return 0.0
        return float(np.median(np.abs(R[mask]) / (k[mask] ** 2 * np.abs(Uhat[mask]))))

    # ── diagnóstico textual ───────────────────
    def diagnose(self, t_val: float, N: int = 512) -> str:
        freqs, amps = self.fft_residual(t_val, N)
        total = np.sum(amps ** 2)
        if total < 1e-14:
            return f"t={t_val:.2f} | Resíduo ≈ 0 → PINN convergiu bem."

        mid = len(freqs) // 2
        low = np.sum(amps[:mid] ** 2)
        high = np.sum(amps[mid:] ** 2)
        nu_est = self.nu_numerical(t_val, N)
        tipo = "Difusão Numérica" if low > 5 * high else "Dispersão Numérica"

        return (
            f"t={t_val:.2f} | {tipo} dominante\n"
            f"  ν_num ≈ {nu_est:.3e} | "
            f"E_baixa={low:.3e} | E_alta={high:.3e}"
        )

    # ── energia L² ────────────────────────────
    def l2_energy(self, t_val: float, N: int = 512) -> float:
        """∫ u²(x,t) dx por regra do trapézio."""
        cfg = self.cfg
        _, u = self.predict(t_val, N)
        dx = (cfg.x_max - cfg.x_min) / (N - 1)
        return float(np.trapezoid(u ** 2, dx=dx))

    def gronwall_bound(self, t_val: float, norm_a_C1: float, N: int = 512) -> float:
        """Cota de Gronwall: exp(||a||_C¹ · t) · E(0)."""
        E0 = self.l2_energy(0.0, N)
        return np.exp(norm_a_C1 * t_val) * E0

    def scan_energy(
        self, norm_a_C1: float, N_times: int = 20, N_grid: int = 256
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Varre t e retorna (t_vals, energias, cotas_gronwall).
        """
        cfg = self.cfg
        t_vals = np.linspace(cfg.t_min, cfg.t_max, N_times)
        energias = np.array([self.l2_energy(tv, N_grid) for tv in t_vals])
        cotas = np.array([self.gronwall_bound(tv, norm_a_C1, N_grid) for tv in t_vals])
        return t_vals, energias, cotas
