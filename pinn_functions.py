"""
pinn_functions.py
=================
Catálogo de condições iniciais u₀(x) para testes com PINNs.
MS901/MT861 — Diagnóstico Físico do Resíduo.

Categorias
----------
  SUAVES        : gaussiana, seno, cosseno, pacote de ondas
  DESCONTÍNUAS  : degrau (Heaviside), Riemann (choque), duplo degrau
  DEFORMADAS    : hat (triangular), trapezoidal, dente de serra, onda quadrada
  COMPOSTAS     : soliton, N-wave, multi-bump, Ricker (mexican hat)
  EXATAS        : soluções analíticas para advecção e Burgers

Todas retornam np.ndarray (para usar com ic_bc_points do pinn_core).
Versões torch disponíveis via wrapper torch_fn().

Uso:
    from pinn_functions import gaussian, riemann_shock, catalog
    u0 = gaussian(center=0, sigma=0.3)
    vals = u0(x)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
# Tipo base: todas as funções recebem np.ndarray → np.ndarray
# ═══════════════════════════════════════════════════════════════
ICFunc = Callable[[np.ndarray], np.ndarray]


def torch_fn(fn: ICFunc) -> "Callable":
    """Wrapper: converte ICFunc (numpy) para uso com torch tensors."""
    import torch

    def _wrapper(x_tensor):
        x_np = x_tensor.detach().cpu().numpy()
        result = fn(x_np.ravel())
        return torch.tensor(result, dtype=torch.float32).reshape_as(x_tensor)

    return _wrapper


# ═══════════════════════════════════════════════════════════════
#  1. FUNÇÕES SUAVES
# ═══════════════════════════════════════════════════════════════

def gaussian(
    center: float = 0.0,
    sigma: float = 0.2,
    amplitude: float = 1.0,
) -> ICFunc:
    """
    Gaussiana:  A · exp(−(x−c)² / (2σ²))

    Perfil suave que testa a capacidade da PINN de propagar
    sem deformar. Na advecção, deve manter o formato exato.
    Na Burgers invíscida, desenvolve choque eventualmente.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    fn.__name__ = f"gaussian(c={center}, σ={sigma})"
    return fn


def sine(k: float = 1.0, amplitude: float = 1.0) -> ICFunc:
    """
    Seno:  A · sin(kπx)

    Padrão clássico de PINN (Burgers viscosa).
    """
    def fn(x: np.ndarray) -> np.ndarray:
        return amplitude * np.sin(k * np.pi * x)
    fn.__name__ = f"sine(k={k})"
    return fn


def neg_sine(k: float = 1.0) -> ICFunc:
    """
    −sin(πx): condição inicial clássica da Burgers viscosa.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        return -np.sin(k * np.pi * x)
    fn.__name__ = f"neg_sine(k={k})"
    return fn


def cosine(k: float = 1.0, amplitude: float = 1.0) -> ICFunc:
    """Cosseno: A · cos(kπx)"""
    def fn(x: np.ndarray) -> np.ndarray:
        return amplitude * np.cos(k * np.pi * x)
    fn.__name__ = f"cosine(k={k})"
    return fn


def wave_packet(
    k0: float = 6.0,
    sigma: float = 0.3,
    center: float = 0.0,
) -> ICFunc:
    """
    Pacote de ondas:  cos(k₀πx) · exp(−(x−c)²/(2σ²))

    Testa resolução de frequências altas pela PINN.
    Ideal para detectar dispersão numérica no espectro de Fourier.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        envelope = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        return np.cos(k0 * np.pi * x) * envelope
    fn.__name__ = f"wave_packet(k₀={k0}, σ={sigma})"
    return fn


def smooth_bump(
    center: float = 0.0,
    radius: float = 0.5,
    amplitude: float = 1.0,
) -> ICFunc:
    """
    Bump suave C∞:  A · exp(−1/(1−((x−c)/r)²))  para |x−c| < r, 0 fora.

    Suporte compacto, infinitamente diferenciável.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        s = (x - center) / radius
        result = np.zeros_like(x)
        mask = np.abs(s) < 1.0
        result[mask] = amplitude * np.exp(-1.0 / (1.0 - s[mask] ** 2))
        return result
    fn.__name__ = f"smooth_bump(c={center}, r={radius})"
    return fn


# ═══════════════════════════════════════════════════════════════
#  2. FUNÇÕES DESCONTÍNUAS
# ═══════════════════════════════════════════════════════════════

def heaviside(x0: float = 0.0, u_left: float = 1.0, u_right: float = 0.0) -> ICFunc:
    """
    Degrau de Heaviside:
        u_left  se x < x0
        u_right se x ≥ x0

    Gera choque na Burgers (se u_left > u_right).
    Gera rarefação (se u_left < u_right).
    """
    def fn(x: np.ndarray) -> np.ndarray:
        return np.where(x < x0, u_left, u_right)
    fn.__name__ = f"heaviside(x₀={x0}, {u_left}/{u_right})"
    return fn


def riemann_shock(x0: float = 0.0) -> ICFunc:
    """
    Dado de Riemann para choque:
        u = 1 se x < x0,  u = 0 se x ≥ x0

    Referência: slides Abreu & Florindo, eq. (3).
    """
    return heaviside(x0, u_left=1.0, u_right=0.0)


def riemann_rarefaction(x0: float = 0.0) -> ICFunc:
    """
    Dado de Riemann para rarefação:
        u = −1 se x < x0,  u = 1 se x ≥ x0

    Referência: slides Abreu & Florindo, eq. (4).
    """
    return heaviside(x0, u_left=-1.0, u_right=1.0)


def double_step(
    x1: float = -0.5,
    x2: float = 0.5,
    u_in: float = 1.0,
    u_out: float = 0.0,
) -> ICFunc:
    """
    Duplo degrau (pulso retangular):
        u_in  se x1 ≤ x ≤ x2
        u_out fora

    Duas descontinuidades → testa dois choques simultâneos.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        return np.where((x >= x1) & (x <= x2), u_in, u_out)
    fn.__name__ = f"double_step([{x1},{x2}])"
    return fn


def three_state(
    x1: float = -1.0,
    x2: float = 1.0,
    u_left: float = 1.0,
    u_mid: float = 0.0,
    u_right: float = -0.5,
) -> ICFunc:
    """
    Três estados:
        u_left  se x < x1
        u_mid   se x1 ≤ x < x2
        u_right se x ≥ x2

    Gera interação complexa de ondas.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        out = np.full_like(x, u_right)
        out[x < x2] = u_mid
        out[x < x1] = u_left
        return out
    fn.__name__ = f"three_state({u_left}/{u_mid}/{u_right})"
    return fn


# ═══════════════════════════════════════════════════════════════
#  3. FUNÇÕES DEFORMADAS / COM CANTOS
# ═══════════════════════════════════════════════════════════════

def hat_triangle(
    center: float = 0.0,
    width: float = 1.0,
    height: float = 1.0,
) -> ICFunc:
    """
    Chapéu triangular (hat function):
        Sobe linearmente até center, desce linearmente.

    Contínua mas NÃO diferenciável no pico → testa cantos.
    """
    hw = width / 2
    def fn(x: np.ndarray) -> np.ndarray:
        return height * np.maximum(0.0, 1.0 - np.abs(x - center) / hw)
    fn.__name__ = f"hat(c={center}, w={width})"
    return fn


def trapezoid(
    x_start: float = -0.5,
    x_rise: float = -0.3,
    x_fall: float = 0.3,
    x_end: float = 0.5,
    height: float = 1.0,
) -> ICFunc:
    """
    Trapezoidal: sobe, platô, desce.
    Contínua mas com cantos nas transições.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        # rampa de subida
        mask_up = (x >= x_start) & (x < x_rise)
        y[mask_up] = height * (x[mask_up] - x_start) / (x_rise - x_start)
        # platô
        mask_top = (x >= x_rise) & (x <= x_fall)
        y[mask_top] = height
        # rampa de descida
        mask_down = (x > x_fall) & (x <= x_end)
        y[mask_down] = height * (x_end - x[mask_down]) / (x_end - x_fall)
        return y
    fn.__name__ = f"trapezoid([{x_start},{x_end}])"
    return fn


def sawtooth(period: float = 1.0, amplitude: float = 1.0) -> ICFunc:
    """
    Dente de serra (sawtooth wave):
        Sobe linearmente, cai abruptamente.
    Periódica com descontinuidades de salto.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        phase = (x / period) % 1.0
        return amplitude * (2 * phase - 1)
    fn.__name__ = f"sawtooth(T={period})"
    return fn


def square_wave(period: float = 1.0, amplitude: float = 1.0) -> ICFunc:
    """
    Onda quadrada: +A e −A alternados.
    Descontinuidades puras a cada meio-período.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        return amplitude * np.sign(np.sin(2 * np.pi * x / period))
    fn.__name__ = f"square_wave(T={period})"
    return fn


def piecewise_linear(
    breakpoints: List[float],
    values: List[float],
) -> ICFunc:
    """
    Interpolação linear por partes definida por breakpoints e valores.
    Fora do intervalo, extrapola constante.

    Exemplo:
        piecewise_linear([-1, -0.5, 0, 0.5, 1], [0, 1, -1, 0.5, 0])
    """
    bp = np.array(breakpoints, dtype=float)
    vl = np.array(values, dtype=float)

    def fn(x: np.ndarray) -> np.ndarray:
        return np.interp(x, bp, vl)
    fn.__name__ = f"piecewise_linear({len(breakpoints)} pontos)"
    return fn


# ═══════════════════════════════════════════════════════════════
#  4. FUNÇÕES COMPOSTAS / ESPECIAIS
# ═══════════════════════════════════════════════════════════════

def soliton(
    center: float = 0.0,
    c: float = 2.0,
) -> ICFunc:
    """
    Soliton tipo KdV:  (c/2) · sech²(√c/2 · (x − center))

    Perfil que mantém a forma durante propagação (em KdV).
    Em Burgers, o comportamento é diferente — útil para comparar.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        arg = np.sqrt(c) / 2 * (x - center)
        return (c / 2) / np.cosh(arg) ** 2
    fn.__name__ = f"soliton(c={c})"
    return fn


def n_wave(amplitude: float = 1.0) -> ICFunc:
    """
    N-wave (perfil em forma de N):
        u(x) = A · x · exp(−x²)

    Solução assintótica da Burgers invíscida — a onda de difusão
    natural que emerge de dados compactos.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        return amplitude * x * np.exp(-x ** 2)
    fn.__name__ = f"n_wave(A={amplitude})"
    return fn


def ricker_wavelet(
    center: float = 0.0,
    sigma: float = 0.2,
    amplitude: float = 1.0,
) -> ICFunc:
    """
    Ricker wavelet (Mexican Hat):
        A · (1 − ((x−c)/σ)²) · exp(−(x−c)²/(2σ²))

    Segundo derivada da Gaussiana. Rica em frequências —
    excelente para testar resolução espectral da PINN.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        s = (x - center) / sigma
        return amplitude * (1 - s ** 2) * np.exp(-s ** 2 / 2)
    fn.__name__ = f"ricker(c={center}, σ={sigma})"
    return fn


def multi_bump(
    centers: Optional[List[float]] = None,
    sigmas: Optional[List[float]] = None,
    amplitudes: Optional[List[float]] = None,
) -> ICFunc:
    """
    Soma de gaussianas: ∑ Aᵢ exp(−(x−cᵢ)²/(2σᵢ²))

    Testa múltiplas escalas e interação entre perfis.
    """
    if centers is None:
        centers = [-0.5, 0.0, 0.5]
    if sigmas is None:
        sigmas = [0.1, 0.15, 0.1]
    if amplitudes is None:
        amplitudes = [1.0, -0.5, 0.8]

    def fn(x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for c, s, a in zip(centers, sigmas, amplitudes):
            y += a * np.exp(-((x - c) ** 2) / (2 * s ** 2))
        return y
    fn.__name__ = f"multi_bump({len(centers)} bumps)"
    return fn


def sin_plus_step(
    k: float = 1.0,
    x0: float = 0.0,
    step_height: float = 0.5,
) -> ICFunc:
    """
    sin(πx) + degrau: combina componente suave + descontinuidade.
    Excelente para diagnosticar como a PINN lida com ambos.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        return np.sin(k * np.pi * x) + step_height * np.where(x >= x0, 1.0, 0.0)
    fn.__name__ = f"sin_plus_step(k={k}, x₀={x0})"
    return fn


def chirp(
    f0: float = 1.0,
    f1: float = 10.0,
    x_min: float = -1.0,
    x_max: float = 1.0,
) -> ICFunc:
    """
    Chirp (varredura de frequência):
        sin(2π · f(x) · x)  onde f varia de f0 a f1.

    Testa a capacidade da PINN em resolver múltiplas escalas
    de frequência simultaneamente.
    """
    def fn(x: np.ndarray) -> np.ndarray:
        t_norm = (x - x_min) / (x_max - x_min + 1e-12)
        freq = f0 + (f1 - f0) * t_norm
        return np.sin(2 * np.pi * freq * t_norm)
    fn.__name__ = f"chirp({f0}→{f1} Hz)"
    return fn


# ═══════════════════════════════════════════════════════════════
#  5. SOLUÇÕES EXATAS (para comparação)
# ═══════════════════════════════════════════════════════════════

def exact_advection(
    u0_fn: ICFunc,
    a: float = 1.0,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Solução exata da advecção linear U_t + a U_x = 0:
        U(x, t) = U₀(x − a·t)

    Retorna callable (x_np, t_val) → u_np.
    """
    def fn(x: np.ndarray, t: float) -> np.ndarray:
        return u0_fn(x - a * t)
    return fn


def exact_burgers_rarefaction(
    u_left: float = -1.0,
    u_right: float = 1.0,
    x0: float = 0.0,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Solução exata do Riemann para Burgers invíscida (rarefação).
    Válida quando u_left < u_right (rarefação fan).

        u(x,t) = u_left               se x < u_left·t + x0
        u(x,t) = (x − x0) / t         se u_left·t ≤ x − x0 ≤ u_right·t
        u(x,t) = u_right              se x > u_right·t + x0
    """
    def fn(x: np.ndarray, t: float) -> np.ndarray:
        if t < 1e-14:
            return np.where(x < x0, u_left, u_right)
        xi = (x - x0) / t
        return np.where(xi < u_left, u_left,
               np.where(xi > u_right, u_right, xi))
    return fn


def exact_burgers_shock(
    u_left: float = 1.0,
    u_right: float = 0.0,
    x0: float = 0.0,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Solução exata do Riemann para Burgers invíscida (choque).
    Válida quando u_left > u_right (condição de Rankine-Hugoniot).

        Velocidade do choque: s = (u_left + u_right) / 2
        u(x,t) = u_left   se x < s·t + x0
        u(x,t) = u_right  se x ≥ s·t + x0
    """
    s = (u_left + u_right) / 2.0

    def fn(x: np.ndarray, t: float) -> np.ndarray:
        return np.where(x < s * t + x0, u_left, u_right)
    return fn


# ═══════════════════════════════════════════════════════════════
#  6. CATÁLOGO (dicionário com todas as funções pré-configuradas)
# ═══════════════════════════════════════════════════════════════

catalog: Dict[str, ICFunc] = {
    # Suaves
    "gaussian":         gaussian(),
    "gaussian_narrow":  gaussian(sigma=0.1),
    "gaussian_wide":    gaussian(sigma=0.5),
    "sine":             sine(),
    "neg_sine":         neg_sine(),
    "cosine":           cosine(),
    "wave_packet":      wave_packet(),
    "smooth_bump":      smooth_bump(),

    # Descontínuas
    "heaviside":            heaviside(),
    "riemann_shock":        riemann_shock(),
    "riemann_rarefaction":  riemann_rarefaction(),
    "double_step":          double_step(),
    "three_state":          three_state(),

    # Deformadas / com cantos
    "hat_triangle":     hat_triangle(),
    "trapezoid":        trapezoid(),
    "sawtooth":         sawtooth(),
    "square_wave":      square_wave(),

    # Compostas
    "soliton":          soliton(),
    "n_wave":           n_wave(),
    "ricker":           ricker_wavelet(),
    "multi_bump":       multi_bump(),
    "sin_plus_step":    sin_plus_step(),
    "chirp":            chirp(),
}


# ═══════════════════════════════════════════════════════════════
#  7. VISUALIZAÇÃO DO CATÁLOGO
# ═══════════════════════════════════════════════════════════════

def plot_catalog(
    x_min: float = -2.0,
    x_max: float = 2.0,
    N: int = 1000,
    subset: Optional[List[str]] = None,
    savefig: Optional[str] = None,
) -> plt.Figure:
    """
    Plota todas as funções do catálogo em subplots organizados.

    Parâmetros
    ----------
    subset : lista de nomes para plotar (None = todas)
    """
    funcs = {k: v for k, v in catalog.items() if subset is None or k in subset}
    n = len(funcs)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.ravel()
    x = np.linspace(x_min, x_max, N)

    for i, (name, fn) in enumerate(funcs.items()):
        ax = axes[i]
        ax.plot(x, fn(x), lw=1.8, color="tab:blue")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.2)

    # Desliga eixos extras
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Catálogo de Condições Iniciais u₀(x)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    return fig


def plot_function(
    fn: ICFunc,
    x_min: float = -2.0,
    x_max: float = 2.0,
    N: int = 1000,
    title: Optional[str] = None,
    savefig: Optional[str] = None,
) -> plt.Figure:
    """Plota uma única função u₀(x)."""
    x = np.linspace(x_min, x_max, N)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x, fn(x), lw=2, color="tab:blue")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("x")
    ax.set_ylabel("u₀(x)")
    ax.set_title(title or getattr(fn, "__name__", "u₀(x)"))
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches="tight")
    return fig
