"""
PINN - Physics-Informed Neural Network
Resolvendo a equacao de Burgers viscosa 1D:

    du/dt + u * du/dx = nu * d²u/dx²

Dominio: x in [-1, 1], t in [0, 1]
Condicao inicial: u(x, 0) = -sin(pi * x)
Condicoes de contorno: u(-1, t) = 0, u(1, t) = 0
Viscosidade: nu = 0.01 / pi

A ideia central de uma PINN:
- Uma rede neural recebe (x, t) e retorna u(x, t)
- A loss function tem DOIS termos:
  1) Loss de DADOS: a rede deve satisfazer as condicoes iniciais e de contorno
  2) Loss de FISICA: o residuo da EDP deve ser ~0 em pontos do dominio
- Usamos autodiferenciacao (autograd) pra calcular du/dt, du/dx, d²u/dx²
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Reproducibilidade
torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1) DEFINIR A REDE NEURAL
# ============================================================
# A rede e simples: recebe (x, t) -> retorna u(x, t)
# Usamos camadas fully-connected com ativacao tanh
# (tanh e boa pra PINNs porque e suave e diferenciavel)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 40),    # entrada: (x, t) -> 40 neuronios
            nn.Tanh(),
            nn.Linear(40, 40),   # camada oculta 1
            nn.Tanh(),
            nn.Linear(40, 40),   # camada oculta 2
            nn.Tanh(),
            nn.Linear(40, 40),   # camada oculta 3
            nn.Tanh(),
            nn.Linear(40, 1),    # saida: u(x, t)
        )

    def forward(self, x, t):
        # Concatena x e t como entrada da rede
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


# ============================================================
# 2) CALCULAR O RESIDUO DA EDP (a "fisica")
# ============================================================
# Aqui esta a magica da PINN: usamos autograd pra calcular
# as derivadas parciais de u em relacao a x e t.
# O residuo e: f = du/dt + u * du/dx - nu * d²u/dx²
# Se a rede aprendeu a solucao correta, f ≈ 0.

nu = 0.01 / np.pi  # viscosidade

def compute_residual(model, x, t):
    """Calcula o residuo da equacao de Burgers."""
    # Precisamos de gradientes em relacao a x e t
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    # Derivadas de primeira ordem via autograd
    # grad retorna du/dx e du/dt
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]

    # Derivada de segunda ordem: d²u/dx²
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]

    # Residuo da EDP de Burgers: du/dt + u*du/dx - nu*d²u/dx² = 0
    residual = u_t + u * u_x - nu * u_xx
    return residual


# ============================================================
# 3) GERAR OS DADOS DE TREINAMENTO
# ============================================================
# Precisamos de dois tipos de pontos:
#   a) Pontos de contorno/inicial (onde sabemos o valor de u)
#   b) Pontos de colocacao no dominio (onde a EDP deve ser satisfeita)

# --- Condicao inicial: u(x, 0) = -sin(pi*x) ---
N_ic = 200  # pontos na condicao inicial
x_ic = torch.linspace(-1, 1, N_ic).reshape(-1, 1)
t_ic = torch.zeros(N_ic, 1)
u_ic = -torch.sin(np.pi * x_ic)  # valor conhecido

# --- Condicoes de contorno: u(-1, t) = 0 e u(1, t) = 0 ---
N_bc = 100  # pontos em cada contorno
t_bc = torch.linspace(0, 1, N_bc).reshape(-1, 1)

# Contorno esquerdo x = -1
x_bc_left = -torch.ones(N_bc, 1)
u_bc_left = torch.zeros(N_bc, 1)

# Contorno direito x = 1
x_bc_right = torch.ones(N_bc, 1)
u_bc_right = torch.zeros(N_bc, 1)

# Juntando todos os dados conhecidos
x_data = torch.cat([x_ic, x_bc_left, x_bc_right])
t_data = torch.cat([t_ic, t_bc, t_bc])
u_data = torch.cat([u_ic, u_bc_left, u_bc_right])

# --- Pontos de colocacao (onde a EDP deve valer) ---
# Pontos aleatorios no dominio [-1,1] x [0,1]
N_f = 10000  # pontos de colocacao
x_f = (2 * torch.rand(N_f, 1) - 1)   # x in [-1, 1]
t_f = torch.rand(N_f, 1)              # t in [0, 1]


# ============================================================
# 4) TREINAR A PINN
# ============================================================
# A loss total tem dois termos:
#   L_total = L_dados + lambda * L_fisica
#
# L_dados = MSE entre u_pred e u_real nos pontos de contorno/inicial
# L_fisica = MSE do residuo da EDP nos pontos de colocacao
#
# Isso e exatamente o que aparece no slide 9 do seu material!

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Treinamento com Adam + reducao de learning rate
epochs = 10000
lambda_physics = 1.0  # peso do termo de fisica

print("Treinando a PINN...")
print(f"{'Epoca':<10} {'Loss Total':<15} {'Loss Dados':<15} {'Loss Fisica':<15}")
print("-" * 55)

for epoch in range(epochs):
    optimizer.zero_grad()

    # --- Loss de dados (condicoes iniciais e de contorno) ---
    u_pred_data = model(x_data, t_data)
    loss_data = torch.mean((u_pred_data - u_data) ** 2)

    # --- Loss de fisica (residuo da EDP) ---
    residual = compute_residual(model, x_f, t_f)
    loss_physics = torch.mean(residual ** 2)

    # --- Loss total ---
    loss = loss_data + lambda_physics * loss_physics

    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"{epoch:<10} {loss.item():<15.6f} {loss_data.item():<15.6f} {loss_physics.item():<15.6f}")

    # Reduz learning rate na metade do treino
    if epoch == 5000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4

print(f"\nTreinamento concluido! Loss final: {loss.item():.6f}")


# ============================================================
# 5) VISUALIZAR O RESULTADO
# ============================================================
# Vamos plotar a solucao u(x, t) aprendida pela PINN

model.eval()

# Criar grid para plot
x_plot = torch.linspace(-1, 1, 256).reshape(-1, 1)
t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

fig, axes = plt.subplots(1, len(t_values), figsize=(18, 4))
fig.suptitle("PINN - Solucao da Equacao de Burgers Viscosa\n"
             r"$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x}"
             r" = \nu\frac{\partial^2 u}{\partial x^2}$, "
             r"$\nu = 0.01/\pi$",
             fontsize=13)

for i, t_val in enumerate(t_values):
    t_plot = t_val * torch.ones(256, 1)
    with torch.no_grad():
        u_plot = model(x_plot, t_plot).numpy()

    axes[i].plot(x_plot.numpy(), u_plot, 'r-', linewidth=2, label='PINN')
    axes[i].set_title(f't = {t_val}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('u(x, t)')
    axes[i].set_ylim([-1.1, 1.1])
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

plt.tight_layout()
plt.savefig('/home/medeiros/Projetos/MS901Pinns/resultado_pinn_burgers.png', dpi=150)
plt.show()
print("\nGrafico salvo em: resultado_pinn_burgers.png")


# --- Plot 2D (heatmap) ---
x_grid = torch.linspace(-1, 1, 200)
t_grid = torch.linspace(0, 1, 200)
X, T = torch.meshgrid(x_grid, t_grid, indexing='ij')

x_flat = X.reshape(-1, 1)
t_flat = T.reshape(-1, 1)

with torch.no_grad():
    u_flat = model(x_flat, t_flat).numpy()

U = u_flat.reshape(200, 200)

fig2, ax2 = plt.subplots(figsize=(8, 5))
c = ax2.contourf(T.numpy(), X.numpy(), U, levels=100, cmap='RdBu_r')
fig2.colorbar(c, ax=ax2, label='u(x, t)')
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_title('PINN - Solucao u(x,t) da Eq. de Burgers')
plt.tight_layout()
plt.savefig('/home/medeiros/Projetos/MS901Pinns/resultado_pinn_burgers_2d.png', dpi=150)
plt.show()
print("Heatmap salvo em: resultado_pinn_burgers_2d.png")
