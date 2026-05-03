# PINN Studio - Design System e Referências UI/UX

Para criar uma interface moderna, leve e agradável aos olhos (inspirada no **Cursor**, Vercel, Linear, etc.), adotaremos os seguintes princípios de design:

## 1. Filosofia Visual (Minimalismo Técnico)
* **Dark Mode Nativo:** Fundo predominantemente escuro (tons de cinza chumbo, não preto absoluto) para reduzir a fadiga visual, comum em ferramentas de desenvolvimento e ciência de dados.
* **Bordas Sutis e "Glassmorphism":** Elementos de interface (painéis, modais) com bordas finas semi-transparentes (`rgba(255,255,255,0.1)`) e leve desfoque de fundo (`backdrop-filter: blur()`).
* **Foco no Conteúdo:** A interface deve "desaparecer". Os gráficos (Matplotlib/Plotly) e os códigos devem ser as estrelas da tela.

## 2. Paleta de Cores (Sugestão "Cursor/Linear Style")

### Tema Escuro (Dark)
* **Fundo Principal (App):** `#0e0e11` (Muito escuro, quase preto)
* **Fundo de Painéis/Cards:** `#16161a` ou `#1c1c21`
* **Texto Principal:** `#ededed` (Cinza muito claro)
* **Texto Secundário (Labels, Muted):** `#a1a1aa`
* **Bordas e Separadores:** `#27272a`
* **Cor de Destaque (Accent/Ação):** `#3b82f6` (Azul brilhante) ou `#8b5cf6` (Roxo tech).

## 3. Tipografia
* **Texto Geral (UI):** Fontes sem serifa limpas e modernas.
  * Sugestões: `Inter`, `Roboto`, `Geist` ou a fonte nativa do sistema (`system-ui, -apple-system, sans-serif`).
* **Código e Valores Numéricos (Monospace):** Usado para exibir Loss, Hiperparâmetros, Fórmulas.
  * Sugestões: `JetBrains Mono`, `Fira Code` ou `Geist Mono`.

## 4. Estrutura de Layout (Dashboard)

A tela deve ser dividida de forma funcional para maximizar o espaço dos gráficos:

* **Sidebar (Esquerda - 250px a 300px):**
  * Logo / Nome do App.
  * Formulário compacto de configuração (Sliders, Dropdowns para Advecção, Neurônios, etc.).
  * Botão de Ação Primário: **"Run Simulation"** (com animação de carregamento elegante).
* **Área Principal (Direita - Restante da tela):**
  * **Header:** Título do experimento atual, status (ex: `🟢 Completed`, `🟡 Running`) e abas de navegação (Ex: `Dashboard`, `Histórico`, `Comparação`).
  * **Grid de Resultados:**
    * Linha 1 (Cards de Métricas/KPIs): Loss Final, Viscosidade Numérica, Tempo de Execução.
    * Linha 2 (Gráficos Principais): Painéis contendo o Heatmap 2D e o Espectro de Fourier.
    * Linha 3 (Análise Expandida): Gráfico de Loss e Monitor de Energia/Comparativo Analítico.

## 5. Micro-Interações e UX
* **Feedback Visual Instantâneo:** Quando o usuário clica em "Run", os botões devem ser desabilitados e um estado de "Loading..." deve aparecer na tela principal.
* **Preview em Tempo Real:** Se o usuário muda o Slider do $\sigma$ da Gaussiana, um pequeno gráfico de *preview* da Condição Inicial se atualiza instantaneamente antes mesmo de rodar a PINN completa.
* **Hover Effects:** Botões e painéis devem ter uma transição suave de brilho (ex: `transition: all 0.2s ease`) ao passar o mouse.

## 6. Stack Front-End para Implementação Rápida
* **React + Vite:** Para um servidor de desenvolvimento ultra-rápido.
* **Vanilla CSS (Moderno):** Uso pesado de CSS Variables (`--bg-color`, `--text-primary`) para facilitar a implementação do design system sem frameworks pesados.
* **Ícones:** `Lucide React` ou `Phosphor Icons` (ícones minimalistas e com peso de linha consistente).
* **Gráficos:** `Recharts` ou renderização direta de imagens geradas pelo Matplotlib via base64/URL da FastAPI.

---
*Este documento guiará a construção dos componentes CSS e React na próxima fase.*
