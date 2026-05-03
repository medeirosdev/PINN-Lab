# PINN Studio: Plataforma de Simulação e Análise

## 1. Visão Geral
O **PINN Studio** será uma aplicação completa (com interface gráfica) projetada para orquestrar, executar e analisar simulações de Redes Neurais Informadas pela Física (PINNs). O objetivo principal é facilitar a experimentação com diferentes Equações Diferenciais Parciais (EDPs), como Advecção Linear e Burgers, testando variadas condições iniciais e hiperparâmetros, sem a necessidade de alterar código manualmente a cada teste.

O grande diferencial será o armazenamento persistente dos resultados (métricas, diagnósticos de difusão/dispersão numérica, histórico de *loss* e imagens geradas) em um banco de dados, permitindo buscas, filtragens e comparações lado a lado de simulações passadas.

## 2. Funcionalidades Principais (Core Features)

* **Configurador de Experimentos:**
    * **Seleção de Modelo Físico:** Escolha entre Advecção Linear, Burgers Invíscida, Burgers Viscosa, etc.
    * **Definição de Domínio:** Configuração interativa de $x_{min}, x_{max}, t_{min}, t_{max}$.
    * **Condições Iniciais (Catálogo):** Seleção visual a partir do catálogo (Gaussiana, Seno, Degrau, Hat, etc.) com ajustes de parâmetros (ex: amplitude, sigma, frequência).
    * **Hiperparâmetros da Rede:** Configuração de camadas, neurônios, função de ativação, taxa de aprendizado (Adam) e épocas (Adam/L-BFGS).
* **Monitoramento em "Tempo Real":**
    * Exibição do progresso do treinamento (época atual, *loss* total, de dados e da física).
    * Gráficos dinâmicos atualizados durante o treinamento (ex: histórico de *loss*).
* **Análise e Visualização (Pós-processamento Automático):**
    * Geração e exibição imediata do "Diagnóstico Físico do Resíduo" (Difusão vs. Dispersão).
    * Visualização interativa das soluções em fatias temporais, Heatmap 2D, Superfície 3D e Animações (player embutido).
    * Espectro de Fourier e Monitoramento de Energia (Cota de Gronwall).
* **Gerenciamento de Histórico (Banco de Dados):**
    * Salvamento automático de cada experimento (metadata, hiperparâmetros, métricas finais).
    * Armazenamento das imagens geradas (referenciadas no banco de dados).
    * **Dashboard de Comparação:** Selecionar dois ou mais experimentos passados e comparar seus gráficos de *loss*, erro final e diagnóstico de Fourier lado a lado.

## 3. Arquitetura do Sistema

Para alcançar o objetivo de uma interface fluida com processamento pesado (PyTorch) em background, sugere-se uma arquitetura cliente-servidor leve ou uma aplicação Desktop moderna.

**Opção Recomendada (Web App Local):**
* **Backend / Motor Computacional:** Python (FastAPI ou Flask). Responsável por receber as configurações, rodar o código do `pinn_core.py` em *background tasks* ou *workers* (Celery/Redis se necessário, ou `asyncio` nativo para simplicidade local) e interagir com o banco de dados.
* **Frontend:** Interface Web moderna (React, Vue.js, ou Streamlit/Gradio para desenvolvimento ultra-rápido). O Streamlit é altamente recomendado para a primeira versão, pois integra perfeitamente com Python e Matplotlib/Plotly.
* **Banco de Dados:** **SQLite** (padrão, local, arquivo único `pinns.db`). Para escalabilidade futura ou implantação via Docker, pode-se usar SQLAlchemy como ORM, permitindo trocar facilmente para PostgreSQL no futuro sem reescrever o código.

## 4. Estrutura do Banco de Dados (Esquema Proposto)

Usando um ORM (como SQLAlchemy), teremos uma tabela principal para centralizar os dados de cada corrida (run).

**Tabela: `experiments`**
| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `id` | Integer (PK) | Identificador único da simulação |
| `timestamp` | DateTime | Data e hora de início |
| `model_type` | String | Ex: "advection_linear", "burgers_viscous" |
| `a_velocity` | Float | (Específico Advecção) Velocidade |
| `nu_viscosity`| Float | (Específico Burgers) Viscosidade |
| `u0_name` | String | Ex: "gaussian", "sine" |
| `u0_params` | JSON | Parâmetros da CI (ex: `{"sigma": 0.2, "center": 0.5}`) |
| `n_layers` | Integer | Número de camadas ocultas |
| `n_neurons` | Integer | Neurônios por camada |
| `activation` | String | Função de ativação (ex: "tanh") |
| `loss_final` | Float | Valor final da função de perda |
| `nu_numerical`| Float | Viscosidade numérica estimada via Fourier |
| `diagnostico` | String | Texto do diagnóstico físico |
| `status` | String | "RUNNING", "COMPLETED", "FAILED" |
| `results_dir` | String | Caminho local para a pasta com as imagens geradas |

## 5. Fases de Desenvolvimento

### Fase 1: Fundação do Banco de Dados e Backend
* Configurar o SQLAlchemy com SQLite local.
* Criar as funções de CRUD (Create, Read, Update, Delete) para os experimentos.
* Refatorar o script `main.py` atual para ser uma função invocável (ex: `run_simulation(config_dict)`) que salva automaticamente no banco ao finalizar, em vez de ser um script monolítico.

### Fase 2: Interface Básica (Configuração e Execução)
* Criar a interface de usuário (sugestão: **Streamlit** ou **Gradio**).
* Implementar formulários no painel lateral para configurar o `PINNConfig` e escolher as funções do `pinn_functions.py`.
* Botão "Iniciar Simulação" que dispara o backend e mostra um spinner de progresso.

### Fase 3: Visualização e Dashboard
* Integrar as funções de visualização do `pinn_viz.py` para exibir os resultados diretamente na interface web (Streamlit renderiza figuras do Matplotlib nativamente).
* Exibir os diagnósticos em caixas de destaque (KPIs).

### Fase 4: Histórico e Comparação
* Criar uma aba "Histórico" na interface.
* Exibir uma tabela listando todos os experimentos do banco SQLite.
* Implementar seleção múltipla para abrir uma view de "Comparação", plotando gráficos de diferentes IDs de experimentos lado a lado (útil para ver o impacto de alterar de 40 para 10 neurônios, por exemplo).

## 6. Por que esta abordagem?
* **Produtividade:** Evita a execução via terminal e edição manual de variáveis.
* **Reprodutibilidade:** O banco de dados saberá exatamente qual seed, qual taxa de aprendizado e qual condição inicial gerou cada gráfico e resultado.
* **Escalabilidade (Docker):** A arquitetura permite que, futuramente, o backend, o frontend e um banco PostgreSQL sejam conteinerizados via `docker-compose`, tornando-se uma ferramenta de laboratório instalável em qualquer servidor.



Experimentação e Configuração
O plano atual trata cada simulação como independente, mas seria muito útil ter um modo de varredura de hiperparâmetros (hyperparameter sweep) — você define um grid (ex: neurons ∈ {10, 40, 100}, lr ∈ {1e-3, 1e-4}) e o sistema gera e enfileira automaticamente todas as combinações. Isso transforma o PINN Studio numa ferramenta de pesquisa de verdade. Complementando isso, um sistema de tags e notas por experimento (anotações livres tipo "testei reduzir LR na época 3000") tornaria o histórico muito mais navegável.
Diagnóstico e Análise Científica
O plano já prevê diagnóstico de difusão/dispersão, mas algumas adições elevariam bastante o valor analítico: um gráfico de convergência com detecção automática de stagnação (identificar quando o loss para de cair e sugerir ajustes), uma métrica de conservação de massa ao longo do tempo (integral de u(x,t) dx), e um comparativo com solução analítica quando disponível — para Advecção Linear e Burgers viscosa existem soluções de referência que permitem calcular o erro L2 real, não apenas o loss.
Gerenciamento de Experimentos
Além do histórico, valeria ter exportação de experimentos em formatos portáteis (CSV de métricas, ZIP com imagens e JSON de configuração completa) e a possibilidade de reimportar uma configuração passada diretamente no formulário para replicar ou variar um experimento anterior com um clique — algo tipo "clonar este experimento".
Interface e UX
Um modo de pausa/retomada de treinamento seria valioso para explorar o estado intermediário da rede antes de continuar. Também faz sentido um preview da condição inicial que atualiza em tempo real enquanto o usuário ajusta os parâmetros (sigma, amplitude etc.) antes de rodar — evita surpresas e acelera a intuição física. Para o dashboard de comparação, uma visualização de radar/spider das métricas principais permitiria comparar 5+ experimentos de uma vez sem poluir a tela com gráficos.
Robustez e Infraestrutura
Algumas adições pequenas com impacto grande: checkpointing automático dos pesos (salvar o estado da rede a cada N épocas em arquivo .pt), permitindo retomar sem perder progresso em caso de crash. Um log estruturado por experimento (arquivo .log separado por run, referenciado no banco) facilita debugging. E um modo "dry run" que valida a configuração e estima o tempo de execução antes de rodar de verdade.
Visão de Longo Prazo
Se o projeto crescer, duas ideias se destacam: um sistema de plugins para novas EDPs (uma interface padrão que qualquer novo modelo precisa implementar, tornando a adição de Navier-Stokes ou equação do calor trivial), e suporte a equações inversas — onde a PINN é usada para inferir um parâmetro desconhecido (ex: estimar a viscosidade ν a partir de dados observados), que é uma das aplicações mais poderosas de PINNs na prática.