import React, { useState, useEffect } from 'react';
import { Activity, Play, RefreshCw, BarChart2, Layers, Trash2, Home, BookOpen, MonitorPlay, Save } from 'lucide-react';
import './index.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [currentRoute, setCurrentRoute] = useState('home');
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedExp, setSelectedExp] = useState(null);
  const [modalComments, setModalComments] = useState('');
  
  // Filtros
  const [filterStatus, setFilterStatus] = useState('ALL');
  const [filterModel, setFilterModel] = useState('ALL');
  const [filterU0, setFilterU0] = useState('ALL');

  const [formData, setFormData] = useState({
    model_type: 'advection_linear',
    a_velocity: 1.0,
    nu_viscosity: 0.00318,
    x_min: -1.0,
    x_max: 1.0,
    t_min: 0.0,
    t_max: 1.0,
    u0_name: 'neg_sine',
    u0_params: { center: 0.0, sigma: 0.2, amplitude: 1.0 },
    n_layers: 4,
    n_neurons: 40,
    epochs_adam: 2000,
    epochs_lbfgs: 500,
    lr_adam: 0.001
  });

  const fetchExperiments = async () => {
    try {
      const res = await fetch(`${API_URL}/experiments`);
      if (res.ok) {
        const data = await res.json();
        setExperiments(data);
      }
    } catch (err) {
      console.error("Failed to fetch experiments", err);
    }
  };

  const deleteExperiment = async (id, e) => {
    e.stopPropagation();
    if (!window.confirm(`Tem certeza que deseja deletar o experimento #${id}? Isso removerá a pasta de resultados.`)) return;
    try {
      await fetch(`${API_URL}/experiments/${id}`, { method: 'DELETE' });
      fetchExperiments();
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    if (selectedExp) {
      setModalComments(selectedExp.comments || '');
    }
  }, [selectedExp]);

  const saveComments = async () => {
    try {
      const res = await fetch(`${API_URL}/experiments/${selectedExp.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comments: modalComments })
      });
      if (res.ok) {
        const updatedExp = await res.json();
        setSelectedExp(updatedExp);
        fetchExperiments();
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchExperiments();
    const interval = setInterval(fetchExperiments, 5000); // Auto refresh
    return () => clearInterval(interval);
  }, []);

  const filteredExperiments = experiments.filter(exp => {
    if (filterStatus !== 'ALL' && exp.status !== filterStatus) return false;
    if (filterModel !== 'ALL' && exp.model_type !== filterModel) return false;
    if (filterU0 !== 'ALL' && exp.u0_name !== filterU0) return false;
    return true;
  });

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'number' ? Number(value) : value
    });
  };

  const handleParamsChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      u0_params: {
        ...formData.u0_params,
        [name]: Number(value)
      }
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await fetch(`${API_URL}/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          u0_params: {}, // Simplify for now
        })
      });
      fetchExperiments();
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div className="app-container">
      {/* Navigation Bar */}
      <nav className="navbar">
        <div className="navbar-brand" onClick={() => setCurrentRoute('home')}>
          <Activity color="var(--accent-color)" size={28} />
          PINN Lab
        </div>
        <div className="navbar-nav">
          <span className={`nav-link ${currentRoute === 'home' ? 'active' : ''}`} onClick={() => setCurrentRoute('home')}>Home</span>
          <span className={`nav-link ${currentRoute === 'studio' ? 'active' : ''}`} onClick={() => setCurrentRoute('studio')}>Lab</span>
          <span className={`nav-link ${currentRoute === 'experiments' ? 'active' : ''}`} onClick={() => setCurrentRoute('experiments')}>Experimentos</span>
          <span className={`nav-link ${currentRoute === 'manual' ? 'active' : ''}`} onClick={() => setCurrentRoute('manual')}>Manual</span>
        </div>
      </nav>

      {currentRoute === 'home' && (
        <div className="home-container">
          <div className="hero-text">
            <h1>Simulações Físicas com Redes Neurais</h1>
            <p>Plataforma interativa para explorar, treinar e diagnosticar modelos Physics-Informed Neural Networks (PINNs) utilizando equações de Advecção e Burgers.</p>
          </div>
          <div className="home-grid">
            <div className="home-card" onClick={() => setCurrentRoute('studio')}>
              <div className="icon-wrapper"><MonitorPlay size={26} /></div>
              <h3>PINN Lab</h3>
              <p>Ambiente completo de simulação. Configure hiperparâmetros, inicie treinamentos e veja as soluções convergirem em tempo real.</p>
            </div>
            <div className="home-card" onClick={() => setCurrentRoute('experiments')}>
              <div className="icon-wrapper"><Layers size={26} /></div>
              <h3>Meus Experimentos</h3>
              <p>Acesse o painel com o histórico de todos os seus experimentos. Compare métricas, adicione anotações e analise os gráficos 3D.</p>
            </div>
            <div className="home-card" onClick={() => setCurrentRoute('manual')}>
              <div className="icon-wrapper"><BookOpen size={26} /></div>
              <h3>Manual & Teoria</h3>
              <p>Aprenda como configurar as PINNs, entenda a teoria da Equação Modificada e explore o catálogo de condições iniciais disponíveis.</p>
            </div>
          </div>
        </div>
      )}

      {currentRoute === 'manual' && (
        <div style={{ padding: '40px', overflowY: 'auto', flex: 1, maxWidth: '900px', margin: '0 auto', color: 'var(--text-primary)' }}>
          <h1 style={{ marginBottom: '20px', color: '#fff' }}>Manual do PINN Lab</h1>
          <div className="card" style={{ padding: '30px' }}>
            <h3 style={{ color: 'var(--accent-color)', marginBottom: '15px' }}>1. O que são PINNs?</h3>
            <p style={{ marginBottom: '20px', lineHeight: '1.6', color: 'var(--text-muted)' }}>As Redes Neurais Informadas pela Física (PINNs) aprendem a resolver Equações Diferenciais Parciais (EDPs) minimizando não apenas o erro nos dados (Condições Iniciais e de Contorno), mas também o resíduo da equação física no interior do domínio.</p>
            
            <h3 style={{ color: 'var(--accent-color)', marginBottom: '15px' }}>2. Parâmetros de Treinamento</h3>
            <p style={{ marginBottom: '20px', lineHeight: '1.6', color: 'var(--text-muted)' }}>A plataforma utiliza um esquema de otimização híbrido. O otimizador <strong>Adam</strong> é excelente para a convergência inicial rápida, enquanto o <strong>L-BFGS</strong> (um método de quase-Newton) é usado no final do treinamento para atingir altíssima precisão numérica na loss. Tente usar 2000 épocas de Adam e 500 de L-BFGS.</p>

            <h3 style={{ color: 'var(--accent-color)', marginBottom: '15px' }}>3. Catálogo de Condições Iniciais</h3>
            <p style={{ marginBottom: '20px', lineHeight: '1.6', color: 'var(--text-muted)' }}>O Lab suporta mais de 24 condições iniciais pré-programadas para estressar a rede neural. Utilize funções suaves (como a Gaussiana e o Seno) para avaliar o quão bem a rede advecta, e funções descontínuas (como o Degrau de Heaviside ou Dente de Serra) para forçar o fenômeno de difusão/dispersão numérica e a formação de choques de Burgers.</p>

            <h3 style={{ color: 'var(--accent-color)', marginBottom: '15px' }}>4. Diagnóstico Físico via Equação Modificada</h3>
            <p style={{ marginBottom: '20px', lineHeight: '1.6', color: 'var(--text-muted)' }}>No "Dashboard" de cada experimento, o gráfico do "Espectro de Fourier do Resíduo" lhe dirá exatamente onde a rede está errando. Uma concentração de erro em baixas frequências espaciais (k pequeno) indica que a rede está injetando difusão (suavizando muito a onda). Erros em altas frequências indicam dispersão (oscilações indesejadas).</p>
          </div>
        </div>
      )}

      {(currentRoute === 'studio' || currentRoute === 'experiments') && (
        <div className="main-layout">
          {currentRoute === 'studio' && (
            <div className="sidebar">
              <h1><Activity size={24} /> PINN Lab</h1>
              
              <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          <details className="sidebar-category" open>
            <summary>Física & Condição Inicial</summary>
            <div className="category-content">
              <div className="form-group">
                <label>Modelo Físico</label>
                <select name="model_type" value={formData.model_type} onChange={handleChange}>
                  <option value="advection_linear">Advecção Linear</option>
                  <option value="burgers_viscous">Burgers Viscosa</option>
                  <option value="burgers_inviscid">Burgers Invíscida</option>
                </select>
              </div>

              {formData.model_type === 'advection_linear' && (
                <div className="form-group">
                  <label>Velocidade (a)</label>
                  <input type="number" step="0.1" name="a_velocity" value={formData.a_velocity} onChange={handleChange} />
                </div>
              )}

              {formData.model_type === 'burgers_viscous' && (
                <div className="form-group">
                  <label>Viscosidade (ν)</label>
                  <input type="number" step="0.001" name="nu_viscosity" value={formData.nu_viscosity} onChange={handleChange} placeholder="ex: 0.00318" />
                </div>
              )}

              <div className="form-group">
                <label>Condição Inicial</label>
                <select name="u0_name" value={formData.u0_name} onChange={handleChange}>
                  <optgroup label="Suaves">
                    <option value="gaussian">Gaussiana</option>
                    <option value="gaussian_narrow">Gaussiana (Estreita)</option>
                    <option value="gaussian_wide">Gaussiana (Larga)</option>
                    <option value="sine">Seno</option>
                    <option value="neg_sine">-Seno</option>
                    <option value="cosine">Cosseno</option>
                    <option value="wave_packet">Pacote de Ondas</option>
                    <option value="smooth_bump">Bump Suave</option>
                  </optgroup>
                  <optgroup label="Descontínuas">
                    <option value="heaviside">Degrau (Heaviside)</option>
                    <option value="riemann_shock">Choque de Riemann</option>
                    <option value="riemann_rarefaction">Rarefação de Riemann</option>
                    <option value="double_step">Degrau Duplo</option>
                    <option value="three_state">Três Estados</option>
                  </optgroup>
                  <optgroup label="Com Cantos / Deformadas">
                    <option value="hat_triangle">Chapéu (Triângulo)</option>
                    <option value="trapezoid">Trapézio</option>
                    <option value="sawtooth">Dente de Serra</option>
                    <option value="square_wave">Onda Quadrada</option>
                  </optgroup>
                  <optgroup label="Compostas / Exóticas">
                    <option value="soliton">Soliton</option>
                    <option value="n_wave">Onda-N</option>
                    <option value="ricker">Wavelet Ricker</option>
                    <option value="multi_bump">Multi-Bump</option>
                    <option value="sin_plus_step">Seno + Degrau</option>
                    <option value="chirp">Chirp (Freq. Var.)</option>
                  </optgroup>
                </select>
              </div>

              {formData.u0_name === 'gaussian' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <div className="form-group">
                    <label>Sigma (Largura)</label>
                    <input type="number" step="0.01" name="sigma" value={formData.u0_params.sigma} onChange={handleParamsChange} />
                  </div>
                  <div className="form-group">
                    <label>Centro</label>
                    <input type="number" step="0.1" name="center" value={formData.u0_params.center} onChange={handleParamsChange} />
                  </div>
                </div>
              )}
            </div>
          </details>

          <details className="sidebar-category">
            <summary>Domínio Espaço-Tempo</summary>
            <div className="category-content" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px 10px' }}>
              <div className="form-group">
                <label>x min</label>
                <input type="number" step="0.1" name="x_min" value={formData.x_min} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>x max</label>
                <input type="number" step="0.1" name="x_max" value={formData.x_max} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>t min</label>
                <input type="number" step="0.1" name="t_min" value={formData.t_min} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>t max</label>
                <input type="number" step="0.1" name="t_max" value={formData.t_max} onChange={handleChange} />
              </div>
            </div>
          </details>

          <details className="sidebar-category" open>
            <summary>Rede & Treinamento</summary>
            <div className="category-content">
              <div className="form-group">
                <label><Layers size={14} style={{display:'inline', marginRight:'4px', position:'relative', top:'2px'}}/> Camadas Ocultas</label>
                <input type="number" name="n_layers" value={formData.n_layers} onChange={handleChange} />
              </div>

              <div className="form-group">
                <label>Neurônios por Camada</label>
                <input type="number" name="n_neurons" value={formData.n_neurons} onChange={handleChange} />
              </div>

              <div className="form-group">
                <label>Learning Rate (Adam)</label>
                <input type="number" step="0.0001" name="lr_adam" value={formData.lr_adam} onChange={handleChange} />
              </div>

              <div className="form-group">
                <label>Épocas (Adam)</label>
                <input type="number" step="1000" name="epochs_adam" value={formData.epochs_adam} onChange={handleChange} />
              </div>

              <div className="form-group">
                <label>Épocas Otimizador (L-BFGS)</label>
                <input type="number" step="100" name="epochs_lbfgs" value={formData.epochs_lbfgs} onChange={handleChange} />
              </div>
            </div>
          </details>

          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? <RefreshCw size={18} className="animate-spin" /> : <Play size={18} />}
            Run Simulation
          </button>
        </form>
            </div>
          )}

      <div className="main-area">
        <div className="header">
          <h2><BarChart2 size={20} style={{display:'inline', marginRight:'8px', position:'relative', top:'3px'}}/> Dashboard de Simulações</h2>
          <p>Acompanhe e compare seus experimentos PINN em tempo real.</p>
        </div>

        {/* Filter Bar */}
        <div className="filter-bar" style={{ display: 'flex', gap: '15px', marginBottom: '25px', padding: '15px', backgroundColor: 'var(--bg-panel)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
          <div className="form-group" style={{ flex: 1 }}>
            <label>Status</label>
            <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)}>
              <option value="ALL">Todos</option>
              <option value="RUNNING">Em Andamento</option>
              <option value="COMPLETED">Concluídos</option>
              <option value="FAILED">Falhos</option>
            </select>
          </div>
          <div className="form-group" style={{ flex: 1 }}>
            <label>Modelo Físico</label>
            <select value={filterModel} onChange={e => setFilterModel(e.target.value)}>
              <option value="ALL">Todos</option>
              <option value="advection_linear">Advecção Linear</option>
              <option value="burgers_viscous">Burgers Viscosa</option>
              <option value="burgers_inviscid">Burgers Invíscida</option>
            </select>
          </div>
          <div className="form-group" style={{ flex: 1 }}>
            <label>Condição Inicial</label>
            <select value={filterU0} onChange={e => setFilterU0(e.target.value)}>
              <option value="ALL">Todas</option>
              <optgroup label="Suaves">
                <option value="gaussian">Gaussiana</option>
                <option value="gaussian_narrow">Gaussiana (Estreita)</option>
                <option value="gaussian_wide">Gaussiana (Larga)</option>
                <option value="sine">Seno</option>
                <option value="neg_sine">-Seno</option>
                <option value="cosine">Cosseno</option>
                <option value="wave_packet">Pacote de Ondas</option>
                <option value="smooth_bump">Bump Suave</option>
              </optgroup>
              <optgroup label="Descontínuas">
                <option value="heaviside">Degrau (Heaviside)</option>
                <option value="riemann_shock">Choque de Riemann</option>
                <option value="riemann_rarefaction">Rarefação de Riemann</option>
                <option value="double_step">Degrau Duplo</option>
                <option value="three_state">Três Estados</option>
              </optgroup>
              <optgroup label="Com Cantos / Deformadas">
                <option value="hat_triangle">Chapéu (Triângulo)</option>
                <option value="trapezoid">Trapézio</option>
                <option value="sawtooth">Dente de Serra</option>
                <option value="square_wave">Onda Quadrada</option>
              </optgroup>
              <optgroup label="Compostas / Exóticas">
                <option value="soliton">Soliton</option>
                <option value="n_wave">Onda-N</option>
                <option value="ricker">Wavelet Ricker</option>
                <option value="multi_bump">Multi-Bump</option>
                <option value="sin_plus_step">Seno + Degrau</option>
                <option value="chirp">Chirp (Freq. Var.)</option>
              </optgroup>
            </select>
          </div>
        </div>

        <div className="experiments-grid">
          {filteredExperiments.map(exp => (
            <div key={exp.id} className="card">
              <div className="card-header">
                <span className="card-title">Exp #{exp.id}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span className={`status-badge status-${exp.status}`}>{exp.status}</span>
                  <button onClick={(e) => deleteExperiment(exp.id, e)} className="btn-close" style={{ padding: '4px', color: 'var(--danger-color)' }} title="Deletar Experimento">
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
              <div className="card-body">
                <p>Modelo: <span>{exp.model_type} ({exp.model_type.includes('burgers') ? `ν=${exp.nu_viscosity ?? 0}` : `a=${exp.a_velocity}`})</span></p>
                <p>U0: <span>{exp.u0_name}</span></p>
                <p>Domínio: <span>x:[{exp.x_min}, {exp.x_max}], t:[{exp.t_min}, {exp.t_max}]</span></p>
                <p>Rede: <span>{exp.n_layers}x{exp.n_neurons}</span></p>
                <p>LR: <span>{exp.lr_adam || '1e-3'}</span></p>
                {exp.status === 'COMPLETED' && (
                  <div style={{marginTop: '15px'}}>
                    <p style={{borderTop: '1px solid var(--border-color)', paddingTop: '10px'}}>Loss Final: <span>{exp.loss_final?.toExponential(4)}</span></p>
                    <p style={{marginTop: '10px', fontSize:'0.8rem', color:'var(--accent-hover)'}}>{exp.diagnostico}</p>
                    {exp.results_dir && (
                      <button 
                        type="button"
                        className="btn-primary" 
                        style={{width: '100%', marginTop: '15px', backgroundColor: 'transparent', border: '1px solid var(--accent-color)', color: 'var(--accent-color)'}}
                        onClick={() => setSelectedExp(exp)}
                      >
                        Abrir Análise Detalhada
                      </button>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          {experiments.length === 0 && (
            <p style={{color: 'var(--text-muted)'}}>Nenhum experimento rodado ainda. Inicie uma simulação no painel lateral.</p>
          )}
          {experiments.length > 0 && filteredExperiments.length === 0 && (
            <p style={{color: 'var(--text-muted)'}}>Nenhum experimento corresponde aos filtros selecionados.</p>
          )}
        </div>
      </div>
    </div>
  )}
      
      {/* Modal Detalhes */}
      {selectedExp && (
        <div className="modal-overlay" onClick={() => setSelectedExp(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title">
                <BarChart2 size={24} color="var(--accent-color)" />
                Análise do Experimento #{selectedExp.id}
              </div>
              <button className="btn-close" onClick={() => setSelectedExp(null)}>✕ Fechar</button>
            </div>
            
            <div className="modal-body">
              <div style={{display: 'flex', gap: '20px', marginBottom: '30px', flexWrap: 'wrap'}}>
                <div style={{flex: 1, minWidth: '200px'}}>
                  <p style={{margin: '0 0 5px 0', color: 'var(--text-muted)', fontSize: '0.85rem'}}>Física</p>
                  <p style={{margin: 0, fontSize: '1rem', fontWeight: 500}}>
                    {selectedExp.model_type} ({selectedExp.model_type.includes('burgers') ? `ν=${selectedExp.nu_viscosity ?? 0}` : `a=${selectedExp.a_velocity}`})
                  </p>
                </div>
                <div style={{flex: 1, minWidth: '200px'}}>
                  <p style={{margin: '0 0 5px 0', color: 'var(--text-muted)', fontSize: '0.85rem'}}>Rede</p>
                  <p style={{margin: 0, fontSize: '1rem', fontWeight: 500}}>
                    {selectedExp.n_layers} Camadas × {selectedExp.n_neurons} Neurônios
                  </p>
                </div>
                <div style={{flex: 1, minWidth: '200px'}}>
                  <p style={{margin: '0 0 5px 0', color: 'var(--text-muted)', fontSize: '0.85rem'}}>Diagnóstico (Fourier)</p>
                  <p style={{margin: 0, fontSize: '1rem', fontWeight: 500, color: 'var(--accent-hover)'}}>
                    {selectedExp.diagnostico}
                  </p>
                </div>
              </div>

              <div style={{marginBottom: '30px'}}>
                <p style={{margin: '0 0 5px 0', color: 'var(--text-muted)', fontSize: '0.85rem'}}>Anotações do Experimento</p>
                <div style={{display: 'flex', gap: '10px'}}>
                  <textarea 
                    value={modalComments} 
                    onChange={e => setModalComments(e.target.value)}
                    placeholder="Anote aqui suas observações sobre este experimento (ex: 'O choque se formou em t=0.5, testar com mais épocas L-BFGS')..."
                    style={{flex: 1, minHeight: '60px', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-main)', color: 'var(--text-primary)', fontFamily: 'Inter, sans-serif', resize: 'vertical'}}
                  />
                  <button onClick={saveComments} className="btn-primary" style={{alignSelf: 'flex-start', padding: '10px 15px'}}>Salvar</button>
                </div>
              </div>

              <div className="modal-grid">
                <div className="modal-image-card">
                  <h3>Dashboard Geral</h3>
                  <img src={`${API_URL}/results/${selectedExp.results_dir.split('/').pop()}/dashboard.png`} alt="Dashboard" />
                </div>
                <div className="modal-image-card">
                  <h3>Animação (Tempo Real)</h3>
                  <img src={`${API_URL}/results/${selectedExp.results_dir.split('/').pop()}/animation.gif`} alt="Animação GIF" />
                </div>
                <div className="modal-image-card">
                  <h3>Superfície 3D (Animada)</h3>
                  <img src={`${API_URL}/results/${selectedExp.results_dir.split('/').pop()}/animation_3d.gif`} alt="Superfície 3D Animada" />
                </div>
                <div className="modal-image-card">
                  <h3>Evolução da Solução (Slices)</h3>
                  <img src={`${API_URL}/results/${selectedExp.results_dir.split('/').pop()}/slices.png`} alt="Slices Temporais" />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
