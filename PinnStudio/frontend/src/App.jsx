import React, { useState, useEffect, useMemo } from 'react';
import { Activity, Play, RefreshCw, BarChart2, Layers, Trash2, Home, BookOpen, MonitorPlay, Save, PieChart } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ScatterChart, Scatter, ZAxis, ResponsiveContainer, LineChart, Line } from 'recharts';
import './index.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [currentRoute, setCurrentRoute] = useState('home');
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedExp, setSelectedExp] = useState(null);
  const [modalComments, setModalComments] = useState('');
  
  // Comparação
  const [selectedForCompare, setSelectedForCompare] = useState([]);

  const handleSelectCompare = (id, checked) => {
    if (checked) {
      if (selectedForCompare.length < 3) {
        setSelectedForCompare([...selectedForCompare, id]);
      } else {
        alert("Você pode comparar no máximo 3 experimentos simultaneamente.");
      }
    } else {
      setSelectedForCompare(selectedForCompare.filter(expId => expId !== id));
    }
  };

  const selectedExperimentsData = experiments.filter(exp => selectedForCompare.includes(exp.id));
  
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

  const [deleteConfirm, setDeleteConfirm] = useState(null);

  const deleteExperiment = async (id) => {
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
          <span className={`nav-link ${currentRoute === 'analytics' ? 'active' : ''}`} onClick={() => setCurrentRoute('analytics')}>Dados Gerais</span>
          <span className={`nav-link ${currentRoute === 'manual' ? 'active' : ''}`} onClick={() => setCurrentRoute('manual')}>Manual</span>
        </div>
      </nav>

      {currentRoute === 'home' && (
        <div className="home-container">
          <div className="hero-content">
            <div className="hero-text">
              <h1>Simulações Físicas com Redes Neurais</h1>
              <p>Plataforma interativa para explorar, treinar e diagnosticar modelos Physics-Informed Neural Networks (PINNs) utilizando equações de Advecção e Burgers.</p>
            </div>
            <div className="home-grid">
              <div className="home-card" onClick={() => setCurrentRoute('studio')}>
                <div className="icon-wrapper"><MonitorPlay size={26} /></div>
                <div>
                  <h3>PINN Lab</h3>
                  <p>Ambiente completo de simulação em tempo real.</p>
                </div>
              </div>
              <div className="home-card" onClick={() => setCurrentRoute('experiments')}>
                <div className="icon-wrapper"><Layers size={26} /></div>
                <div>
                  <h3>Meus Experimentos</h3>
                  <p>Histórico completo e Dashboard 3D.</p>
                </div>
              </div>
              <div className="home-card" onClick={() => setCurrentRoute('analytics')}>
                <div className="icon-wrapper"><PieChart size={26} /></div>
                <div>
                  <h3>Dados Gerais (Analytics)</h3>
                  <p>Métricas globais de todos os modelos e arquiteturas.</p>
                </div>
              </div>
              <div className="home-card" onClick={() => setCurrentRoute('manual')}>
                <div className="icon-wrapper"><BookOpen size={26} /></div>
                <div>
                  <h3>Manual & Teoria</h3>
                  <p>Equações, funções e diagnóstico de Fourier.</p>
                </div>
              </div>
            </div>
          </div>
          <div className="hero-image-wrapper">
            <img src="/hero-image.png" alt="Physics-Informed Neural Network" className="hero-image" />
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

            <h3 style={{ color: 'var(--accent-color)', marginBottom: '15px', marginTop: '30px' }}>3. Catálogo Completo de Condições Iniciais <span style={{fontFamily:'serif'}}>u(x, t=0)</span></h3>
            <p style={{ marginBottom: '20px', lineHeight: '1.6', color: 'var(--text-muted)' }}>O PINN Lab suporta 24 condições iniciais pré-programadas para avaliar a capacidade da rede de capturar diferentes fenômenos físicos. Abaixo, as expressões matemáticas originais de cada uma:</p>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
              <div style={{ backgroundColor: 'rgba(255,255,255,0.03)', padding: '15px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                <h4 style={{color:'#fff', marginBottom:'10px', fontSize:'1rem'}}>Família Suave (Advecção)</h4>
                <ul style={{ color:'var(--text-muted)', fontSize:'0.9rem', paddingLeft:'20px', display:'flex', flexDirection:'column', gap:'8px' }}>
                  <li><strong>Gaussiana:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = A · exp( -(x-c)² / 2σ² )</code></li>
                  <li><strong>Seno:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = A · sin(k·π·x)</code></li>
                  <li><strong>Cosseno:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = A · cos(k·π·x)</code></li>
                  <li><strong>Pacote de Ondas:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = exp(-x² / 2σ²) · sin(k·π·x)</code></li>
                  <li><strong>Bump Suave:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = exp(1 - 1/(1-x²))</code> para |x|&lt;1, senão 0</li>
                </ul>
              </div>

              <div style={{ backgroundColor: 'rgba(255,255,255,0.03)', padding: '15px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                <h4 style={{color:'#fff', marginBottom:'10px', fontSize:'1rem'}}>Família Descontínua (Choques)</h4>
                <ul style={{ color:'var(--text-muted)', fontSize:'0.9rem', paddingLeft:'20px', display:'flex', flexDirection:'column', gap:'8px' }}>
                  <li><strong>Heaviside (Degrau):</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = 1 (x ≤ 0); 0 (x &gt; 0)</code></li>
                  <li><strong>Choque de Riemann:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = 1.0 (x ≤ 0); 0.2 (x &gt; 0)</code></li>
                  <li><strong>Rarefação de Riemann:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = 0.2 (x ≤ 0); 1.0 (x &gt; 0)</code></li>
                  <li><strong>Degrau Duplo:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = 1 (x&lt;-0.5); 0.5 (|x|≤0.5); 0 (x&gt;0.5)</code></li>
                  <li><strong>Três Estados:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = 1 (x&lt;-1); 0 (|x|≤1); -1 (x&gt;1)</code></li>
                </ul>
              </div>

              <div style={{ backgroundColor: 'rgba(255,255,255,0.03)', padding: '15px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                <h4 style={{color:'#fff', marginBottom:'10px', fontSize:'1rem'}}>Família Com Cantos (Difusão Numérica)</h4>
                <ul style={{ color:'var(--text-muted)', fontSize:'0.9rem', paddingLeft:'20px', display:'flex', flexDirection:'column', gap:'8px' }}>
                  <li><strong>Dente de Serra:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = x - ⌊x⌋</code></li>
                  <li><strong>Onda Quadrada:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = sgn(sin(k·π·x))</code></li>
                  <li><strong>Chapéu (Triângulo):</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = max(0, 1 - |x|)</code></li>
                  <li><strong>Trapézio:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = 1 se |x|&lt;0.5; transição linear; 0 se |x|&gt;1</code></li>
                </ul>
              </div>

              <div style={{ backgroundColor: 'rgba(255,255,255,0.03)', padding: '15px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                <h4 style={{color:'#fff', marginBottom:'10px', fontSize:'1rem'}}>Família Composta / Exótica</h4>
                <ul style={{ color:'var(--text-muted)', fontSize:'0.9rem', paddingLeft:'20px', display:'flex', flexDirection:'column', gap:'8px' }}>
                  <li><strong>Soliton (KdV):</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = A · sech²( √(A/2) · x )</code></li>
                  <li><strong>Onda-N (Burgers):</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = x/(1+t) · exp(-x²/4ν(1+t))</code></li>
                  <li><strong>Wavelet Ricker:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = (1 - x²/σ²) · exp(-x²/2σ²)</code></li>
                  <li><strong>Multi-Bump:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = Σ exp(-(x-c_i)² / 2σ²)</code></li>
                  <li><strong>Chirp:</strong> <code style={{color:'var(--accent-hover)'}}>u(x) = sin( k · π · x² )</code> (Frequência var.)</li>
                </ul>
              </div>
            </div>

            <h3 style={{ color: 'var(--accent-color)', marginBottom: '15px', marginTop: '30px' }}>4. Diagnóstico Físico via Equação Modificada</h3>
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
                  {currentRoute === 'experiments' && exp.status === 'COMPLETED' && (
                    <input 
                      type="checkbox" 
                      title="Selecionar para comparação"
                      checked={selectedForCompare.includes(exp.id)}
                      onChange={(e) => handleSelectCompare(exp.id, e.target.checked)}
                      style={{ cursor: 'pointer', width: '18px', height: '18px', accentColor: 'var(--accent-color)' }}
                    />
                  )}
                  <button onClick={(e) => { e.stopPropagation(); setDeleteConfirm(exp.id); }} className="btn-close" style={{ padding: '4px', color: 'var(--danger-color)' }} title="Deletar Experimento">
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
        
        {currentRoute === 'experiments' && selectedForCompare.length >= 2 && (
          <button 
            onClick={() => setCurrentRoute('compare')}
            style={{
              position: 'fixed', bottom: '40px', right: '40px',
              padding: '15px 30px', fontSize: '1.1rem', fontWeight: 600,
              backgroundColor: 'var(--accent-color)', color: '#fff',
              border: 'none', borderRadius: '30px', cursor: 'pointer',
              boxShadow: '0 10px 25px rgba(59, 130, 246, 0.4)',
              display: 'flex', alignItems: 'center', gap: '10px',
              animation: 'slideUp 0.3s ease-out'
            }}
          >
            <BarChart2 size={24} />
            Comparar Selecionados ({selectedForCompare.length})
          </button>
        )}
      </div>
    </div>
  )}

      {currentRoute === 'analytics' && (() => {
        const completedExp = experiments.filter(e => e.status === 'COMPLETED');
        if(completedExp.length === 0) return <div style={{padding:'40px',color:'#fff'}}>Nenhum experimento concluído ainda.</div>;

        const totalTime = completedExp.reduce((s, e) => s + (e.time_taken_sec || 0), 0);
        
        const archStats = completedExp.reduce((acc, exp) => {
          const arch = `${exp.n_layers}x${exp.n_neurons}`;
          if (!acc[arch]) acc[arch] = { name: arch, count: 0, sum: 0 };
          acc[arch].count++;
          acc[arch].sum += exp.loss_final || 0;
          return acc;
        }, {});
        
        const archData = Object.values(archStats).map(d => ({ name: d.name, avgLoss: d.sum / d.count }));
        const bestArch = archData.length > 0 ? archData.reduce((prev, curr) => prev.avgLoss < curr.avgLoss ? prev : curr) : null;

        const top5 = [...completedExp].sort((a,b) => (a.loss_final||99) - (b.loss_final||99)).slice(0,5);

        return (
          <div style={{ padding: '40px', overflowY: 'auto', flex: 1, color: 'var(--text-primary)', maxWidth: '1400px', margin: '0 auto', width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
              <h1 style={{ color: '#fff' }}><PieChart style={{display:'inline', marginRight:'10px'}}/> Dados Gerais (Analytics)</h1>
            </div>

            {/* KPIs */}
            <div className="kpi-grid">
              <div className="kpi-card">
                <h4>Total de Treinamentos</h4>
                <div className="kpi-value">{completedExp.length}</div>
                <div className="kpi-subtext">Experimentos concluídos</div>
              </div>
              <div className="kpi-card">
                <h4>Tempo Computacional Total</h4>
                <div className="kpi-value">{(totalTime / 60).toFixed(1)} <span style={{fontSize:'1rem'}}>min</span></div>
                <div className="kpi-subtext">{(totalTime / 3600).toFixed(2)} horas</div>
              </div>
              <div className="kpi-card">
                <h4>Melhor Arquitetura Média</h4>
                <div className="kpi-value" style={{color: 'var(--accent-color)'}}>{bestArch ? bestArch.name : '-'}</div>
                <div className="kpi-subtext">Loss: {bestArch ? bestArch.avgLoss.toExponential(2) : '-'}</div>
              </div>
              <div className="kpi-card">
                <h4>Menor Erro Atingido</h4>
                <div className="kpi-value" style={{color: '#10b981'}}>{top5.length > 0 ? top5[0].loss_final.toExponential(2) : '-'}</div>
                <div className="kpi-subtext">Pelo Experimento #{top5.length > 0 ? top5[0].id : '-'}</div>
              </div>
            </div>

            <div className="analytics-grid">
              {/* Gráfico 1: Loss Média por Arquitetura */}
              <div className="card" style={{ padding: '25px', height: '350px' }}>
                <h3 style={{ marginBottom: '20px', color: 'var(--accent-color)' }}>Loss Média por Arquitetura (Camadas x Neurônios)</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={archData} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="name" stroke="var(--text-muted)" angle={-45} textAnchor="end" />
                    <YAxis scale="log" domain={['auto', 'auto']} stroke="var(--text-muted)" tickFormatter={(v) => v.toExponential(1)} />
                    <RechartsTooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{backgroundColor: 'var(--bg-panel)', border: '1px solid var(--border-color)', borderRadius: '8px'}} formatter={(val) => val.toExponential(4)} />
                    <Bar dataKey="avgLoss" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Loss Média" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Gráfico 2: Tempo vs Épocas (Dispersão) */}
              <div className="card" style={{ padding: '25px', height: '350px' }}>
                <h3 style={{ marginBottom: '20px', color: '#10b981' }}>Custo Computacional: Tempo (s) vs Épocas Totais</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis type="number" dataKey="epochs" name="Épocas" stroke="var(--text-muted)" />
                    <YAxis type="number" dataKey="time" name="Tempo (s)" stroke="var(--text-muted)" />
                    <ZAxis type="category" dataKey="model" name="Física" />
                    <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{backgroundColor: 'var(--bg-panel)', border: '1px solid var(--border-color)', borderRadius: '8px'}} />
                    <Scatter name="Experimentos" data={completedExp.map(exp => ({
                      id: exp.id, epochs: (exp.epochs_adam||0) + (exp.epochs_lbfgs||0), time: exp.time_taken_sec || 0, model: exp.model_type
                    }))} fill="#10b981" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>

              {/* Tabela: Top 5 Experimentos */}
              <div className="card analytics-full" style={{ padding: '25px' }}>
                <h3 style={{ marginBottom: '20px', color: '#f59e0b' }}>Top 5 Experimentos (Menor Loss Final)</h3>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                        <th style={{ padding: '10px' }}>ID</th>
                        <th style={{ padding: '10px' }}>Física</th>
                        <th style={{ padding: '10px' }}>Condição Inicial</th>
                        <th style={{ padding: '10px' }}>Rede</th>
                        <th style={{ padding: '10px' }}>Loss Final</th>
                        <th style={{ padding: '10px' }}>Erro L2</th>
                        <th style={{ padding: '10px' }}>Difusão Num.</th>
                        <th style={{ padding: '10px' }}>Tempo</th>
                      </tr>
                    </thead>
                    <tbody>
                      {top5.map(exp => (
                        <tr key={exp.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '10px', color: '#fff', fontWeight: 'bold' }}>#{exp.id}</td>
                          <td style={{ padding: '10px' }}>{exp.model_type}</td>
                          <td style={{ padding: '10px' }}>{exp.u0_name}</td>
                          <td style={{ padding: '10px' }}>{exp.n_layers}x{exp.n_neurons}</td>
                          <td style={{ padding: '10px', color: '#ef4444' }}>{exp.loss_final?.toExponential(4)}</td>
                          <td style={{ padding: '10px', color: '#10b981' }}>{exp.l2_error ? exp.l2_error.toExponential(4) : '-'}</td>
                          <td style={{ padding: '10px', color: 'var(--accent-hover)' }}>{exp.nu_numerical ? exp.nu_numerical.toExponential(4) : '-'}</td>
                          <td style={{ padding: '10px' }}>{exp.time_taken_sec ? exp.time_taken_sec.toFixed(1) + 's' : '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        );
      })()}

      {currentRoute === 'compare' && (
        <div style={{ padding: '40px', overflowY: 'auto', flex: 1, color: 'var(--text-primary)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
            <h1 style={{ color: '#fff' }}><BarChart2 style={{display:'inline', marginRight:'10px'}}/> Comparação de Experimentos</h1>
            <button className="btn-close" onClick={() => setCurrentRoute('experiments')}>Voltar para Experimentos</button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${selectedExperimentsData.length}, 1fr)`, gap: '20px', marginBottom: '30px' }}>
            {selectedExperimentsData.map(exp => (
              <div key={exp.id} className="card" style={{ padding: '20px' }}>
                <h3 style={{ color: 'var(--accent-color)', marginBottom: '15px' }}>Experimento #{exp.id}</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '0.9rem', marginBottom: '15px' }}>
                  <p style={{color: 'var(--text-muted)'}}>Física: <strong style={{color:'#fff'}}>{exp.model_type}</strong></p>
                  <p style={{color: 'var(--text-muted)'}}>u0: <strong style={{color:'#fff'}}>{exp.u0_name}</strong></p>
                  <p style={{color: 'var(--text-muted)'}}>Rede: <strong style={{color:'#fff'}}>{exp.n_layers}x{exp.n_neurons}</strong></p>
                  <p style={{color: 'var(--text-muted)'}}>Épocas: <strong style={{color:'#fff'}}>{exp.epochs_adam} + {exp.epochs_lbfgs}</strong></p>
                </div>
                <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '15px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <p style={{ display: 'flex', justifyContent: 'space-between' }}><span>Loss Total:</span> <strong style={{color: '#ef4444'}}>{exp.loss_final?.toExponential(4) || 'N/A'}</strong></p>
                  <p style={{ display: 'flex', justifyContent: 'space-between' }}><span>Erro L2:</span> <strong style={{color: '#10b981'}}>{exp.l2_error ? exp.l2_error.toExponential(4) : 'N/A'}</strong></p>
                  <p style={{ display: 'flex', justifyContent: 'space-between' }}><span>Tempo (s):</span> <strong>{exp.time_taken_sec ? exp.time_taken_sec.toFixed(1) : 'N/A'}</strong></p>
                </div>
                {exp.results_dir && (
                  <div style={{ marginTop: '20px', borderRadius: '8px', overflow: 'hidden' }}>
                    <p style={{textAlign:'center', fontSize:'0.85rem', marginBottom:'5px', color:'var(--text-muted)'}}>Superfície 3D</p>
                    <img src={`${API_URL}/results/${exp.results_dir.split('/').pop()}/animation_3d.gif`} alt="3D" style={{ width: '100%', borderRadius: '8px' }} />
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="card" style={{ padding: '30px' }}>
            <h3 style={{ marginBottom: '20px' }}>Curvas de Convergência (Loss History)</h3>
            <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
              <img 
                src={`${API_URL}/compare/loss?ids=${selectedForCompare.join(',')}`} 
                alt="Loss Comparison Chart" 
                style={{ width: '100%', maxWidth: '900px', borderRadius: '12px', border: '1px solid var(--border-color)' }}
              />
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

      {/* Modal Confirmar Exclusão */}
      {deleteConfirm && (
        <div className="modal-overlay" onClick={() => setDeleteConfirm(null)}>
          <div className="modal-content" style={{maxWidth: '400px', height: 'auto', padding: '25px', display: 'block'}} onClick={e => e.stopPropagation()}>
            <h2 style={{marginTop: 0, color: 'var(--danger-color)'}}>Confirmar Exclusão</h2>
            <p style={{margin: '20px 0', color: 'var(--text-primary)', lineHeight: '1.5'}}>
              Tem certeza que deseja deletar o experimento <strong>#{deleteConfirm}</strong>? Essa ação removerá a pasta de resultados permanentemente.
            </p>
            <div style={{display: 'flex', gap: '15px', justifyContent: 'flex-end', marginTop: '25px'}}>
              <button className="btn-close" style={{padding: '10px 20px'}} onClick={() => setDeleteConfirm(null)}>Cancelar</button>
              <button className="btn-primary" style={{backgroundColor: 'var(--danger-color)', padding: '10px 20px'}} onClick={() => {
                deleteExperiment(deleteConfirm);
                setDeleteConfirm(null);
              }}>Deletar</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
