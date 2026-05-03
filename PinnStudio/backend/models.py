from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from .database import Base
import datetime

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    model_type = Column(String, index=True) # e.g., advection_linear
    a_velocity = Column(Float, nullable=True)
    nu_viscosity = Column(Float, nullable=True)
    u0_name = Column(String)
    u0_params = Column(JSON)
    x_min = Column(Float, default=0.0)
    x_max = Column(Float, default=2.0)
    t_min = Column(Float, default=0.0)
    t_max = Column(Float, default=1.0)
    n_layers = Column(Integer)
    n_neurons = Column(Integer)
    activation = Column(String)
    lr_adam = Column(Float)
    epochs_adam = Column(Integer)
    epochs_lbfgs = Column(Integer)
    
    # Results
    status = Column(String, default="RUNNING") # RUNNING, COMPLETED, FAILED
    time_taken_sec = Column(Float, nullable=True) # Tempo total de treinamento
    loss_final = Column(Float, nullable=True)
    loss_f_final = Column(Float, nullable=True) # Resíduo da PDE
    loss_u_final = Column(Float, nullable=True) # Erro de fronteira/inicial
    l2_error = Column(Float, nullable=True) # Erro L2 Relativo (se solução exata existir)
    nu_numerical = Column(Float, nullable=True)
    diagnostico = Column(String, nullable=True)
    results_dir = Column(String, nullable=True)
    comments = Column(String, nullable=True)
