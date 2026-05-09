from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from .database import Base
import datetime


class BatchExperiment(Base):
    __tablename__ = "batch_experiments"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    n_seeds = Column(Integer)
    status = Column(String, default="RUNNING")  # RUNNING, COMPLETED

    # Configuração espelhada dos experimentos filhos (para exibição)
    model_type = Column(String, index=True)
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
    n_collocation = Column(Integer, default=8000)


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    model_type = Column(String, index=True)
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
    n_collocation = Column(Integer, default=8000)
    seed = Column(Integer, default=42)
    batch_id = Column(Integer, nullable=True)

    # Results
    status = Column(String, default="QUEUED")  # QUEUED, RUNNING, COMPLETED, FAILED
    time_taken_sec = Column(Float, nullable=True)
    loss_final = Column(Float, nullable=True)
    loss_f_final = Column(Float, nullable=True)
    loss_u_final = Column(Float, nullable=True)
    l2_error = Column(Float, nullable=True)
    linf_error = Column(Float, nullable=True)
    has_exact_solution = Column(Boolean, default=False)
    nu_numerical = Column(Float, nullable=True)
    mu_numerical = Column(Float, nullable=True)
    diagnostico = Column(String, nullable=True)
    results_dir = Column(String, nullable=True)
    comments = Column(String, nullable=True)
