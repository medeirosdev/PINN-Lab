from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime


class ExperimentCreate(BaseModel):
    model_type: str
    a_velocity: Optional[float] = 1.0
    nu_viscosity: Optional[float] = None
    u0_name: str = "gaussian"
    u0_params: Dict[str, Any] = {"center": 0.5, "sigma": 0.2, "amplitude": 1.0}
    x_min: float = 0.0
    x_max: float = 2.0
    t_min: float = 0.0
    t_max: float = 1.0
    n_layers: int = 4
    n_neurons: int = 40
    activation: str = "tanh"
    lr_adam: float = 1e-3
    epochs_adam: int = 1000
    epochs_lbfgs: int = 0
    n_collocation: int = 8000
    seed: int = 42


class ExperimentResponse(ExperimentCreate):
    id: int
    timestamp: datetime
    status: str
    batch_id: Optional[int] = None
    time_taken_sec: Optional[float] = None
    loss_final: Optional[float] = None
    loss_f_final: Optional[float] = None
    loss_u_final: Optional[float] = None
    l2_error: Optional[float] = None
    linf_error: Optional[float] = None
    has_exact_solution: bool = False
    nu_numerical: Optional[float] = None
    mu_numerical: Optional[float] = None
    diagnostico: Optional[str] = None
    results_dir: Optional[str] = None
    comments: Optional[str] = None

    class Config:
        from_attributes = True


class ExperimentUpdate(BaseModel):
    comments: Optional[str] = None


# ── Batch ──────────────────────────────────────────────────────────────────────

class BatchCreate(BaseModel):
    model_type: str
    a_velocity: Optional[float] = 1.0
    nu_viscosity: Optional[float] = None
    u0_name: str = "gaussian"
    u0_params: Dict[str, Any] = {"center": 0.5, "sigma": 0.2, "amplitude": 1.0}
    x_min: float = 0.0
    x_max: float = 2.0
    t_min: float = 0.0
    t_max: float = 1.0
    n_layers: int = 4
    n_neurons: int = 40
    activation: str = "tanh"
    lr_adam: float = 1e-3
    epochs_adam: int = 1000
    epochs_lbfgs: int = 0
    n_collocation: int = 8000
    n_seeds: int = 5


class BatchResponse(BaseModel):
    id: int
    timestamp: datetime
    n_seeds: int
    status: str
    model_type: str
    a_velocity: Optional[float] = None
    nu_viscosity: Optional[float] = None
    u0_name: str
    u0_params: Dict[str, Any]
    x_min: float
    x_max: float
    t_min: float
    t_max: float
    n_layers: int
    n_neurons: int
    activation: str
    lr_adam: float
    epochs_adam: int
    epochs_lbfgs: int
    n_collocation: int = 8000
    # Campos computados dinamicamente no endpoint
    completed_runs: int = 0
    experiment_ids: List[int] = []
    has_exact_solution: bool = False
    l2_mean: Optional[float] = None
    l2_std: Optional[float] = None
    linf_mean: Optional[float] = None
    linf_std: Optional[float] = None
    loss_mean: Optional[float] = None
    loss_std: Optional[float] = None

    class Config:
        from_attributes = True
