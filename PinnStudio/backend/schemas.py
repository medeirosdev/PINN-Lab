from pydantic import BaseModel
from typing import Optional, Dict, Any
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

class ExperimentResponse(ExperimentCreate):
    id: int
    timestamp: datetime
    status: str
    loss_final: Optional[float]
    nu_numerical: Optional[float]
    diagnostico: Optional[str] = None
    results_dir: Optional[str] = None
    comments: Optional[str] = None

    class Config:
        from_attributes = True

class ExperimentUpdate(BaseModel):
    comments: Optional[str] = None
