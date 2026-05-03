from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List
import os

from . import models, schemas
from .database import engine, SessionLocal
from .worker import run_simulation

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="PINN Studio API")

# Mount results directory
results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
os.makedirs(results_path, exist_ok=True)
app.mount("/results", StaticFiles(directory=results_path), name="results")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/experiments", response_model=List[schemas.ExperimentResponse])
def get_experiments(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    experiments = db.query(models.Experiment).order_by(models.Experiment.id.desc()).offset(skip).limit(limit).all()
    return experiments

@app.get("/experiments/{experiment_id}", response_model=schemas.ExperimentResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    db_exp = db.query(models.Experiment).filter(models.Experiment.id == experiment_id).first()
    if db_exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return db_exp

@app.post("/experiments", response_model=schemas.ExperimentResponse)
def create_experiment(exp: schemas.ExperimentCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    db_exp = models.Experiment(**exp.model_dump())
    db.add(db_exp)
    db.commit()
    db.refresh(db_exp)
    
    # Run simulation in background
    background_tasks.add_task(run_simulation, db_exp.id)
    
    return db_exp

@app.delete("/experiments/{experiment_id}")
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    db_exp = db.query(models.Experiment).filter(models.Experiment.id == experiment_id).first()
    if not db_exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if db_exp.results_dir and os.path.exists(db_exp.results_dir):
        import shutil
        shutil.rmtree(db_exp.results_dir, ignore_errors=True)
        
    db.delete(db_exp)
    db.commit()
    return {"message": "Experiment deleted successfully"}

@app.put("/experiments/{experiment_id}", response_model=schemas.ExperimentResponse)
def update_experiment(experiment_id: int, update_data: schemas.ExperimentUpdate, db: Session = Depends(get_db)):
    db_exp = db.query(models.Experiment).filter(models.Experiment.id == experiment_id).first()
    if not db_exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if update_data.comments is not None:
        db_exp.comments = update_data.comments
        
    db.commit()
    db.refresh(db_exp)
    return db_exp
