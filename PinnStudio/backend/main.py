from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
import os
import threading
import queue
import traceback

from . import models, schemas
from .database import engine, SessionLocal
from .worker import run_simulation

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="PINN Studio API")

# ── Fila de processamento ──────────────────────────────────────────────────────
experiment_queue = queue.Queue()

# Progresso em tempo real: { exp_id: { epoch, total_epochs, loss, phase } }
training_progress = {}


def queue_worker():
    while True:
        exp_id = experiment_queue.get()
        if exp_id is None:
            break
        try:
            training_progress[exp_id] = {"epoch": 0, "total_epochs": 0, "loss": None, "phase": "Iniciando"}
            print(f"[Queue Worker] Processando Experimento #{exp_id}...")
            run_simulation(exp_id, progress_dict=training_progress)
            print(f"[Queue Worker] Experimento #{exp_id} finalizado.")
        except Exception as e:
            print(f"[Queue Worker] Erro no Experimento #{exp_id}: {e}")
            traceback.print_exc()
        finally:
            training_progress.pop(exp_id, None)
            experiment_queue.task_done()


threading.Thread(target=queue_worker, daemon=True, name="PINN_Queue_Worker").start()

# ── Static files ───────────────────────────────────────────────────────────────
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


# ── Experiments ────────────────────────────────────────────────────────────────

@app.get("/progress")
def get_progress():
    return training_progress


@app.get("/experiments", response_model=List[schemas.ExperimentResponse])
def get_experiments(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.Experiment).order_by(models.Experiment.id.desc()).offset(skip).limit(limit).all()


@app.get("/experiments/{experiment_id}", response_model=schemas.ExperimentResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    db_exp = db.query(models.Experiment).filter(models.Experiment.id == experiment_id).first()
    if not db_exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return db_exp


@app.post("/experiments", response_model=schemas.ExperimentResponse)
def create_experiment(exp: schemas.ExperimentCreate, db: Session = Depends(get_db)):
    db_exp = models.Experiment(**exp.model_dump())
    db_exp.status = "QUEUED"
    db.add(db_exp)
    db.commit()
    db.refresh(db_exp)
    experiment_queue.put(db_exp.id)
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


@app.get("/compare/loss")
def compare_losses(ids: str, db: Session = Depends(get_db)):
    """PNG com curvas de loss sobrepostas para os experimentos informados (ids=1,2,3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import tempfile

    exp_ids = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    if not exp_ids:
        raise HTTPException(status_code=400, detail="Nenhum ID válido fornecido.")

    experiments = db.query(models.Experiment).filter(models.Experiment.id.in_(exp_ids)).all()
    if not experiments:
        raise HTTPException(status_code=404, detail="Experimentos não encontrados.")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']

    for i, exp in enumerate(experiments):
        if not exp.results_dir:
            continue
        history_path = os.path.join(exp.results_dir, "history.npy")
        if not os.path.exists(history_path):
            continue
        history = np.load(history_path, allow_pickle=True).item()
        loss = history.get("loss", [])
        c = colors[i % len(colors)]
        ax.plot(loss, label=f"Exp {exp.id} (seed={exp.seed})", color=c, alpha=0.8, linewidth=2)

    ax.set_yscale("log")
    ax.set_xlabel("Épocas", fontsize=12)
    ax.set_ylabel("Loss Total (MSE)", fontsize=12)
    ax.set_title("Comparação de Convergência (Loss)", fontsize=14, pad=15)
    ax.grid(True, which="both", ls="--", alpha=0.2)
    ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#ffffff')
    plt.close(fig)
    return FileResponse(path, media_type="image/png", background=None)


# ── Batches ────────────────────────────────────────────────────────────────────

def _batch_stats(batch_id: int, db: Session) -> dict:
    """Computa estatísticas agregadas do batch a partir dos experimentos filhos."""
    import numpy as np

    exps = db.query(models.Experiment).filter(models.Experiment.batch_id == batch_id).all()
    completed = [e for e in exps if e.status == "COMPLETED"]

    def agg(field):
        vals = [getattr(e, field) for e in completed if getattr(e, field) is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    l2_mean, l2_std     = agg("l2_error")
    linf_mean, linf_std = agg("linf_error")
    loss_mean, loss_std = agg("loss_final")

    # Marca batch como COMPLETED quando todos os filhos terminaram
    if exps and len(completed) == len(exps):
        batch = db.query(models.BatchExperiment).filter(models.BatchExperiment.id == batch_id).first()
        if batch and batch.status != "COMPLETED":
            batch.status = "COMPLETED"
            db.commit()

    return {
        "completed_runs": len(completed),
        "experiment_ids": [e.id for e in exps],
        "has_exact_solution": any(e.has_exact_solution for e in completed),
        "l2_mean": l2_mean, "l2_std": l2_std,
        "linf_mean": linf_mean, "linf_std": linf_std,
        "loss_mean": loss_mean, "loss_std": loss_std,
    }


def _batch_to_dict(batch: models.BatchExperiment) -> dict:
    return {c.name: getattr(batch, c.name) for c in batch.__table__.columns}


@app.get("/batches", response_model=List[schemas.BatchResponse])
def get_batches(db: Session = Depends(get_db)):
    batches = db.query(models.BatchExperiment).order_by(models.BatchExperiment.id.desc()).all()
    return [{**_batch_to_dict(b), **_batch_stats(b.id, db)} for b in batches]


@app.get("/batches/{batch_id}", response_model=schemas.BatchResponse)
def get_batch(batch_id: int, db: Session = Depends(get_db)):
    b = db.query(models.BatchExperiment).filter(models.BatchExperiment.id == batch_id).first()
    if not b:
        raise HTTPException(status_code=404, detail="Batch not found")
    return {**_batch_to_dict(b), **_batch_stats(b.id, db)}


@app.post("/batches", response_model=schemas.BatchResponse)
def create_batch(batch: schemas.BatchCreate, db: Session = Depends(get_db)):
    config = batch.model_dump(exclude={"n_seeds"})
    db_batch = models.BatchExperiment(n_seeds=batch.n_seeds, **config)
    db.add(db_batch)
    db.commit()
    db.refresh(db_batch)

    for seed in range(batch.n_seeds):
        exp = models.Experiment(**config, seed=seed, batch_id=db_batch.id, status="QUEUED")
        db.add(exp)
        db.commit()
        db.refresh(exp)
        experiment_queue.put(exp.id)

    return {**_batch_to_dict(db_batch), **_batch_stats(db_batch.id, db)}


@app.delete("/batches/{batch_id}")
def delete_batch(batch_id: int, db: Session = Depends(get_db)):
    b = db.query(models.BatchExperiment).filter(models.BatchExperiment.id == batch_id).first()
    if not b:
        raise HTTPException(status_code=404, detail="Batch not found")

    child_exps = db.query(models.Experiment).filter(models.Experiment.batch_id == batch_id).all()
    for exp in child_exps:
        if exp.results_dir and os.path.exists(exp.results_dir):
            import shutil
            shutil.rmtree(exp.results_dir, ignore_errors=True)
        db.delete(exp)

    db.delete(b)
    db.commit()
    return {"message": f"Batch {batch_id} e {len(child_exps)} experimentos deletados."}


@app.get("/batches/{batch_id}/loss_chart")
def batch_loss_chart(batch_id: int, db: Session = Depends(get_db)):
    """PNG com curvas individuais (leves) + banda média±σ para um batch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import tempfile

    exps = db.query(models.Experiment).filter(
        models.Experiment.batch_id == batch_id,
        models.Experiment.status == "COMPLETED",
    ).all()

    if not exps:
        raise HTTPException(status_code=404, detail="Nenhum experimento concluído neste batch.")

    histories = []
    for exp in exps:
        if not exp.results_dir:
            continue
        path = os.path.join(exp.results_dir, "history.npy")
        if os.path.exists(path):
            h = np.load(path, allow_pickle=True).item()
            if h.get("loss"):
                histories.append(np.array(h["loss"]))

    if not histories:
        raise HTTPException(status_code=404, detail="Histórico de loss não encontrado.")

    # Trunca todas ao comprimento mínimo para poder empilhar
    min_len = min(len(h) for h in histories)
    mat = np.stack([h[:min_len] for h in histories], axis=0)  # (N_seeds, epochs)

    # Trabalha em log para que a banda seja simétrica na escala log do gráfico
    log_mat = np.log10(np.maximum(mat, 1e-16))
    log_mean = log_mat.mean(axis=0)
    log_std  = log_mat.std(axis=0, ddof=1) if len(histories) > 1 else np.zeros_like(log_mean)
    epochs = np.arange(min_len)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_yscale("log")

    # Curvas individuais (transparentes)
    for i, h in enumerate(histories):
        ax.plot(h[:min_len], color="#3b82f6", alpha=0.25, linewidth=1.2,
                label="Runs individuais" if i == 0 else None)

    # Banda ±1σ em log — simétrica na escala logarítmica
    lower = 10 ** (log_mean - log_std)
    upper = 10 ** (log_mean + log_std)
    ax.fill_between(epochs, lower, upper, alpha=0.25, color="#3b82f6", label="Média ± 1σ (log)")

    # Curva média
    ax.plot(epochs, 10 ** log_mean, color="#1d4ed8", linewidth=2.5, label="Média geométrica")

    batch = db.query(models.BatchExperiment).filter(models.BatchExperiment.id == batch_id).first()
    cfg_label = f"{batch.model_type} | {batch.n_layers}×{batch.n_neurons} | {len(histories)} seeds" if batch else ""

    ax.set_xlabel("Épocas (Adam)", fontsize=12)
    ax.set_ylabel("Loss Total (MSE)", fontsize=12)
    ax.set_title(f"Batch #{batch_id} — Convergência com Intervalo de Confiança\n{cfg_label}",
                 fontsize=13, pad=12)
    ax.grid(True, which="both", ls="--", alpha=0.2)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    return FileResponse(path, media_type="image/png", background=None)


@app.on_event("shutdown")
def shutdown_event():
    experiment_queue.put(None)
