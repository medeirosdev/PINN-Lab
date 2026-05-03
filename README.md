# PINN Lab
Interactive Physics-Informed Neural Networks Studio

PINN Lab is a full-stack research platform designed to train, evaluate, and diagnose Physics-Informed Neural Networks (PINNs) in real-time. It provides a visual and interactive environment to observe how neural networks solve Partial Differential Equations (PDEs), offering deep insights into phenomena such as numerical diffusion and dispersion through Modified Equation Analysis.

## Core Features

- PDE Solvers: Supports 1D Linear Advection and Burgers equations (both Viscous and Inviscid).
- Comprehensive Initial Conditions: Over 24 distinct initial conditions to stress-test neural PDE solvers, categorized into Smooth, Discontinuous, Sharp Corners, and Exotic structures (e.g., KdV Solitons, N-Waves, Chirps).
- Modified Equation Diagnosis: Performs discrete Fourier analysis on the residual to calculate and distinguish artificial numerical diffusion from dispersion injected by the network architecture.
- Advanced Visualizations: Real-time generation of 3D rotating solution surfaces, temporal slices, training dashboard metrics, and convergence animations.
- Hybrid Optimization: Seamlessly orchestrates training using Adam followed by L-BFGS for high-precision scientific machine learning.
- Experiment Comparison: Select up to 3 experiments to overlay loss trajectories on a unified logarithmic scale, comparing L2 relative error, PDE loss, and total computation time.

## Architecture

The project follows a decoupled client-server architecture:
- Core Engine: Python, PyTorch, NumPy, Matplotlib.
- Backend API: FastAPI, SQLAlchemy, SQLite (handles asynchronous task queuing and historical experiment persistence).
- Frontend UI: React, Vite, CSS Grid (features a modern dark-mode aesthetic with side-by-side comparison modules).

## Installation

### Backend Setup

1. Navigate to the project root and create a virtual environment:
   `python -m venv venv`
   `source venv/bin/activate`

2. Install the required dependencies:
   `pip install torch numpy matplotlib fastapi uvicorn sqlalchemy pydantic`

3. Start the backend server:
   `cd PinnStudio`
   `uvicorn backend.main:app --reload`

### Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
   `cd PinnStudio/frontend`

2. Install Node.js dependencies:
   `npm install`

3. Start the development server:
   `npm run dev`

Access the interface at `http://localhost:5173`.

## Usage

1. Open the "PINN Lab" tab to configure a new experiment.
2. Select the physical model, initial condition, domain bounds, and neural network depth/width.
3. Define the epochs for Adam and L-BFGS optimizers.
4. Click "Run Simulation". The worker will train the model asynchronously.
5. Navigate to "Experiments" to view the historical log. Select experiments using the checkbox to compare their 3D surfaces and convergence rates.

## License

This project is intended for research and educational purposes in Scientific Machine Learning.
