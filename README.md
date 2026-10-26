# 2D XY Model Simulator

An interactive simulation of a two-dimensional XY spin system using **Metropolis Monte Carlo** updates. A Flask application exposes the simulation state, and a JavaScript Canvas interface displays spin directions and the model's paired vortex markers.

**Stack:** Python · Flask · NumPy · SciPy · Matplotlib · JavaScript · HTML/CSS

**Team:** Anna Oleszko, Bartosz Kawalec, and Kacper Saj.

**Acknowledgment:** The simulation builds on [Shiling42/XY-MODEL](https://github.com/Shiling42/XY-MODEL), as credited in this repository's description.

## Features

- Simulate planar spins on a square lattice with periodic boundary conditions.
- Adjust temperature, lattice width, delay between updates, and sweeps per update.
- Start and stop continuous Monte Carlo evolution.
- Inspect energy per spin and request an equilibration run.
- Display paired vortex and antivortex markers.
- Use the `XYSystem` class for plotting and annealing experiments in Python.

The browser interface is in Polish.

## Model

Each spin is represented by an angle. Nearest-neighbor interactions use the reduced-unit Hamiltonian:

```text
H = -sum over neighboring pairs (i,j) of cos(theta_i - theta_j)
```

One sweep attempts an update for every spin in random order. A proposed angular change is accepted according to the Metropolis criterion, using temperature in units where the interaction strength and Boltzmann constant are set to one. The `get_energy()` method returns energy per spin.

The browser's time-step control sets the **delay between Monte Carlo updates**. It does not represent a physical integration time step.

## Quick start

Install Python 3 and create a virtual environment:

```bash
git clone https://github.com/kapitanek23/XY-Model.git
cd XY-Model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install Flask numpy scipy matplotlib
python app.py
```

On Windows PowerShell, create the environment with `python -m venv .venv` and activate with `.\.venv\Scripts\Activate.ps1`.

Open `http://127.0.0.1:5000`. The repository currently has no pinned dependency file. SciPy and Matplotlib are imported by `XY_model.py`, including when starting the web application.

### First experiment

1. Keep the default **10 × 10** lattice and temperature **2.5**.
2. Click **Start** to evolve the configuration.
3. Change the temperature to observe how the spin configuration responds.
4. Click **Stop** before changing the lattice width or requesting **Równoważenie**.
5. Keep temperature **greater than zero** when equilibrating.

Runs are stochastic. Repeating the same controls does not reproduce the same configuration because the application does not set random seeds.

## Local API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | Browser interface |
| `GET` | `/data` | Spins, width, temperature, energy, and paired markers |
| `POST` | `/start` | Start the background simulation loop |
| `POST` | `/stop` | Stop continuous evolution |
| `POST` | `/update` | Change parameters through a JSON body |
| `POST` | `/equilibrate` | Run the energy-change stopping heuristic while stopped |

For example, with the server running:

```bash
curl http://127.0.0.1:5000/data
curl -X POST http://127.0.0.1:5000/update \
  -H "Content-Type: application/json" \
  -d '{"temperature":1.5,"width":10,"time_step":0.05,"steps_per_frame":2}'
```

The `spins` array is flattened in row-major order and contains angles in radians.

## Project structure

| Path | Purpose |
| --- | --- |
| [XY_model.py](XY_model.py) | Spin system, Monte Carlo sweeps, energy, paired markers, and plotting |
| [app.py](app.py) | Flask API and background simulation loop |
| [templates/index.html](templates/index.html) | Controls and Canvas layout |
| [static/script.js](static/script.js) | Requests, polling, and Canvas drawing |
| [static/charts/](static/charts) | Stored chart images |

## Scope and limitations

This is an educational team project. The web app uses shared process-wide simulation state and a development server with debug mode enabled. Treat it as a local, single-user application.

The vortex routine estimates a winding value from neighboring angles and returns only nearby paired markers. It is an exploratory visualization; independent validation is needed before using its counts for quantitative topological analysis.

The web equilibration endpoint stops after a small energy change or 1,000 sweeps. This heuristic does not establish statistical equilibrium. The class-level annealing path applies additional energy normalization, so its energy values should be checked against `get_energy()` before comparing outputs.

No automated test suite or CI workflow is included.

## Authors and feedback

Anna Oleszko · [Bartosz Kawalec](https://github.com/kapitanek23) · Kacper Saj

Report issues with the parameters and steps to reproduce through [GitHub Issues](https://github.com/kapitanek23/XY-Model/issues).

## License

No license file is currently included in this repository.
