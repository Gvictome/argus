# ARGUS simulation report

A notebook for the sponsor and team that answers three questions with the simulator in `sim/`:

1. Does federated learning beat a node training alone? (including a new camera joining the fleet)
2. Does a node keep detecting at 12 FPS while it trains?
3. Can one bad camera poison the fleet, and does `src/federated/robust.py` stop it?

## Run it

From the repo root:

```bash
python sim/report/run_report.py                 # full run, about 25 min
python sim/report/run_report.py --reuse-fps     # reuse the cached frame-rate runs, about 15 min
python sim/report/run_report.py --copy-to "C:/Users/colab/Dropbox/gio/argus/argus"
```

Close other heavy apps first. Section 2 measures frame rate on this PC, so background load shows up as a training cost.

Needs `numpy pandas matplotlib psutil flwr nbformat nbconvert jupyter`, plus Chrome or Edge for the PDF.

## Outputs

| File | What |
|---|---|
| `ARGUS_Simulation_Results.pdf` | the report to send |
| `ARGUS_Simulation_Results.html` | the same report, no code |
| `ARGUS_Simulation_Results.ipynb` | executed notebook, code cells collapsed |

## Files

| File | Role |
|---|---|
| `experiments.py` | the experiments; reuses `sim/` and `robust.py` without changing them |
| `build_notebook.py` | writes the notebook |
| `summary.md`, `findings.md` | the written conclusions, pulled into the notebook |
| `run_report.py` | build, execute, export HTML and PDF |

The accuracy and robustness numbers are seeded and repeat exactly. The frame-rate numbers change from run to run.
If you change the experiments, re-read `summary.md` and `findings.md` against the new tables.
