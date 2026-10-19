"""
Build sim/report/ARGUS_Simulation_Results.ipynb.

    python sim/report/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace sim/report/ARGUS_Simulation_Results.ipynb
    jupyter nbconvert --to html --no-input sim/report/ARGUS_Simulation_Results.ipynb

The written findings live in findings.md and summary.md beside this file, so
the prose can be revised after a run without touching the code cells.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
OUT = HERE / "ARGUS_Simulation_Results.ipynb"
HIDDEN = {"jupyter": {"source_hidden": True}}

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip(), metadata=HIDDEN))


def prose(name: str) -> str:
    p = HERE / name
    return p.read_text(encoding="utf-8") if p.exists() else f"_{name} is written after the first run._"


md("""
# ARGUS Simulation Results
**Federated learning, frame rate under training, and poisoning resistance**

Prepared for the sponsor and team. Code cells are collapsed. Every number below is produced by the
simulator in `sim/` on the `feat/fine-taxonomy` branch, and the notebook re-runs end to end.
""")
code("""
import subprocess, datetime
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
print(f"Run {datetime.date.today()}  |  branch {branch} @ {commit}")
""")

md("## Summary")
md(prose("summary.md"))

md("""
## What is being simulated

**The federated model is not the detector.** YOLO stays frozen and finds person / vehicle / animal / package.
The federated part is a small head (205,445 parameters, 19 inputs) that decides whether a detection is
**routine for this site** or an **anomaly**. A delivery van at 14:00 is routine, and the same van at 03:00 is not.

Three simulated camera sites have different traffic and different rules for what counts as abnormal:

| Site | Mostly sees | What is abnormal there |
|---|---|---|
| driveway | vehicles, residents | anything at night, loitering |
| backdoor | packages, staff | any vehicle, at any hour |
| yard | animals | people at night |

The two headline metrics:
- **Balanced accuracy:** average accuracy across the 5 labels, so rare labels count as much as common ones.
- **Anomaly recall:** of the events that really were abnormal, the share the head flagged. This is the security number.

All accuracy results are averaged over 3 random seeds. Error bars show the spread between seeds.
""")
code("""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path.cwd()
while not (ROOT / "sim").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "sim" / "report"))
import experiments as ex

SEEDS = (1, 2, 3)
import os
# run_report.py --reuse-fps sets this to 0 to reuse the cached frame-rate runs in sim/_run
REFRESH_FPS = os.environ.get("ARGUS_REFRESH_FPS", "1") == "1"
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
ARM_STYLE = {
    "untrained": dict(color="#9e9e9e", ls=":"),
    "local only": dict(color="#1f77b4"),
    "federated": dict(color="#d62728"),
    "federated + local fine-tune": dict(color="#2ca02c", lw=2.5),
    "fleet model as-is": dict(color="#d62728"),
    "fleet model + local fine-tune": dict(color="#2ca02c", lw=2.5),
}


def curve_plot(df, x, arms, title, xlabel, logx=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, metric, label in zip(axes, ["balanced_acc", "anomaly_recall"], ["Balanced accuracy", "Anomaly recall"]):
        for arm in arms:
            g = df[df.arm == arm].groupby(x)[metric]
            m, s = g.mean(), g.std().fillna(0)
            ax.errorbar(m.index, m.values, yerr=s.values, marker="o", capsize=3, label=arm, **ARM_STYLE[arm])
        ax.set(title=f"{title}: {label.lower()}", xlabel=xlabel, ylabel=label, ylim=(0, 1))
        if logx:
            xs = sorted(df[x].unique())
            ax.set_xscale("symlog", linthresh=50)
            ax.set_xticks(xs, [str(v) for v in xs])
            ax.set_xlim(xs[0] - 10 if xs[0] == 0 else xs[0] * 0.8, xs[-1] * 1.25)
    axes[0].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()


def mean_table(df, index, columns="arm", metric="balanced_acc"):
    return df.pivot_table(index=index, columns=columns, values=metric, aggfunc="mean").round(3)
""")

md("""
## 1. Does federated learning beat a node training alone?
This is the sponsor's question. Each site's head is obtained four ways:

| Arm | How the head is trained |
|---|---|
| untrained | random weights (the floor) |
| local only | trains on its own events only, and never shares |
| federated | FedAvg across the 3 sites, 6 rounds; every site runs the same global model |
| federated + local fine-tune | the federated model, then a short training pass on the site's own events |

Local-only gets the same number of local training epochs as federated + fine-tune ends up with, so it does not
lose by training less. The federated arms still learn from 3 sites' data instead of 1; that is the point of federating.

### 1a. Same amount of data at every site
""")
code("""
fl = pd.DataFrame(ex.fl_vs_local(sizes=[50, 150, 500, 1500, 5000], seeds=SEEDS))
ARMS = ["untrained", "local only", "federated", "federated + local fine-tune"]
curve_plot(fl, "n_per_site", ARMS, "Events per site", "Labelled events per site")
print("Balanced accuracy (mean of sites and seeds)")
display(mean_table(fl, "n_per_site")[ARMS])
print("Anomaly recall")
display(mean_table(fl, "n_per_site", metric="anomaly_recall")[ARMS])
""")
code("""
# Per-site view at a realistic first-month volume: where does the shared model hurt?
display(fl[fl.n_per_site == 500].pivot_table(index="site", columns="arm", values="balanced_acc").round(3)[ARMS])
""")

md("""
### 1b. A new camera joins the fleet
Leave-one-site-out: the fleet trains on two sites (3,000 events each). The third site gets a camera that
the fleet model has **never seen**, with only `local n` events of its own (0 means it was installed today).
This is the day-one experience of a new customer.
""")
code("""
cs = pd.DataFrame(ex.cold_start(local_sizes=[0, 50, 150, 500, 1500], seeds=SEEDS))
CS_ARMS = ["untrained", "local only", "fleet model as-is", "fleet model + local fine-tune"]
curve_plot(cs, "local_n", CS_ARMS, "New camera", "Labelled events at the new camera")
display(mean_table(cs, "local_n").reindex(columns=CS_ARMS))
display(mean_table(cs, "local_n", metric="anomaly_recall").reindex(columns=CS_ARMS))
""")

md("""
### 1c. Fine taxonomy (carrier, make, package size) and the household-car problem
This re-runs `sim/compare_taxonomy.py` across 4 sites (adding `curbside`) and adds the fine-tune arm.
"Household car, night" is the resident's own car arriving late. A shared model cannot know whose car
it is, so this slice tests whether site-specific knowledge survives federation.
""")
code("""
tx_raw = pd.DataFrame(ex.taxonomy_with_finetune(accs=[1.0, 0.88, 0.76], seeds=SEEDS))
cols = ["balanced accuracy", "household car, night", "stranger car, night",
        "carrier truck, out of window", "usps regular mail"]
tx = tx_raw.groupby(["classifier_acc", "taxonomy", "arm"], as_index=False)[cols].mean()
display(tx.set_index(["classifier_acc", "taxonomy", "arm"])[cols].round(3))
print("Validation events per slice, summed over 4 sites (mean per seed). Small slices swing a lot:")
display(tx_raw.groupby("seed")[[f"n: {c}" for c in cols[1:]]].first().mean().round(0).astype(int).to_frame("events"))
""")
code("""
sub = tx[tx.classifier_acc == 0.88]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, metric in zip(axes, ["balanced accuracy", "household car, night"]):
    sub.pivot(index="arm", columns="taxonomy", values=metric).loc[
        ["local only", "federated", "federated + local fine-tune"]].plot.bar(ax=ax, rot=0, color=["#bbbbbb", "#ff7f0e"])
    ax.set(title=f"{metric} (crop classifier 88% accurate)", ylim=(0, 1), xlabel="")
plt.tight_layout()
plt.show()
""")

md("""
## 2. Does a node keep detecting while it trains?
`sim/run_node.py` runs the detection loop at 12 FPS. A **real Flower round** (gRPC, separate OS processes,
5 rounds x 12 epochs on 20,000 events) runs alongside it, pinned to 4 cores like a Pi 5. Frame rate is measured
in three phases: before, during, and after training.

| Configuration | Meaning |
|---|---|
| pi5-hailo | target build: YOLO + threat model on the Hailo, serial loop |
| pi5-hailo pipelined | same, with the Hailo overlapped with CPU work |
| pi5-hailo-fine | target build plus the carrier/make/size crop classifier |
| pi5-threat-on-cpu | risk case: the threat model never gets compiled for the Hailo |
| pi5-cpu-only | fallback: the Hailo is missing or fails its probe |

Hailo and CPU stage times are budgets from Proposal Rev 2.2 Table 11. They are **not Pi measurements**.
""")
code("""
FPS_RUNS = {
    "pi5-hailo": ["--profile", "pi5-hailo"],
    "pi5-hailo pipelined": ["--profile", "pi5-hailo", "--pipelined"],
    "pi5-hailo-fine": ["--profile", "pi5-hailo-fine", "--taxonomy", "fine"],
    "pi5-threat-on-cpu": ["--profile", "pi5-threat-on-cpu"],
    "pi5-cpu-only": ["--profile", "pi5-cpu-only"],
}
fps = {name: ex.run_fps(name.replace(" ", "_"), args, refresh=REFRESH_FPS) for name, args in FPS_RUNS.items()}
fps_df = pd.DataFrame([
    {"configuration": n, "phase": ph, **v}
    for n, r in fps.items() for ph, v in r["phases"].items()
])
tbl = fps_df.pivot(index="configuration", columns="phase", values="fps")[["baseline", "fl", "recovery"]]
tbl["change during training"] = ((tbl["fl"] - tbl["baseline"]) / tbl["baseline"] * 100).round(1).astype(str) + "%"
tbl["p95 ms during training"] = fps_df[fps_df.phase == "fl"].set_index("configuration")["p95"]
tbl["holds 12 FPS while training"] = np.where(tbl["fl"] >= 12, "yes", "NO")
display(tbl.loc[list(FPS_RUNS)].round(2))
""")
code("""
ax = fps_df.pivot(index="configuration", columns="phase", values="fps").loc[list(FPS_RUNS), ["baseline", "fl", "recovery"]] \\
    .rename(columns={"fl": "during FL round"}).plot.bar(figsize=(11, 4.2), rot=0, color=["#1f77b4", "#d62728", "#2ca02c"])
ax.axhline(12, color="k", ls="--", lw=1, label="12 FPS target")
ax.set(title="Detection frame rate before, during and after a federated training round", ylabel="Frames per second", xlabel="")
ax.legend()
plt.tight_layout()
plt.show()
""")

md("""
## 3. Can one bad camera poison the fleet?
A 6-camera fleet (2 per site) trains for 6 rounds. One driveway camera sends a bad update **every round**.
The attacks:

| Attack | What the bad camera sends |
|---|---|
| none | honest update (control) |
| NaN (diverged node) | a crashed or diverged trainer: all NaN |
| random noise | garbage weights |
| sign flip x4 | its real update reversed and amplified, to drag the model backwards |
| label flip (hide anomalies) | trained honestly, but on data where every anomaly is labelled routine (a quiet, targeted attack) |
| label flip, boosted x2.5 | the same, scaled just under robust.py's norm limit (a careful attacker) |
| label flip, boosted x6 | the same, scaled up to take over the average (model replacement) |

Server-side defences:
- **FedAvg:** what the Flower server runs today.
- **Trimmed mean:** drop the highest and lowest value of each weight, then average.
- **robust.py:** `src/federated/robust.py` with its shipped default settings. Written and tested, but **not yet wired into the server**.
- **robust.py + validation gate:** also refuses a new model whose balanced accuracy drops on a small server-side
  holdout (600 events the nodes never train on and the final score never uses).
""")
code("""
ATTACKS = ["none", "NaN (diverged node)", "random noise", "sign flip x4",
           "label flip (hide anomalies)", "label flip, boosted x2.5", "label flip, boosted x6"]
STRATS = ["FedAvg", "trimmed mean", "robust.py", "robust.py + validation gate"]
rb = pd.DataFrame(ex.robustness(ATTACKS, seeds=SEEDS))
for metric, label in [("anomaly_recall", "Anomaly recall"), ("balanced_acc", "Balanced accuracy")]:
    print(label + " of the fleet model after 6 rounds (0.000 = model destroyed)")
    display(rb.pivot_table(index="attack", columns="strategy", values=metric).loc[ATTACKS, STRATS].round(3))
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
for ax, metric, label in zip(axes, ["anomaly_recall", "balanced_acc"], ["Anomaly recall", "Balanced accuracy"]):
    piv = rb.pivot_table(index="attack", columns="strategy", values=metric).loc[ATTACKS, STRATS]
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(STRATS)), STRATS, rotation=20, ha="right")
    ax.set_yticks(range(len(ATTACKS)), ATTACKS)
    ax.grid(False)
    for i in range(len(ATTACKS)):
        for j in range(len(STRATS)):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(label)
fig.colorbar(im, ax=axes, shrink=0.8)
plt.show()

rob = rb[rb.strategy.str.startswith("robust")]
print("Rounds (of 6) in which robust.py rejected the bad camera, and honest updates it wrongly rejected (of 30):")
display(rob.groupby(["attack", "strategy"])[["attacker_rejected_rounds", "honest_rejections"]].mean()
        .unstack().loc[ATTACKS].round(1))
""")

md("## Findings and recommendations")
md(prose("findings.md"))

md("""
## What this does not show
- **Synthetic events.** Site traffic, delivery windows, and anomaly rules are simulation assumptions. No real ARGUS footage is used.
  The 2026-09-20 hardware run (322 labelled events, balanced accuracy 0.75) is the only real-data point so far.
- **Frame-rate budgets, not Pi measurements.** The stage times come from Proposal Rev 2.2 Table 11. No Pi 5 + Hailo benchmark exists in the repo yet.
- **This machine's cores, not a Pi's.** Pinning to 4 cores matches the core count, not the per-core speed.
- **In-process federation for the accuracy studies.** It uses the same FedAvg maths as the Flower server, without the network.
  Section 2 uses the real Flower gRPC path.
- **Attacks are simple.** A patient attacker who stays just under the norm limit and agrees with the consensus direction is
  harder to stop. Section 3 shows the quiet label flip is already one of these.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
nbf.write(nb, OUT)
print("wrote", OUT)
