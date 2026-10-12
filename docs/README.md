# ARGUS documentation

Markdown here is the source of truth; `docs/pdf/` holds rendered copies
(`python scripts/build_docs_pdf.py`, `--master`, `--guides`).

| Folder | What is in it | Start with |
|---|---|---|
| [reference/](reference/) | The technical reference, ten sections, current. The master PDF is built from it | [01-system-overview.md](reference/01-system-overview.md) |
| [guides/](guides/) | Eight plain-language guides, one topic each | [01-what-is-argus.md](guides/01-what-is-argus.md) |
| [operations/](operations/) | Running it: quick start, demo runbook, production FL, Hailo, tunnel, hardware | [QUICKSTART.md](operations/QUICKSTART.md) |
| [design/](design/) | How the parts are designed and measured: overview, subsystems, federated architecture, threat stage | [OVERVIEW.md](design/OVERVIEW.md) |
| [research/](research/) | Experiments and data: dataset plan, simulations, detector and threat evaluations | [datasets.md](research/datasets.md) |
| [project/](project/) | Proposal, showcase PRD, sprint plan, demo script | [PROPOSAL.md](project/PROPOSAL.md) |
| [archive/](archive/) | January-era design docs and the Prometheus voice tool. Kept for history; the reference wins where they disagree | |

## Research

| Doc | Status |
|---|---|
| [datasets.md](research/datasets.md) | Decision 2026-10-12: MEVA replaces CIFAR-10; Stanford Cars + MEVA crops for make; ShanghaiTech for anomaly testing |
| [fine-taxonomy-simulation.md](research/fine-taxonomy-simulation.md) | Done 2026-10-03: 35 features lift FedAvg 0.860 → 0.894; local-only still ahead |
| [detector-comparison.md](research/detector-comparison.md) | Laptop done 2026-10-11; Pi speed still to run |
| [threat-eval.md](research/threat-eval.md) | In progress: weapon test set, download paused |

Team task tracking: [ARGUS roadmap](https://claude.ai/artifact/DVn9kV5xUDup13ohbHnWTJ).
