"""
ARGUS single-node simulator.

Runs the detection loop at a target frame rate, then fires a real Flower
federated round against it while it keeps detecting, and reports what that
did to the frame rate.

That interaction is the point. ARGUS_Materials_List_B_Sponsor_Spec.md
rejected every 8GB board against "REQ 3" on the claim that a live node
"stops detecting while it trains". That claim was written about Jetson
unified memory. This measures whether it is true for the architecture
actually being built, where detection is offloaded to the Hailo and the
trainer is a ~200K parameter head on the CPU.

    python sim/run_node.py
    python sim/run_node.py --profile pi5-threat-on-cpu --target-fps 12
    python sim/run_node.py --pipelined --rounds 8 --cores 4
"""

import os

# MUST precede any numpy import. numpy's BLAS is multithreaded by default, so
# a single "CPU stage" would otherwise saturate every core -- and with the
# trainer process doing the same, the two oversubscribe the machine and both
# collapse. One BLAS thread per process means one stage occupies one core,
# which is the thing being modelled.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import psutil

import pipeline as pl
import profiles
import workload
from head import FederatedHead

HERE = Path(__file__).resolve().parent
RUN = HERE / "_run"
EVENTS = RUN / "fl_events.jsonl"

try:
    from prometheus_client import Gauge, Counter, start_http_server
    _PROM = True
except ImportError:
    _PROM = False


class Metrics:
    """Prometheus surface. Mirrors the names the node already exports."""

    def __init__(self, port):
        self.on = _PROM and port > 0
        if not self.on:
            return
        self.fps = Gauge("argus_fps", "Rolling achieved frame rate")
        self.lat = Gauge("argus_frame_latency_ms", "Per-frame latency", ["quantile"])
        self.stage = Gauge("argus_stage_ms", "Mean stage time", ["stage"])
        self.frames = Gauge("argus_frames_total", "Frames processed")
        self.behind = Gauge("argus_frames_behind_total", "Frames that missed deadline")
        self.training = Gauge("argus_fl_training_active", "1 while an FL round runs")
        self.round = Gauge("argus_fl_round", "Completed FL rounds")
        self.acc = Gauge("argus_fl_accuracy", "Federated head accuracy")
        self.bal = Gauge("argus_fl_balanced_accuracy", "Balanced accuracy")
        self.loss = Gauge("argus_fl_loss", "Federated head loss")
        self.rss = Gauge("argus_rss_bytes", "Resident set, all sim processes")
        self.cpu = Gauge("argus_cpu_percent", "System CPU utilisation")
        start_http_server(port)

    def tick(self, stats, training, rss, cpu, fl=None):
        if not self.on:
            return
        if fl:
            self.round.set(fl["rounds"])
            self.acc.set(fl["acc"])
            self.bal.set(fl["bal_acc"])
            self.loss.set(fl["loss"])
        self.fps.set(stats.fps_over())
        lat = sorted(stats.latencies[-200:]) or [0.0]
        self.lat.labels("p50").set(lat[len(lat) // 2])
        self.lat.labels("p95").set(lat[int(len(lat) * 0.95) - 1])
        for k, v in stats.stage_ms.items():
            self.stage.labels(k).set(v)
        self.frames.set(stats.frames)
        self.behind.set(stats.behind)
        self.training.set(1 if training else 0)
        self.rss.set(rss)
        self.cpu.set(cpu)


def phase_fps(latencies):
    if not latencies:
        return 0.0
    return 1000.0 / (sum(latencies) / len(latencies))


def pct(latencies, q):
    if not latencies:
        return 0.0
    s = sorted(latencies)
    return s[min(len(s) - 1, int(len(s) * q))]


def latest_fl():
    """Most recent evaluate event, for the live FL gauges."""
    if not EVENTS.exists():
        return None
    acc = bal = loss = 0.0
    n = 0
    try:
        for line in EVENTS.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("kind") == "evaluate":
                n += 1
                acc, bal, loss = e["acc"], e["bal_acc"], e["loss"]
    except Exception:
        return None
    return {"rounds": n, "acc": acc, "bal_acc": bal, "loss": loss} if n else None


def total_rss(procs):
    tot = 0
    for p in procs:
        try:
            tot += p.memory_info().rss
        except Exception:
            pass
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=profiles.DEFAULT_PROFILE,
                    choices=list(profiles.PROFILES))
    ap.add_argument("--target-fps", type=float, default=12.0)
    ap.add_argument("--pipelined", action="store_true",
                    help="Overlap the accelerator with the CPU path")
    ap.add_argument("--site", default="driveway", choices=list(workload.SITES))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--clients", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--samples", type=int, default=25000)
    ap.add_argument("--warmup", type=float, default=12.0)
    ap.add_argument("--cooldown", type=float, default=10.0)
    ap.add_argument("--cores", type=int, default=0,
                    help="Pin the sim to N cores (4 models a Pi 5)")
    ap.add_argument("--port", type=int, default=9108, help="0 disables Prometheus")
    args = ap.parse_args()

    RUN.mkdir(parents=True, exist_ok=True)
    EVENTS.unlink(missing_ok=True)

    me = psutil.Process()
    if args.cores:
        try:
            me.cpu_affinity(list(range(args.cores)))
        except Exception as e:
            print(f"  (could not pin affinity: {e})")

    prof = profiles.get(args.profile)
    unit = pl.calibrate()

    site = workload.SITES[args.site]
    Xf, yf = workload.generate(site, 4000, seed=99)
    cursor = {"i": 0}

    def feed():
        i = cursor["i"] % len(Xf)
        cursor["i"] += 1
        return Xf[i], yf[i]

    head = FederatedHead(seed=0)
    metrics = Metrics(args.port)

    print()
    print("=" * 74)
    print(f"  ARGUS node simulator  |  profile: {prof.label}")
    print("=" * 74)
    print(f"  site            {site.key} -- {site.label}")
    print(f"  head            {head.n_params:,} params (CON-3 bound ~200K)")
    print(f"  mode            {'pipelined' if args.pipelined else 'serial (Table 11)'}")
    print(f"  target          {args.target_fps:.0f} FPS")
    cap = prof.pipelined_ms if args.pipelined else prof.serial_ms
    print(f"  budget ceiling  {1000 / cap:.1f} FPS   "
          f"(cpu {prof.cpu_ms:.0f} ms + accel {prof.accel_ms:.0f} ms)")
    print(f"  cores           {args.cores or psutil.cpu_count()} "
          f"{'(pinned)' if args.cores else '(all)'}")
    print(f"  calibration     1 matmul = {unit:.3f} ms")
    if metrics.on:
        print(f"  prometheus      http://127.0.0.1:{args.port}/metrics")
    print("=" * 74)
    print()

    procs = [me]
    state = {"training": False, "phase": "baseline"}
    marks = {}

    def on_tick(stats):
        rss = total_rss(procs)
        cpu = psutil.cpu_percent()
        metrics.tick(stats, state["training"], rss, cpu, latest_fl())
        flag = "TRAINING" if state["training"] else "        "
        print(f"  [{state['phase']:8}] {flag}  fps={stats.fps_over():5.2f}  "
              f"p95={pct(stats.latencies[-200:], 0.95):6.1f}ms  "
              f"frames={stats.frames:5d}  behind={stats.behind:4d}  "
              f"rss={rss / 1e6:6.1f}MB  cpu={cpu:5.1f}%")

    p = pl.Pipeline(prof, head, feed, args.target_fps, pipelined=args.pipelined)

    # ---- phase 1: baseline ------------------------------------------------
    p.run(args.warmup, on_tick)
    marks["baseline"] = list(p.stats.latencies)

    # ---- phase 2: federated round, detection still running ----------------
    state["phase"] = "fl-round"
    state["training"] = True
    py = sys.executable
    server = subprocess.Popen(
        [py, str(HERE / "fl.py"), "serve", "--rounds", str(args.rounds),
         "--clients", str(args.clients)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    p.run(2.0, on_tick)   # keep detecting while the server binds
    clients = []
    sites = list(workload.SITES)
    for i in range(args.clients):
        clients.append(subprocess.Popen(
            [py, str(HERE / "fl.py"), "client",
             "--site", sites[i % len(sites)],
             "--samples", str(args.samples),
             "--epochs", str(args.epochs),
             "--seed", str(i + 1)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
    for proc in [server] + clients:
        try:
            procs.append(psutil.Process(proc.pid))
        except Exception:
            pass

    start = len(p.stats.latencies)
    t0 = time.perf_counter()
    while server.poll() is None and (time.perf_counter() - t0) < 600:
        p.run(0.25, on_tick)
    for c in clients:
        try:
            c.wait(timeout=10)
        except Exception:
            c.kill()
    marks["fl"] = p.stats.latencies[start:]
    state["training"] = False

    # ---- phase 3: recovery ------------------------------------------------
    state["phase"] = "recovery"
    start = len(p.stats.latencies)
    p.run(args.cooldown, on_tick)
    marks["recovery"] = p.stats.latencies[start:]

    report(args, prof, p, marks, head)


def report(args, prof, p, marks, head):
    print()
    print("=" * 74)
    print("  FRAME RATE")
    print("=" * 74)
    print(f"  {'phase':12} {'frames':>7} {'fps':>7} {'p50 ms':>8} "
          f"{'p95 ms':>8} {'p99 ms':>8}")
    base = phase_fps(marks["baseline"])
    for name in ("baseline", "fl", "recovery"):
        lat = marks[name]
        f = phase_fps(lat)
        delta = "" if name == "baseline" else f"  ({(f - base) / base * 100:+.1f}%)"
        print(f"  {name:12} {len(lat):7d} {f:7.2f} {pct(lat, 0.50):8.1f} "
              f"{pct(lat, 0.95):8.1f} {pct(lat, 0.99):8.1f}{delta}")

    print()
    print("  mean stage time")
    for s in prof.stages:
        got = p.stats.stage_ms.get(s.name, 0.0)
        tag = {"cpu": "CPU", "accel": "HAILO", "measured": "LIVE"}[s.unit]
        budget = f"budget {s.ms:5.1f}" if s.unit != "measured" else "  measured"
        print(f"    {s.name:9} {tag:6} {got:6.2f} ms   {budget}   {s.source}")

    ev = []
    if EVENTS.exists():
        for line in EVENTS.read_text(encoding="utf-8").splitlines():
            try:
                ev.append(json.loads(line))
            except Exception:
                pass

    print()
    print("=" * 74)
    print("  FEDERATED ROUNDS (Flower FedAvg over gRPC)")
    print("=" * 74)
    evals = [e for e in ev if e["kind"] == "evaluate"]
    fits = [e for e in ev if e["kind"] == "fit"]
    done = next((e for e in ev if e["kind"] == "server_done"), None)

    # With N clients there are N evaluate events per round. The server-side
    # aggregate is the per-round number; the raw events are per-site.
    agg_acc = dict((int(r), v) for r, v in (done or {}).get("accuracy", []))
    agg_bal = dict((int(r), v) for r, v in (done or {}).get("balanced_accuracy", []))
    agg_loss = dict((int(r), v) for r, v in (done or {}).get("losses", []))

    if not agg_acc and evals:
        n = max(1, args.clients)
        for r in range(1, len(evals) // n + 1):
            grp = evals[(r - 1) * n:r * n]
            tot = sum(g["acc"] for g in grp)
            agg_acc[r] = tot / len(grp)
            agg_bal[r] = sum(g["bal_acc"] for g in grp) / len(grp)
            agg_loss[r] = sum(g["loss"] for g in grp) / len(grp)

    if agg_acc:
        print(f"  {'round':>5} {'loss':>8} {'accuracy':>10} "
              f"{'bal_acc':>9} {'fit s':>8}   (FedAvg aggregate over "
              f"{args.clients} client{'s' if args.clients > 1 else ''})")
        per_round = max(1, args.clients)
        for r in sorted(agg_acc):
            sl = fits[(r - 1) * per_round:r * per_round]
            fs = max((f["secs"] for f in sl), default=None)
            fss = f"{fs:8.2f}" if fs is not None else "       -"
            print(f"  {r:5d} {agg_loss.get(r, float('nan')):8.4f} "
                  f"{agg_acc[r]:10.4f} {agg_bal.get(r, float('nan')):9.4f} {fss}")
        ks = sorted(agg_acc)
        f0, f1 = ks[0], ks[-1]
        print()
        print(f"  accuracy      {agg_acc[f0]:.4f} -> {agg_acc[f1]:.4f}  "
              f"({(agg_acc[f1] - agg_acc[f0]) * 100:+.2f} pts)")
        print(f"  balanced acc  {agg_bal.get(f0, 0):.4f} -> {agg_bal.get(f1, 0):.4f}  "
              f"({(agg_bal.get(f1, 0) - agg_bal.get(f0, 0)) * 100:+.2f} pts)")
        if fits:
            tot = sum(f["secs"] for f in fits)
            print(f"  local train   {tot:.1f} s total, "
                  f"{tot / len(fits):.2f} s per client-round, {args.epochs} epochs each")

        # Per-site final standing. This is where non-IID shows itself.
        if args.clients > 1:
            print()
            print("  final round, per site (the non-IID spread):")
            tail = evals[-args.clients:]
            for e in tail:
                print(f"    {e['site']:9} acc={e['acc']:.4f}  "
                      f"bal_acc={e['bal_acc']:.4f}  loss={e['loss']:.4f}")
            spread = max(e["bal_acc"] for e in tail) - min(e["bal_acc"] for e in tail)
            print(f"    balanced-accuracy spread across sites: {spread:.3f}")
    else:
        print("  no rounds recorded -- check sim/_run/fl_events.jsonl")

    ready = [e for e in ev if e["kind"] == "client_ready"]
    if ready:
        print()
        for r in ready:
            print(f"  client {r['site']:9} train={r['train']} val={r['val']} "
                  f"params={r['params']:,}")

    print()
    print("=" * 74)
    print("  VERDICT")
    print("=" * 74)
    fl_fps = phase_fps(marks["fl"])
    rec = phase_fps(marks["recovery"])
    drop = (base - fl_fps) / base * 100 if base else 0.0
    verb = "lost" if drop > 0 else "gained"
    print(f"  Detection held {fl_fps:.2f} FPS while training -- "
          f"{verb} {abs(drop):.1f}% against a {base:.2f} FPS baseline; "
          f"recovered to {rec:.2f}.")
    if fl_fps >= args.target_fps:
        print(f"  Target of {args.target_fps:.0f} FPS held THROUGH the federated round.")
    else:
        print(f"  Target of {args.target_fps:.0f} FPS NOT held during the round.")
    if args.clients == 1:
        print()
        print("  Note: with one client, FedAvg averages a single update, so the")
        print("  round exercises the full path (fit -> aggregate -> validate ->")
        print("  distribute) but the accuracy curve is just local training. Run")
        print("  --clients 3 to see averaging across the three sites actually bite.")
    if args.clients > 1:
        print()
        print("  !! The frame-rate table above is NOT a node measurement in this")
        print("     mode. All {} trainers ran on THIS machine alongside the".format(args.clients))
        print("     pipeline; in the field each client is its own Pi. Use")
        print("     --clients 1 for frame rate, --clients 3 for FedAvg behaviour.")
    print()
    print("  Caveats, so these numbers are not over-read:")
    print("    - CPU stage times are budgets from Rev 2.2 Table 11, not Pi")
    print("      measurements. No Pi 5 + Hailo-10H benchmark exists yet.")
    print("    - The head is numpy. Keeping TensorFlow in the loop adds")
    print("      roughly 1 GB of framework RSS this does not model.")
    print("    - Timings here are this machine's cores, not the Pi's. Use")
    print("      --cores 4 to at least match the core count.")
    print()


if __name__ == "__main__":
    main()
