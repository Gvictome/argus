#!/usr/bin/env python3
"""
Prepare a node's events for a federated round: dedupe, prioritise, queue,
review, label.

    build   Read a node's data directory, group parked-object repeats,
            give every event a priority, and write a labelling queue with
            a review page (photos with boxes) and a held-out split.
    apply   Send the reviewed training labels to the node; keep the held-out
            labels on this machine as the frozen test set.

Typical use (Node A):

    PI      python scripts/prep_round.py build --data data --node http://localhost:8000 --out /tmp/round1
    LAPTOP  scp -r viccia@10.0.0.140:/tmp/round1 $HOME\\Documents\\argus-round1
            open review.html, check every card, "Download labels.json"
    LAPTOP  python scripts/prep_round.py apply --node http://10.0.0.140:8000 --labels labels.json

Priorities (docs/design: clip retention):
    P1  the node's head calls it unusual for the site, or it is labelled anomaly
    P2  everything else that is a real, distinct event
    P3  a parked/stationary repeat of an object already on record

Parked repeats use each event's own features (centre, size, speed), the
same rule the live collector now applies: same class, box overlap >= 0.8
with a known still object, low speed. One representative per object per
hour goes into the queue.

The held-out split is the latest 20% of each class's queue by time, not a
random draw: near-identical frames of one parked car on both sides of a
random split would make the test set a copy of the training set.
"""

from __future__ import annotations

import argparse
import base64
import collections
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASSES = ("person", "vehicle", "animal", "package")
LABELS = ("routine_person", "routine_vehicle", "routine_animal", "routine_package", "anomaly")
STATIONARY_CLASSES = {"vehicle", "package"}
STILL_SPEED = 0.05        # feature x[12]; parked vehicles: median 0.009, p90 0.028; people walk at ~0.05
GROUP_IOU = 0.8
FORGET_S = 1800.0         # a parked object unseen this long has left
KEEP_S = 3600.0           # one representative per parked object per hour
PERSON_SPACING_S = 600.0  # people: at most one queued per 10 minutes
CAPS = {"person": 80, "vehicle": 120, "animal": 60, "package": 60, "flagged": 80}
HOLDOUT_FRAC = 0.2
MIN_TAUGHT = 10           # routine labels a class needs before the head's "unusual" counts


def _cls(x):
    head = x[:4]
    return CLASSES[max(range(4), key=lambda i: head[i])] if any(head) else "unknown"


def _feature_box(x):
    cx, cy, w, h = x[8], x[9], x[4], x[5]
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_rows(data: Path):
    path = data / "training" / "samples.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return [r for r in rows if not (r.get("meta") or {}).get("synthetic")]


def node_status(node: str | None):
    """Event id -> the live head's call ("stranger" = unusual), if reachable."""
    if not node:
        return {}
    try:
        with urllib.request.urlopen(f"{node}/api/events?limit=100000", timeout=60) as resp:
            return {e["id"]: e["status"] for e in json.load(resp)}
    except Exception as exc:  # the queue still builds without the head's view
        print(f"note: could not read {node}/api/events ({exc}); P1 = labelled anomalies only")
        return {}


def group_parked(rows):
    """Assign parked-object groups. Returns {n: (group_id, is_representative)}."""
    out, parked, next_id = {}, [], 1
    for r in sorted(rows, key=lambda r: r["t"]):
        x, cls = r["x"], _cls(r["x"])
        if cls not in STATIONARY_CLASSES or x[12] > STILL_SPEED:
            continue
        box, now = _feature_box(x), r["t"]
        parked = [p for p in parked if now - p["last_seen"] <= FORGET_S]
        match = next((p for p in parked if p["cls"] == cls and _iou(p["box"], box) >= GROUP_IOU), None)
        if match is None:
            match = {"id": next_id, "cls": cls, "box": box, "last_seen": now, "last_kept": now}
            parked.append(match)
            next_id += 1
            out[r["n"]] = (match["id"], True)
            continue
        match["last_seen"], match["box"] = now, box
        rep = now - match["last_kept"] >= KEEP_S
        if rep:
            match["last_kept"] = now
        out[r["n"]] = (match["id"], rep)
    return out


def build(args):
    data = Path(args.data)
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows = load_rows(data)
    status = node_status(args.node)
    groups = group_parked(rows)

    # The head's "unusual" call only means something for classes it has
    # been taught; for an unlabelled class (vehicles on Node A) it flags
    # everything, because everything of that class is unlike its data.
    taught = collections.Counter(_cls(r["x"]) for r in rows if r.get("y") is not None and r["y"] != 4)
    trusted = {c for c in CLASSES if taught[c] >= MIN_TAUGHT}

    events = []
    for r in rows:
        n, x = r["n"], r["x"]
        cls = _cls(x)
        labelled = r.get("y") is not None
        head_says = status.get(n) == "stranger"
        flagged = (head_says and cls in trusted) or (labelled and r["y"] == 4)
        g = groups.get(n)
        # A parked repeat is P3 whatever the head says: until it has seen a
        # labelled vehicle it calls every parked car unusual (1,340 of them
        # on Node A), which is the label gap, not 1,340 anomalies.
        if g and not g[1]:
            prio = "P3"
        elif flagged and not g:
            prio = "P1"
        else:
            prio = "P2"
        events.append({
            "n": n, "t": r["t"], "cls": cls, "labelled": labelled,
            "label": LABELS[r["y"]] if labelled else None, "priority": prio,
            "group": g[0] if g else None, "head_flagged": head_says, "speed": round(x[12], 4),
            "confidence": round(x[18], 2), "meta": r.get("meta") or {}, "x": x,
        })

    # --- the queue: unlabelled, distinct, spread out ---------------------
    queue, picked = [], set()

    def take(items, cap, why):
        for e in items[:cap]:
            if e["n"] not in picked:
                picked.add(e["n"])
                queue.append({**e, "why": why})

    unl = [e for e in events if not e["labelled"]]
    take(sorted([e for e in unl if e["priority"] == "P1"], key=lambda e: e["t"]), CAPS["flagged"], "flagged unusual")
    veh = [e for e in unl if e["cls"] == "vehicle" and e["priority"] == "P2"]
    take(_spread(veh, CAPS["vehicle"]), CAPS["vehicle"], "vehicle (one per parked car per hour, or moving)")
    people, last = [], -1e18
    for e in sorted([e for e in unl if e["cls"] == "person" and e["priority"] == "P2"], key=lambda e: e["t"]):
        if e["t"] - last >= PERSON_SPACING_S:
            people.append(e)
            last = e["t"]
    take(_spread(people, CAPS["person"]), CAPS["person"], "person (10 min apart)")
    take(sorted([e for e in unl if e["cls"] == "animal"], key=lambda e: e["t"]), CAPS["animal"], "animal (default skip: on Node A these were a person's leg)")
    take(sorted([e for e in unl if e["cls"] == "package"], key=lambda e: e["t"]), CAPS["package"], "package")

    # --- held-out split: latest 20% of each class by time ----------------
    by_cls = collections.defaultdict(list)
    for q in queue:
        by_cls[q["cls"]].append(q)
    for items in by_cls.values():
        items.sort(key=lambda q: q["t"])
        cut = len(items) - int(round(len(items) * HOLDOUT_FRAC))
        for i, q in enumerate(items):
            q["split"] = "holdout" if i >= cut else "train"
    queue.sort(key=lambda q: (q["cls"], q["t"]))

    # --- photos with boxes -----------------------------------------------
    from src.detection.snapshot_overlay import draw_overlay, event_box
    import cv2
    import numpy as np

    snaps = data / "snapshots"
    for q in queue:
        name = q["meta"].get("snapshot")
        q["img"] = None
        if not name or not (snaps / name).exists():
            continue
        box = event_box(q["meta"], snaps) or _feature_box(q["x"])
        jpeg = draw_overlay((snaps / name).read_bytes(), box, f"{q['cls']} {int(q['confidence'] * 100)}%",
                            q["priority"] == "P1")
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None and img.shape[1] > 360:
            img = cv2.resize(img, (360, int(img.shape[0] * 360 / img.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "img" / f"{q['n']}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        q["img"] = f"img/{q['n']}.jpg"

    # --- outputs -----------------------------------------------------------
    for q in queue:
        q.pop("x", None)
        q.pop("meta", None)
        q["time"] = dt.datetime.fromtimestamp(q["t"]).strftime("%Y-%m-%d %H:%M")
        q["suggest"] = "anomaly" if q["priority"] == "P1" else f"routine_{q['cls']}"
        # Detector mistakes are not site behaviour: labelling a leg detected
        # as an animal "routine_person" would teach the head a false rule.
        if q["cls"] == "animal":
            q["suggest"] = "skip"
        if q.get("head_flagged") and q["priority"] != "P1":
            q["why"] += "; head flags it (no labels for this yet)"
    (out / "queue.json").write_text(json.dumps(queue, indent=1))
    summary = _summary(events, queue, groups, trusted)
    (out / "summary.md").write_text(summary)
    (out / "review.html").write_text(_review_html(queue, summary), encoding="utf-8")
    print(summary)
    print(f"\nwrote {out / 'review.html'} ({len(queue)} cards)")


def _spread(items, cap):
    """Up to `cap` items evenly spaced through time."""
    items = sorted(items, key=lambda e: e["t"])
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def _summary(events, queue, groups, trusted):
    by = collections.Counter((e["priority"], e["cls"]) for e in events)
    lab = collections.Counter(e["label"] for e in events if e["labelled"])
    parked_objects = len({g for g, _ in groups.values()})
    lines = [
        "# Round prep summary",
        "",
        f"Events: {len(events)} (labelled {sum(1 for e in events if e['labelled'])})",
        f"Labelled now: " + ", ".join(f"{k} {v}" for k, v in sorted(lab.items())),
        f"Parked objects found: {parked_objects}, covering {len(groups)} still vehicle/package events",
        f"Head trusted for: {', '.join(sorted(trusted)) or 'none'} (>= {MIN_TAUGHT} routine labels)",
        f"Head calls unusual: {sum(1 for e in events if e['head_flagged'])} "
        f"(of them parked/still vehicle or package: {sum(1 for e in events if e['head_flagged'] and e['group'])})",
        "",
        "| priority | " + " | ".join(CLASSES) + " |",
        "|---|" + "---|" * len(CLASSES),
    ]
    for p in ("P1", "P2", "P3"):
        lines.append(f"| {p} | " + " | ".join(str(by[(p, c)]) for c in CLASSES) + " |")
    qc = collections.Counter((q["cls"], q["split"]) for q in queue)
    lines += ["", f"Queue: {len(queue)} cards", "",
              "| class | train | holdout |", "|---|---|---|"]
    for c in CLASSES:
        lines.append(f"| {c} | {qc[(c, 'train')]} | {qc[(c, 'holdout')]} |")
    return "\n".join(lines) + "\n"


def _review_html(queue, summary):
    data = base64.b64encode(json.dumps(queue).encode()).decode()
    options = "".join(f"<option>{o}</option>" for o in (*LABELS, "skip"))
    return """<!doctype html><html><head><meta charset="utf-8"><title>Round review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{margin:0;font:13px system-ui,Segoe UI,Arial;background:#0e1116;color:#e6edf3}
header{position:sticky;top:0;background:#161b22;border-bottom:1px solid #2a313c;padding:10px 14px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;z-index:2}
button,select{background:#0e1116;color:#e6edf3;border:1px solid #2a313c;border-radius:6px;padding:5px 8px;font:inherit}
button.primary{border-color:#3fb950;color:#3fb950}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px;padding:12px}
.card{background:#161b22;border:1px solid #2a313c;border-radius:8px;overflow:hidden}
.card.changed{border-color:#d29922}.card.skip{opacity:.45}
.card img{width:100%;display:block;background:#000;aspect-ratio:16/9;object-fit:cover}
.meta{padding:6px 8px;color:#8b949e;display:flex;flex-wrap:wrap;gap:4px 8px}
.meta b{color:#e6edf3}.p1{color:#f85149}.ho{color:#58a6ff}
.row{padding:0 8px 8px}.row select{width:100%}
pre{white-space:pre-wrap;color:#8b949e;margin:0 14px;font-size:12px}
</style></head><body>
<header>
 <b>Round review</b>
 <label>class <select id="f"><option value="">all</option><option>person</option><option>vehicle</option><option>animal</option><option>package</option></select></label>
 <label>split <select id="s"><option value="">all</option><option>train</option><option>holdout</option></select></label>
 <span id="count"></span>
 <button class="primary" id="dl">Download labels.json</button>
</header>
<details><summary style="padding:8px 14px;cursor:pointer">Summary</summary><pre id="sum"></pre></details>
<main id="grid"></main>
<script>
const Q = JSON.parse(atob('""" + data + """'));
const SUMMARY = """ + json.dumps(summary) + """;
const OPTS = '""" + options + """';
document.getElementById('sum').textContent = SUMMARY;
const grid = document.getElementById('grid');
const choice = {};
Q.forEach(q => {
  choice[q.n] = q.suggest;
  const c = document.createElement('div');
  c.className = 'card'; c.dataset.cls = q.cls; c.dataset.split = q.split;
  c.innerHTML = (q.img ? `<img loading="lazy" src="${q.img}">` : '<img alt="no photo">') +
    `<div class="meta"><b>#${q.n}</b><span>${q.time}</span><span class="${q.priority==='P1'?'p1':''}">${q.priority}</span>` +
    `<span class="${q.split==='holdout'?'ho':''}">${q.split}</span><span>${q.why}</span></div>` +
    `<div class="row"><select>${OPTS}</select></div>`;
  const sel = c.querySelector('select');
  sel.value = q.suggest;
  if (q.suggest === 'skip') c.classList.add('skip');
  sel.onchange = () => { choice[q.n] = sel.value; c.classList.toggle('changed', sel.value !== q.suggest); c.classList.toggle('skip', sel.value === 'skip'); };
  grid.appendChild(c);
});
function filter(){ const f=document.getElementById('f').value, s=document.getElementById('s').value; let n=0;
  grid.querySelectorAll('.card').forEach(c=>{ const on=(!f||c.dataset.cls===f)&&(!s||c.dataset.split===s); c.style.display=on?'':'none'; n+=on; });
  document.getElementById('count').textContent = n + ' shown'; }
document.getElementById('f').onchange = filter; document.getElementById('s').onchange = filter; filter();
document.getElementById('dl').onclick = () => {
  const out = Q.filter(q => choice[q.n] !== 'skip').map(q => ({n:q.n, label:choice[q.n], split:q.split, cls:q.cls, t:q.t}));
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type:'application/json'}));
  a.download = 'labels.json'; a.click();
};
</script></body></html>"""


def apply(args):
    items = json.loads(Path(args.labels).read_text())
    train = [i for i in items if i["split"] == "train"]
    holdout = [i for i in items if i["split"] == "holdout"]
    bad = [i for i in items if i["label"] not in LABELS]
    if bad:
        sys.exit(f"unknown labels: {sorted({i['label'] for i in bad})}")

    hold_path = Path(args.holdout)
    hold_path.write_text(json.dumps(holdout, indent=1))
    print(f"held out {len(holdout)} labels -> {hold_path} (never sent to the node)")

    counts = collections.Counter(i["label"] for i in train)
    print(f"training labels for {args.node}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if args.dry_run:
        print("dry run: nothing sent")
        return
    sent = failed = 0
    for i in train:
        body = json.dumps({"action": "correct", "label": i["label"]}).encode()
        req = urllib.request.Request(f"{args.node}/api/events/{i['n']}/label", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                sent += resp.status == 200
        except Exception as exc:
            failed += 1
            print(f"  event {i['n']}: {exc}")
    print(f"sent {sent}, failed {failed}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", default="data", help="the node's data directory")
    b.add_argument("--node", default=None, help="node API, for the head's unusual calls (P1)")
    b.add_argument("--out", default="round-prep")
    a = sub.add_parser("apply")
    a.add_argument("--node", required=True)
    a.add_argument("--labels", required=True)
    a.add_argument("--holdout", default="holdout.json")
    a.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    build(args) if args.cmd == "build" else apply(args)


if __name__ == "__main__":
    main()
