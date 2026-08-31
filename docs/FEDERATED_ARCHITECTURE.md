# Federated Architecture

How ARGUS trains across many deployments without collecting anyone's
footage, and how the global model is protected from bad contributors.

```
        ┌──────────────────────────────────────────┐
        │   Central server (ours)                  │
        │   • holds the global model               │
        │   • screens incoming updates             │
        │   • aggregates and validates             │
        └───────────────▲──────────────┬───────────┘
                        │              │
        every 14 days   │              │  new global model
        weights only    │              ▼
        ┌───────────────┴──────┐  ┌──────────────────┐
        │ Customer node A      │  │ Customer node B  │  ...
        │ Jetson Orin          │  │ Jetson Orin      │
        │  ├ camera (wired)    │  │  ├ camera        │
        │  └ camera (wired)    │  │  └ camera        │
        └──────────────────────┘  └──────────────────┘
```

Cameras are wired to their node. Nodes talk only to the central server.
Nodes never talk to each other, and no node ever sees another's data.

---

## 1. What crosses the network — and what never does

This is the claim the whole project rests on, so it is worth being exact.

| Leaves the node | Never leaves the node |
|-----------------|-----------------------|
| Model weight tensors | Camera frames |
| A count of local training samples | Recorded clips |
| Loss and accuracy scalars | Face images |
| A node id | Face embeddings |
| | The known-faces database |

`ArgusFlowerClient.fit()` returns exactly three things: updated weights,
`num_samples`, and a metrics dict. There is no code path that transmits
an image.

### Face recognition is not federated at all

Worth stating plainly, because it is stronger than "federated":

**Face identity never participates in federated learning.** ArcFace
embeddings live in each node's local SQLite database and are used only
for matching on that node. They are not in the model that federates, so
they are not in any update, so they never reach the central server or
any other node.

The model that *does* federate is the object detector — it learns what a
person, vehicle, or animal looks like in general. Nothing in it encodes
who anyone is.

If face recognition is ever federated, the rule is that only the 512-d
normalized embedding may participate, never a frame or a crop. An
embedding is not reversible into a photograph, but it *is* biometric
data, and it should be treated as such rather than as a number.

### One honest caveat about local storage

Federated learning means data does not leave the node. It does not mean
data is not stored. `LocalDetectionDataset` writes 128×128 detection
crops to `data/training/` on the node so it has something to fine-tune
against, and the event recorder writes clips to `media/events/`.

Both stay on the customer's own hardware. But "we never store your
footage" would be false, and a judge asking where training data comes
from deserves the accurate answer: *it is stored locally, on the
customer's device, and only weights are shared.*

---

## 2. Protecting the global model

Plain FedAvg averages every update it receives. One node that is broken,
diverged, or hostile therefore moves the global model — and that damaged
model is pushed back to every other node, so a single bad contributor
degrades the whole fleet and the next round starts from the damage.

`src/federated/robust.py` screens updates before averaging. Four filters,
cheapest first, each defending a different failure:

| # | Filter | Catches |
|---|--------|---------|
| 1 | **Structural** | Wrong tensor shapes, NaN, Inf, zero-sample claims |
| 2 | **Norm** | Updates far larger than peers — scaled or "model replacement" attacks |
| 3 | **Direction** | Updates pointing away from consensus — sign-flip, label-flip, targeted poisoning |
| 4 | **Trimmed mean** | Residual per-coordinate outliers |

Then a **validation gate** refuses to promote a candidate that scores
worse than the model it would replace.

### Why deltas, not weights

Screening operates on `client_weights - global_weights`, not raw weights.
A delta is what a node actually learned; raw weights are dominated by the
shared starting point, so two updates that disagree completely still look
nearly identical when compared directly. Comparing deltas is what lets
the direction filter see disagreement at all.

### Why the median, not the mean

The consensus direction is the **coordinate-wise median** of the deltas,
because the mean is precisely what an attacker is trying to drag. A
median of `n` values tolerates up to `(n-1)/2` arbitrary corruptions; a
mean tolerates none.

### The honest limit

**Consensus filtering needs a majority to be honest, so it needs at least
three contributors.** With two nodes that disagree, there is no
principled way to say which one is wrong — and any system claiming
otherwise is guessing.

Below three contributors the aggregator says so in its report, skips
filters 3 and 4, and relies on the structural and norm checks plus the
validation gate. Those are absolute rather than relative, so they still
work with a single contributor.

This matters for the demo: **Orin + Pi is two nodes.** The full
Byzantine defence engages at three. Say that rather than implying the
two-node demo is running the complete algorithm.

### The report

Every round produces an auditable record:

```json
{
  "accepted": ["node-a", "node-b", "node-c"],
  "rejected": ["node-d"],
  "consensus_applied": true,
  "median_norm": 0.0431,
  "note": "trimmed mean over 3 updates",
  "detail": [
    {"node_id": "node-d", "accepted": false,
     "reason": "direction disagrees with consensus (cos=-0.812)",
     "norm": 0.0455, "cosine": -0.812}
  ]
}
```

"We reject bad models" is a claim. This is evidence.

---

## 3. The two-week cycle

1. **Day 0** — each node holds the current global model and runs
   inference. Detections accumulate local training crops.
2. **Days 1–13** — nodes train locally when idle. Nothing is transmitted.
3. **Day 14, 02:00 local** — each node connects to the central server and
   sends weights only.
4. **Aggregation** — the server screens every update, averages the
   survivors, and validates the candidate against a held-out set.
5. **Promotion** — if the candidate holds up, it becomes the new global
   model and is distributed. If not, **the previous model is kept** and
   the round is logged as rejected.
6. Repeat.

Schedule details, including why the interval is persisted to disk, are in
[`JETSON_MIGRATION.md`](JETSON_MIGRATION.md) section 5. The short version:
with an in-memory timer, any reboot inside a 14-day window restarts the
countdown, and a node rebooted weekly would run FL *never* while looking
perfectly healthy.

A failed round does not advance the schedule, so an unreachable server is
retried rather than deferred another fortnight.

---

## 4. Requirements for participating nodes

- **Identical architecture.** Weights are matched by shape and order, so
  every node must run the same model. A mismatch is rejected by filter 1
  rather than crashing aggregation.
- **PyTorch weights.** FL exchanges `.pt` tensors. On a Jetson the
  TensorRT `.engine` is an inference artifact **rebuilt from the updated
  `.pt` after each round** — engines are not portable and cannot be
  federated.
- **Idle before training.** Rounds run only when no motion has been seen
  for 10 minutes, so training never competes with live surveillance.

---

## 5. What is not built yet

Stated because a judge will ask, and because
`docs/SHOWCASE_SPRINT_PLAN.md` A-9 asks for exactly this honesty:

- **No update signing.** A node is identified by a string it chooses.
  Nothing cryptographically binds an update to a node, so the defence is
  against *bad* updates, not against a forged identity. Client
  certificates are the fix.
- **No secure aggregation.** The server sees each node's weights
  individually. Gradient-inversion research shows updates can leak
  information about training data under some conditions. Secure
  aggregation, where the server sees only the sum, is the mitigation.
- **No differential privacy.** No calibrated noise is added, so there is
  no formal privacy guarantee — only the structural one that raw data
  does not move.
- **The validation set is not defined.** `validate_candidate()` is
  implemented and tested; choosing the held-out set it scores against is
  still open, and the gate is only as good as that set.
- **Robust aggregation is not yet wired into the Flower server.**
  `src/federated/robust.py` is complete and tested standalone;
  `central_server/fl_aggregator.py` still runs stock `FedAvg`.

The first three are the standard next steps for a production federated
system, and they are the right content for the roadmap slide.
