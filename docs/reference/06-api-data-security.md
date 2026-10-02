# 6 · API, data and security

The node is one FastAPI app (`src/api/app.py`, `create_app()`), served by
uvicorn from `main.py` on port 8000. Interactive docs are at
`http://<node>:8000/docs`.

---

## 6.1 Startup order (`src/api/app.py`, `startup_event`)

The order matters; each step depends on the ones before it.

1. **Database**: `Database(settings.DB_PATH).initialize()`.
2. **Auth**: `initialize_auth(db)` creates the admin account if missing
   (from `ARGUS_ADMIN_PASSWORD`, or a generated password printed **once**
   at WARNING level) and warns if `AUTH_REQUIRED` is off or `SECRET_KEY`
   is the default.
3. **Cameras**: build the registry from `CAMERA_SENSORS`; configure
   `camera_service` from settings. Nothing is opened yet.
4. **Detection**: build `DetectionConfig` from settings and
   `detection_service.initialize()` (picks the backend).
5. **Face recognition**: only if `FACE_RECOGNITION_ENABLED` (deprecated).
6. **Threat classifier**: only if `THREAT_ENABLED`; failure is a warning,
   not a crash.
7. **Event recorder**: only if `RECORD_EVENTS`; finished clips are logged
   to the `events` table.
8. **Federated head, sample store, collector**: always, regardless of
   `FL_ENABLED`, because capturing and labelling is useful before any round
   is scheduled. Then restore the last accepted head.
9. **Detection worker**: started if `DETECTION_AUTOSTART` (default true),
   so detection runs with nobody watching.
10. **FL scheduler**: only if `FL_ENABLED`.

Every optional stage is wrapped so its failure leaves the rest running:
losing training data or the threat stage is recoverable; losing the camera
during a demo is not.

---

## 6.2 Routes

`P` marks routes behind `require_auth`, enforced only when
`AUTH_REQUIRED=true`. Routes without P are open in every mode, including
all federated and labelling routes (see 6.4).

### Health and the built-in page (`routes.py`)

| Method | Path | |
|---|---|---|
| GET | `/` | Service banner |
| GET | `/dashboard` | `static/index.html`, the built-in live page |
| GET | `/api/status` | `{fps, accelerator, model_version, nodes_online, node, samples}` |
| GET | `/api/detection/status` | What actually loaded: backend, model, gate, motion rate, worker and collector stats |

### Camera (`routes.py`)

| Method | Path | |
|---|---|---|
| GET | `/api/camera/status` | |
| POST P | `/api/camera/init`, `/api/camera/shutdown` | |
| GET P | `/api/camera/snapshot` | One JPEG |
| GET P | `/api/camera/stream` | MJPEG with boxes; `?token=` accepted because an `<img>` cannot send headers |
| POST P | `/api/camera/record` | Manual recording |
| GET | `/api/cameras`, `/api/cameras/{name}/status` | Multi-camera registry |
| POST P | `/api/cameras/init`, `/api/cameras/{name}/init`, `/api/cameras/{name}/shutdown` | |
| GET P | `/api/cameras/{name}/snapshot`, `/api/cameras/{name}/stream` | |
| GET P | `/api/recordings`, `/api/clips` | Saved footage |

### Events and labelling (`routes.py`, `dashboard_routes.py`)

| Method | Path | |
|---|---|---|
| GET | `/api/events`, `/api/events/{id}` | Real events from the sample store, in the dashboard's `DetectionEvent` shape; synthetic rows excluded |
| GET P | `/api/events/{id}/snapshot` | The still saved for an event |
| POST | `/api/events/{n}/label` | `{"action": "confirm"}` or `{"action": "correct", "label": "<free text>"}` |
| WS | `/ws/events` | Pushes each new real event, polling the store once a second |
| POST | `/api/login` | Dashboard login, returns `{access_token}` |

How a dashboard label becomes a class (`_resolve_label`):

* **confirm** on a flagged event means `anomaly`; on anything else it means
  that class is routine here.
* **correct** with no text inverts the call.
* **correct** with text reads keywords: "stranger", "suspicious",
  "intruder", "threat" and similar mean `anomaly`; "dog", "van",
  "courier" and similar pick the class; "routine", "neighbour", "false
  alarm" mean routine for the event's class. Free text never 422s, because
  the dashboard's Correct box is free text.

### Federated (`trial_routes.py`, `routes.py`)

| Method | Path | |
|---|---|---|
| GET | `/api/federated/status` | Scheduler state, last and next round |
| GET | `/api/federated/samples` | Recent samples and counts |
| POST | `/api/federated/samples/{id}/label` | `{"label": 0..4}` |
| POST | `/api/federated/samples/bootstrap` | `{"site", "n", "seed"}`, synthetic rows |
| GET | `/api/federated/head` | Head size, version, gate history |
| POST | `/api/federated/head/train` | `{"epochs", "min_samples", "tolerance"}`, train a copy and gate it |
| POST | `/api/federated/predict` | `{"features": [19 numbers]}` |
| POST | `/api/federated/round` | `{"server", "epochs", "min_samples", "tolerance"}`, start a round (409 if one is running) |
| GET | `/api/federated/round` | `{running, result, log}` of the current or last round |

### Auth and logs (`routes.py`)

| Method | Path | |
|---|---|---|
| POST | `/api/auth/login`, `/api/auth/logout` | Bearer tokens |
| GET | `/api/auth/me` | Includes `auth_required`, which `run_tunnel.sh` checks |
| GET P | `/api/logs` | Audit log |

### Legacy and dev routes

| Path | Status |
|---|---|
| `/api/faces` (GET, POST, DELETE `{id}`, POST `reset`) | Face enrollment. Works, but face recognition is deprecated and off |
| `/api/devices`, `/api/automations` | Backed by the in-memory `AutomationService` stub; not used by the demo |
| `/api/errors/analyze`, `/api/errors/fix`, `/api/errors/history`, `/api/agents/status` | The error-routing helper agents (section 7) |

---

## 6.3 The database (`src/database/__init__.py`)

SQLite at `DB_PATH` (default `data/database.db`). `ARGUS_DATA_DIR` moves
the database, the sample store, the head and the FL state together, which
is how `mock_demo.py` isolates itself in `data/mock`.

| Table | Holds |
|---|---|
| `events` | Every emitted detection (`event_type="detection"`) and every saved clip (`"clip"`), with a JSON `data` column and `media_path` |
| `users` | Accounts: id, username, PBKDF2 hash, role |
| `audit_log` | Security-relevant actions |
| `faces` | Face embeddings (deprecated path) |
| `devices`, `automations` | Automation stub |

Training data does **not** live in SQLite. It lives in
`data/training/samples.jsonl` (section 5.4).

### Everything on disk

| Path | What | Safe to delete? |
|---|---|---|
| `data/database.db` | events, users, audit | Loses accounts and history |
| `data/training/samples.jsonl` | every sample and its label | **Loses all labelling work** |
| `data/snapshots/` | event stills and crops | Events lose their pictures |
| `data/fl_head.npz` | the accepted head | Node reverts to an untrained head |
| `data/fl_model_state.json` | version and gate history | Version counter restarts |
| `data/fl_state.json` | last round time | Next round is treated as overdue |
| `data/fl_rounds/<ts>/` | per-round artefacts | Yes |
| `media/events/*.mp4` | event clips | Yes, and **nothing deletes these automatically** |
| `models/` | exports and threat weights | Rebuild with the export and fetch scripts |
| `central_server/checkpoints/` | global model per round | On the aggregator only |

---

## 6.4 Security (`src/security/__init__.py`, `src/api/auth.py`)

| Piece | Implementation |
|---|---|
| Password hashing | PBKDF2-HMAC-SHA256, 100,000 iterations, 16-byte random salt |
| Tokens | `secrets.token_urlsafe(32)`, 24 h expiry, held **in memory**: a restart logs everyone out |
| Enforcement | `require_auth` FastAPI dependency on the routes marked P; `Authorization: Bearer` or `?token=` |
| Off by default | `AUTH_REQUIRED=false` so the LAN demo works without login. Acceptable behind a router only |
| Admin bootstrap | Created on first start; `scripts/set_password.py` resets a password or creates users |
| Remote access | `scripts/run_tunnel.sh` refuses to start unless auth is on and proven to return 401 |
| Encryption at rest | `encrypt` / `decrypt` are **placeholders** (TODO AES-256); nothing is encrypted at rest yet |

**Known gaps, both to fix before anything is exposed:**

* **The federated and dashboard routers carry no auth at all.** Only routes
  in `routes.py` marked P use `require_auth`. Everything in
  `trial_routes.py` (`/api/federated/*`: label, bootstrap, train, predict,
  round) and in `dashboard_routes.py` (`POST /api/events/{n}/label`,
  `WS /ws/events`) is open **even with `AUTH_REQUIRED=true`**. Anyone who
  can reach the port can relabel training data, seed synthetic rows, or
  start a round. `run_tunnel.sh` only probes `/api/faces`, so its
  preflight passes regardless. Fix: add `dependencies=[Depends(require_auth)]`
  to those two `APIRouter()`s and probe one of their routes in the tunnel
  preflight.
* The Next.js dashboard does not send its token to the backend.

---

## 6.5 The built-in dashboard (`static/index.html`)

One HTML file served at `/dashboard`, no build step. It shows the live
annotated stream, detection FPS each second, what is currently in frame,
and the event list with each event's still. It talks only to the node it
is served from, so it works whenever the node does, even when the Next.js
dashboard cannot reach it.

The product dashboard, `adam9798/argus-dashboard` (Next.js), is built
against the contract implemented in `dashboard_routes.py`:
`POST /api/login`, `GET /api/status`, `GET /api/events`,
`POST /api/events/{id}/label`, `WS /ws/events`.
