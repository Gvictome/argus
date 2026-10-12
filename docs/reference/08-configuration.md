# 8 · Configuration reference

All node settings live in `src/config.py` (`Settings`) and are read from
environment variables once, at import, by `Settings.from_env()`. Booleans
are the string `true` or `false`. There is no config file for the node.

`scripts/run_node.sh` overrides several defaults; its value is shown where
it differs.

---

## 8.1 Server

| Variable | Default | |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | |
| `DEBUG` | `false` | `true` turns on uvicorn reload and INFO logging. Otherwise uvicorn logs at WARNING, which is why important problems are logged as warnings |
| `SECRET_KEY` | `dev-secret-change-in-production` | Set a real value anywhere beyond a bench |
| `CORS_ORIGINS` | empty (`run_node.sh`: localhost and 127.0.0.1 on 3000 and 3001) | Comma-separated. Only needed for a separately hosted frontend |
| `ARGUS_DATA_DIR` | `data` | Moves the database, samples, head and FL state together. Relative paths are from the repo root |

## 8.2 Camera

| Variable | Default | |
|---|---|---|
| `CAMERA_INDEX` | `0` | Sensor for the single-camera service |
| `CAMERA_SENSORS` | empty | `"0:front,1:back"` for several cameras (Jetson Orin) |
| `CAMERA_SOURCE` | empty | A video file or numeric index; overrides the board camera |
| `CAMERA_SOURCE_REALTIME` | `true` | Pace a file at its own FPS; `false` reads as fast as possible, for benchmarks |
| `CAMERA_SWAP_RB` | `false` | Only if people look blue |

Resolution (1920×1080), camera FPS (30) and rotation (0) are dataclass
defaults in `Settings` with no environment variable.

## 8.3 Detection

| Variable | Default | |
|---|---|---|
| `OBJECT_MODEL` | `yolov8n` | `yolov8s`, `yolov8m`... An accelerated backend needs an export with the same name |
| `OBJECT_BACKEND` | `auto` | `auto`, `pt`, `openvino`, `ncnn`. Accelerators (TensorRT, Hailo) are always preferred when their files exist |
| `OBJECT_IMGSZ` | `640` | Ignored, with a warning, when an export fixes its own size |
| `DETECTION_THRESHOLD` | `0.5` | YOLO confidence |
| `MOTION_GATE` | `true` | `false` runs YOLO and threats on every frame |
| `MOTION_SENSITIVITY` | `25` | Pixel-difference threshold |
| `MOTION_WIDTH` | `320` | Width motion detection works at |
| `DETECTION_AUTOSTART` | `true` | Run the background worker from startup |
| `STREAM_MAX_FPS` | `15` | Viewer stream cap |
| `STREAM_WIDTH` | `960` | Viewer stream encode width |

## 8.4 Threat stage

| Variable | Default | |
|---|---|---|
| `THREAT_ENABLED` | `true` (`run_node.sh`: `false`, `--threat` turns it on) | |
| `THREAT_MODEL_PATH` | `models/threat-yolo11s.pt` | |
| `THREAT_CONFIDENCE` | `0.70` | Chosen by sweep, section 4.3 |
| `THREAT_IMGSZ` | `416` | |

## 8.5 Recording

| Variable | Default | |
|---|---|---|
| `RECORD_EVENTS` | `true` | |
| `RECORD_PRE_ROLL_S` | `5` | |
| `RECORD_POST_ROLL_S` | `10` | |
| `RECORD_MAX_CLIP_S` | `120` | |

## 8.6 Federated learning (node)

| Variable | Default | |
|---|---|---|
| `NODE_NAME` | `Node A` | Shown on the dashboard and sent to the aggregator as `node_id` |
| `CAMERA_ZONE` | `Front door` | |
| `FL_ENABLED` | `false` | Turns on the scheduler. Capture, labelling and manual rounds work without it |
| `FL_SERVER_URL` | `localhost:8080` | Flower gRPC address of the aggregator |
| `FL_CENTRAL_API` | empty | Aggregator REST, e.g. `http://192.168.1.50:8090`; enables node registration |
| `FL_ROUND_INTERVAL_DAYS` | `14` | |
| `FL_ROUND_HOUR` | `2` | |
| `FL_LOCAL_EPOCHS` | `5` | |
| `FL_MIN_SAMPLES` | `50` | |
| `ARGUS_TRAIN_THREADS` | `1` | BLAS threads for the training subprocess |

## 8.7 Security

| Variable | Default | |
|---|---|---|
| `AUTH_REQUIRED` | `false` | Enforce bearer tokens on protected routes. See the gap in section 6.4 |
| `ARGUS_ADMIN_PASSWORD` | unset | Used only when the admin account is first created; otherwise a password is generated and printed once |

## 8.8 Deprecated

| Variable | Default | |
|---|---|---|
| `FACE_RECOGNITION_ENABLED` | `false` | |
| `FACE_RECOGNITION_THRESHOLD` | `0.6` | |
| `FACE_SIMILARITY_THRESHOLD` | `0.4` | |

---

## 8.9 The aggregator (`central_server/fl_aggregator.py`)

Settings are layered: built-in defaults, then the config file
`central_server/central_config.json` (or the path in `ARGUS_CONFIG`), then
environment variables.

| Setting | Code default | Shipped config file | Environment override |
|---|---|---|---|
| Flower address | `0.0.0.0:8080` | `0.0.0.0:8080` | `ARGUS_FL_ADDRESS` |
| REST port | 8090 | 8090 | none |
| Clients per round | 2 | **1** | `ARGUS_MIN_CLIENTS` (`run_central_server.ps1 -MinClients`, default 1) |
| Rounds per session | 50 | **5** | `ARGUS_ROUNDS` (`-Rounds`, default 5) |
| Local epochs sent to clients | 3 | **6** | none |
| Auth token | empty | empty | `ARGUS_AUTH_TOKEN` (`-Token`) |
| Aggregation | `robust` | `robust` | `ARGUS_AGGREGATION` (`-Aggregation robust` or `fedavg`) |

`ARGUS_SERVE_FOREVER` (default `true`) starts a new session when one
finishes, so a node on a 14-day schedule still finds the server.

With one client per round the server will happily "federate" a single
node. Set `-MinClients 2` or more when two nodes are actually running.

## 8.10 The legacy server (`server/config.py`)

`FL_PORT`, `NUM_ROUNDS`, `MIN_CLIENTS`, `STRATEGY` (`fedavg` or
`fedprox`), `FEDPROX_MU`. Not used by the current node.
