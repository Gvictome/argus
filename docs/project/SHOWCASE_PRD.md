# ARGUS Senior Design Showcase PRD

**Status:** Proposed
**Author:** Giovanny (Computer Engineer, AI/ML)
**Date:** 2026-08-12
**Showcase date:** 2026-11-22 (102 days / ~14.5 weeks out)

---

## 1. Summary

ARGUS is a working edge-AI home security system. The backend, detection pipeline, face
recognition, and federated learning components exist and run on a Raspberry Pi 5 with a
Hailo-8L AI HAT+. What does not exist is a **demo**: a repeatable, controlled interaction
that a judge can experience in under two minutes and remember afterward.

This document defines that demo, the minimum work required to support it, and a phased
schedule that keeps the system demo-ready from mid-September onward rather than betting
everything on an integration sprint in November.

The organizing principle: **be demo-ready early and never drop below demo-ready.**

---

## 2. Goals

| # | Goal | Measure |
|---|------|---------|
| G1 | A judge with no prior context understands what ARGUS does within 30 seconds | Dry-run feedback from a non-team observer |
| G2 | The headline demo runs end to end without operator intervention or recovery | 10 consecutive successful runs during Phase 4 |
| G3 | The system is demo-ready by 2026-09-13 and stays that way | Frozen build boots and passes the demo script on demand |
| G4 | Federated learning is communicated as the project's technical differentiator | Dedicated poster section plus a pre-baked narrated segment |
| G5 | No single point of failure ends the demo | Backup video, spare SD card, spare PSU, offline operation |

---

## 3. Non-goals

These are explicitly out of scope for the showcase. They may be valuable to the product;
they are not visible to a judge and they do not make the demo work.

- Cloudflare Tunnel / remote access
- AES-256 encryption completion
- Jetson Nano migration (this is EGN 2 roadmap material, presented as future work)
- Home Assistant integration
- Live federated weight aggregation performed at the booth
- Any feature added after the 2026-11-08 freeze

JWT authentication is a special case: it is a legitimate security gap and belongs on the
roadmap slide, but the demo runs on a single trusted local device and does not require it.

---

## 4. The demo script

This is the core artifact of this document. Everything in Section 5 exists to serve it.

**Setup:** Pi 5 in the HighPi Pro 5S case, Camera Module 3 aimed at the front of the table,
one monitor showing the ARGUS page fullscreen. No internet connection.

| Step | Duration | What the judge sees | What the operator does |
|------|----------|---------------------|------------------------|
| 1 | 0:00-0:15 | Live video with green boxes on people, an FPS counter, and an "on-device / no cloud" indicator | Nothing. It is already running. |
| 2 | 0:15-0:30 | Their own face boxed in red, labeled `UNKNOWN` | "Right now it has never seen you." |
| 3 | 0:30-0:50 | Operator types the judge's first name, clicks Enroll. Confirmation appears, and the known-faces count increments. | Enroll |
| 4 | 0:50-1:10 | Judge steps out of frame and back in. Box turns green, labeled with their name and a confidence score. | Nothing. |
| 5 | 1:10-1:30 | Event feed shows the sequence: `unknown person detected`, `face enrolled`, `known person: <name>` | Point at the feed. |
| 6 | 1:30-2:00 | Optional second segment: the federated learning narration | Switch tabs |

**Failure recovery:** if step 4 does not resolve within 5 seconds, the operator re-enrolls
once and continues. If it fails twice, the operator moves to the backup video without
comment. This is a rehearsed behavior, not an improvisation.

---

## 5. Requirements

### P0 - required for the demo to exist

| ID | Requirement | Owner | Current state |
|----|-------------|-------|---------------|
| P0-1 | Face recognition source committed and pushed to `main` | Giovanny | `src/detection/face_recognition.py` is **untracked**; `src/api/app.py`, `src/api/routes.py`, `src/detection/__init__.py` modified and uncommitted since ~2026-05-03 |
| P0-2 | Annotated MJPEG stream: detection boxes, labels, confidence drawn server-side | Giovanny | `/api/camera/stream` exists but serves raw frames without overlay |
| P0-3 | Single-page demo UI served by FastAPI as a static file | Giovanny + Adam | Does not exist |
| P0-4 | Enroll-from-live-frame endpoint | Giovanny | **Done in code.** `POST /api/faces` captures via `camera_service.get_frame()`, decodes, and calls `recognizer.enroll_face()`. Needs hardware verification only. |
| P0-5 | Event feed backed by real data | Mohammed | **Stub.** `GET /api/events` is `# TODO: Query events from database` and returns an empty list. Real work: write detection events to SQLite, then read them back. |
| P0-6 | Demo page auto-starts on boot (systemd unit), no terminal needed | Christian + Mohammed | Not configured |

**Deliberate constraint on P0-3:** this is **one HTML file with inline CSS and vanilla JS**,
served from FastAPI's static mount. No Next.js, no npm, no build step. The reason is
operational: at a booth table with no internet, a build toolchain is a liability. The Next.js
dashboard is Phase 3 work and does not replace this page as the demo surface.

### P1 - required for the demo to survive a real showcase

| ID | Requirement | Owner |
|----|-------------|-------|
| P1-1 | Fully offline operation, verified with wifi and ethernet physically disconnected | Christian |
| P1-2 | Two-hour continuous thermal soak with no throttle-induced FPS collapse | Christian |
| P1-3 | Enrollment and recognition validated under harsh overhead fluorescent lighting | Giovanny |
| P1-4 | Recognition validated on 5+ faces belonging to people outside the team | Giovanny |
| P1-5 | Similarity threshold tuned and justified (currently `0.4` in `FaceRecognitionService.__init__`) | Giovanny |
| P1-6 | Known-faces database resettable in one click between judges | Giovanny |
| P1-7 | Frozen `demo` branch plus a byte-imaged SD card of the last known-good build | Mohammed |
| P1-8 | Backup video recording of a successful full demo run | Adam |

**P1-6 is not optional.** Over a multi-hour showcase you may enroll dozens of people. An
unbounded and unpruned face database will degrade match quality and eventually produce a
false positive in front of a judge, which is worse than a miss.

### P2 - depth, built only after P0 and P1 hold

| ID | Requirement | Owner |
|----|-------------|-------|
| P2-1 | Next.js dashboard (multi-view, historical events, device panel) | Adam + Mohammed |
| P2-2 | WebSocket endpoint for live event push | Mohammed |
| P2-3 | Event bus / pub-sub | Mohammed |
| P2-4 | Telegram alert on unknown-person detection | Mohammed |
| P2-5 | Remaining API stubs wired to the database (devices, automations) | Mohammed |
| P2-6 | Federated learning demo segment: scripted two-node run, narrated, with the Phase 1 result (88.86% eval accuracy on CIFAR-10, Flower 1.29) | Giovanny |
| P2-7 | Poster, slides, EGN 2 roadmap | Adam |

**Rule governing P2:** any P2 change that breaks the Section 4 demo script is reverted, not
debugged in place. The demo path has priority over the feature.

---

## 6. Technical design

### 6.1 Demo page

Served at `GET /demo` from FastAPI's static mount. Four regions:

```
+--------------------------------------------------+
|  ARGUS            [ON-DEVICE]  [24 FPS]  [HAILO]  |
+---------------------------------+----------------+
|                                 |  KNOWN FACES   |
|      annotated MJPEG stream     |  - Giovanny    |
|      (boxes + name labels)      |  - Christian   |
|                                 |                |
|                                 |  [name______]  |
|                                 |  [  ENROLL  ]  |
|                                 |  [  RESET   ]  |
+---------------------------------+----------------+
|  EVENT FEED                                      |
|  14:02:11  known person: Giovanny (0.71)         |
|  14:02:04  face enrolled: Giovanny               |
|  14:01:58  unknown person detected               |
+--------------------------------------------------+
```

The stream is annotated **server-side** in the existing `process_frame()` cascade
(`src/detection/__init__.py`: motion -> YOLO -> ArcFace). Drawing boxes in the browser
would require a parallel metadata channel and would drift out of sync with the frames.
Server-side annotation keeps the page trivially simple and guarantees the label always
matches the frame it is drawn on.

The event feed polls `GET /api/events?limit=10` every 2 seconds. Polling, not WebSocket.
A WebSocket that drops mid-demo is silent and confusing; a poll that fails retries on its
own two seconds later. WebSocket arrives in P2 for the Next.js dashboard where it belongs.

### 6.2 Endpoints

| Method | Path | Purpose | Status |
|--------|------|---------|--------|
| GET | `/demo` | Static demo page | New |
| GET | `/api/camera/stream` | Annotated MJPEG | Modify (add overlay) |
| GET | `/api/faces` | List known faces | Exists |
| POST | `/api/faces` | Enroll from current live frame | Verify / modify |
| DELETE | `/api/faces/{face_id}` | Remove one face | Exists |
| POST | `/api/faces/reset` | Clear all enrolled faces | New (P1-6) |
| GET | `/api/events` | Recent events | Verify wired to DB |
| GET | `/api/detection/status` | FPS, backend, model | Exists, may need FPS field |

### 6.3 Freeze mechanism

- `main` receives all development.
- `demo` is cut from `main` only after the full Section 4 script passes on the physical unit.
- The showcase Pi runs **only** `demo`.
- After 2026-11-08, `demo` accepts fixes to the demo path and nothing else.

---

## 7. Timeline

| Phase | Dates | Focus | Exit criterion |
|-------|-------|-------|----------------|
| 0 | Aug 12 - Aug 16 | Commit and push face recognition. Tag baseline. | Nothing critical exists only on one machine |
| 1 | Aug 17 - Sep 13 | All P0 items | **The Section 4 script runs end to end on the Pi** |
| 2 | Sep 14 - Oct 4 | All P1 items. Dry run #1 with a non-team observer, ideally the sponsor. | Demo survives bad lighting, no network, 2h runtime, strangers' faces |
| 3 | Oct 5 - Nov 1 | P2 depth. Dashboard, alerts, FL segment. | P2 built without ever breaking the P0 path |
| 4 | Nov 2 - Nov 8 | Full dress dry run #2. **Hard feature freeze Nov 8.** | 10 consecutive clean demo runs |
| 5 | Nov 9 - Nov 22 | Zero new code. Rehearsal, poster, slides, backups. | Every team member can run the demo solo |

Fall term begins inside Phase 1. Phase 1 is scoped at four weeks for roughly two weeks of
actual work to account for that.

Two weeks of freeze appears wasteful. It is the difference between a team that presents and
a team that apologizes.

---

## 8. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Face recognition work lost (uncommitted, single machine, ~3 months) | Project-ending | P0-1, this week |
| Recognition fails on an unfamiliar face at the booth | Judge sees the headline feature miss | P1-3, P1-4, P1-5; rehearsed two-strike recovery |
| Thermal throttling after hours of runtime | FPS collapse mid-demo | P1-2 soak test; active cooler is already spec'd |
| Booth wifi absent or hostile | Total demo failure if anything depends on it | P1-1, tested with cables physically pulled |
| False positive from an overgrown face database | Worse than a miss; undermines credibility | P1-6 reset between judges |
| Phase 3 dashboard work destabilizes the demo path | Loss of demo-ready status late | Frozen `demo` branch; revert-not-debug rule |
| Team availability during term | Schedule slip | Phase 1 buffer; P2 is explicitly cuttable |

---

## 9. Open questions

1. Is the showcase judged, and against what rubric? A rubric would change the weighting
   between demo polish and documented engineering rigor.
2. Does the venue provide a monitor and power, or is that on the team?
3. Owner assignments in Section 5 are proposed, not confirmed with the team.
4. Does the detection pipeline currently write events to SQLite at all, or only expose them
   in-process? Determines whether P0-5 is a read-path fix or a write-path fix as well.
5. `README.md` lists Saifeddine's focus as Federated Learning, but his actual contributions
   have been frontend architecture, documentation, and security audits. Confirm lanes before
   assigning Phase 3 dashboard work.

---

## 10. Appendix: current state as of 2026-08-12

Repository: `C:\Users\colab\prometheus_workspace\argus`, branch `main`.
Last commit `70703ce` (2026-04-11), "Unified Repository: Combined argus and argus-backend
into a single codebase."

Uncommitted working tree:

```
 M src/api/app.py
 M src/api/routes.py
 M src/detection/__init__.py
?? src/detection/face_recognition.py
```

Built and working: camera service (picamera2 + OpenCV fallback, MJPEG, recording), error
routing agents, YOLOv8 object detection with Hailo, YAML/env config, SQLite database layer
(6 tables), ArcFace face recognition via InsightFace (512-d embeddings), cascaded detection
pipeline, Flower federated client/server with Phase 1 validated at 88.86%, model manager for
weight extract/inject/delta, PBKDF2 hashing and token auth.

Not built: JWT, AES-256, WebSocket, event bus, several API stubs, alerts, Next.js dashboard,
Cloudflare Tunnel, Jetson migration.
