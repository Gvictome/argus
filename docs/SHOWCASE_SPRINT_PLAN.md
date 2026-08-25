# ARGUS Showcase Sprint Plan

**Status:** Proposed - owner assignments not yet confirmed with the team
**Companion to:** `docs/SHOWCASE_PRD.md`
**Drafted:** 2026-08-12
**Showcase:** 2026-11-22

---

## 1. How this works

Seven two-week sprints, Monday start, Sunday end. The calendar cooperates: Aug 17 is a
Monday, and seven sprints from there land exactly on Nov 22. Both hard gates fall on sprint
boundaries without adjustment.

### Cadence

| Ceremony | When | Duration | Who |
|----------|------|----------|-----|
| Sprint planning | Monday of week 1, start of sprint | 45 min | All four |
| Async standup | Tuesday + Friday, written in the group chat | 5 min each | All four |
| Sprint review | Sunday of week 2 | 30 min | All four |
| Retro | Immediately after review, sprints 2, 4, 6 only | 15 min | All four |

Async standup answers three things: what I finished, what I am doing next, what is blocking
me. Written, not verbal. Four people with class schedules will not reliably meet twice a
week, and a written trail means the blocked person gets unblocked without a meeting.

### The rule that makes this work

**Every sprint review ends by running the full demo script from PRD Section 4 on the actual
Pi.** Not a code walkthrough, not a screen share of localhost. The physical unit, start to
finish, with someone playing the judge.

From Sprint 2 onward this is expected to pass. When it fails, fixing it becomes the top
item of the next sprint ahead of any planned work. This single rule is what prevents the
week-13 integration disaster, because integration is exercised 7 times instead of once.

### Definition of Done

A task is not done until all five hold:

1. Code is committed and pushed to `origin/main`
2. It runs on the Raspberry Pi, not only on a laptop
3. It does not break the PRD Section 4 demo script
4. Another team member has seen it work
5. If it changes setup or operation, `docs/` is updated in the same commit

Item 2 exists because "works on my machine" and "works on ARM64 with a Hailo HAT" are
different claims. Item 4 exists because a demo only one person can run is a demo that fails
when that person is talking to a judge.

---

## 2. Sprint calendar and gates

| Sprint | Dates | Theme | Gate at sprint end |
|--------|-------|-------|--------------------|
| 0 | Aug 12 - Aug 16 | Stop the bleeding | All local work committed and pushed |
| 1 | Aug 17 - Aug 30 | Demo skeleton | Annotated stream visible in a browser |
| 2 | Aug 31 - Sep 13 | **Demo exists** | **GATE A: full demo script runs end to end** |
| 3 | Sep 14 - Sep 27 | Hardening I | Runs fully offline, survives 2h thermal soak |
| 4 | Sep 28 - Oct 11 | Hardening II | **Dry Run #1 with a non-team observer** |
| 5 | Oct 12 - Oct 25 | Depth | Dashboard and alerts exist, demo still passes |
| 6 | Oct 26 - Nov 8 | Polish + freeze | **GATE B: dry run #2, then HARD FREEZE** |
| 7 | Nov 9 - Nov 22 | Rehearsal | All four can run the demo solo |

**Gate A (Sep 13)** is the important one. If the demo script does not run end to end on
Sep 13, Sprint 3 is cancelled and re-run as Sprint 2 extended. Hardening something that
does not work yet is wasted effort, and every later sprint depends on Gate A holding.

**Gate B (Nov 8)** is a hard stop on new code. After this date the only permitted commits
are fixes to the demo path itself.

---

## 3. Individual lanes and deadlines

Owner assignments below are proposed based on demonstrated contribution, not confirmed.
Section 7 lists the conflicts to resolve at Sprint 0 planning.

### 3.1 Giovanny - AI/ML, detection, face recognition

Owns the headline demo feature. Everything a judge remembers runs through this lane.

| # | Deliverable | PRD ref | Due | Sprint |
|---|-------------|---------|-----|--------|
| G-1 | Commit and push `face_recognition.py` + 3 modified files; tag `v0.1-baseline` | P0-1 | **Aug 16** | 0 |
| G-2 | Server-side box/label/confidence overlay drawn in `process_frame()` | P0-2 | Aug 30 | 1 |
| G-3 | Verify `POST /api/faces` live-frame enrollment on the Pi | P0-4 | Aug 30 | 1 |
| G-4 | `POST /api/faces/reset` endpoint | P1-6 | Sep 13 | 2 |
| G-5 | FPS field exposed on `/api/detection/status` | P0-2 | Sep 13 | 2 |
| G-6 | Similarity threshold tuned and justified in writing (currently `0.4`) | P1-5 | Sep 27 | 3 |
| G-7 | Enrollment validated under harsh overhead fluorescent light | P1-3 | Sep 27 | 3 |
| G-8 | Recognition validated on 5+ non-team faces | P1-4 | Oct 11 | 4 |
| G-9 | Federated learning demo segment, scripted and narrated | P2-6 | Oct 25 | 5 |
| G-10 | FL results slide: 88.86% CIFAR-10, Flower 1.29, privacy argument | P2-6 | Nov 8 | 6 |

**G-1 is due first for a reason.** Roughly three months of the project's headline feature
currently exists on exactly one machine, untracked by git. Every other deadline in this
document is contingent on that work continuing to exist.

**G-6 is the most technically interesting item.** The threshold governs the false accept /
false reject tradeoff, and at a showcase these fail differently: a rejection is a shrug, an
acceptance of the wrong person in front of a judge is a credibility problem. Tune toward
rejection and document why. That justification is also a good answer to the hardest
question a judge is likely to ask.

### 3.2 Mohammed - Backend, API, data

Owns everything behind the demo page and the release mechanics.

| # | Deliverable | PRD ref | Due | Sprint |
|---|-------------|---------|-----|--------|
| M-1 | Detection pipeline writes events to SQLite (write path) | P0-5 | Aug 30 | 1 |
| M-2 | `GET /api/events` reads from SQLite, replacing the stub | P0-5 | Sep 13 | 2 |
| M-3 | systemd unit: ARGUS starts on boot, no terminal required | P0-6 | Sep 13 | 2 |
| M-4 | `demo` branch cut from `main` after Gate A passes | P1-7 | Sep 13 | 2 |
| M-5 | Byte image of the known-good SD card, stored off the device | P1-7 | Sep 27 | 3 |
| M-6 | Boot-to-demo verified from cold power-on, under 90 seconds | P0-6 | Sep 27 | 3 |
| M-7 | WebSocket endpoint for live event push | P2-2 | Oct 25 | 5 |
| M-8 | Event bus / pub-sub | P2-3 | Oct 25 | 5 |
| M-9 | Telegram alert on unknown-person detection | P2-4 | Nov 8 | 6 |
| M-10 | Remaining API stubs wired to DB (devices, automations) | P2-5 | Nov 8 | 6 |

**M-1 and M-2 are two separate jobs, not one.** `GET /api/events` is currently
`# TODO: Query events from database` returning an empty list, but the deeper question is
whether the detection pipeline writes events at all. If it does not, the read-path fix
returns an empty list just as convincingly. Verify the write path first, in Sprint 1.

**M-5 protects the whole project.** An SD card that has been power-cycled at a booth for
six hours is a consumable. The image is the difference between a 10-minute recovery and a
dead table.

### 3.3 Christian - Hardware, power, thermal, sensors

Owns the physical unit and every environmental failure mode. This lane is under-weighted in
software planning and over-represented in showcase disasters.

| # | Deliverable | PRD ref | Due | Sprint |
|---|-------------|---------|-----|--------|
| C-1 | Confirm camera framing and mount height for a standing adult at table distance | P0-2 | Aug 30 | 1 |
| C-2 | PIR sensor + frame-diff motion integration finished, or formally descoped | - | Sep 13 | 2 |
| C-3 | Offline operation verified with wifi and ethernet physically disconnected | P1-1 | Sep 27 | 3 |
| C-4 | Two-hour continuous thermal soak, FPS logged every 5 min | P1-2 | Sep 27 | 3 |
| C-5 | Booth kit assembled: PSU, spare PSU, cables, monitor adapters, SD cards | P1-7 | Oct 11 | 4 |
| C-6 | Portable lighting solution for harsh or dim venue lighting | P1-3 | Oct 11 | 4 |
| C-7 | Enclosure and cable management presentable at table height | - | Oct 25 | 5 |
| C-8 | Power budget documented for the poster (draw, thermals, headroom) | - | Nov 8 | 6 |

**C-2 needs a decision, not a slip.** Motion detection hardware has been in progress since
roughly May. It is not on the demo path, so if it is not landing by Sep 13 it should be
formally moved to the EGN 2 roadmap slide rather than quietly consuming Christian's Sprint 3,
which is where the thermal and offline work lives.

**C-4 produces a number for the poster.** "Sustained 24 FPS across a two-hour soak with a
peak SoC temperature of X degrees" is exactly the kind of measured claim that separates a
senior design project from a demo.

### 3.4 Saifeddine / Adam - Frontend, documentation, presentation

Owns everything the judge looks at other than the video feed.

| # | Deliverable | PRD ref | Due | Sprint |
|---|-------------|---------|-----|--------|
| A-1 | Demo page HTML/CSS shell: 4 regions, fullscreen, readable at 3 feet | P0-3 | Aug 30 | 1 |
| A-2 | Demo page JS: enroll form, 2s event-feed poll, known-faces list | P0-3 | Sep 13 | 2 |
| A-3 | Status indicators: ON-DEVICE badge, FPS, HAILO badge | P0-3 | Sep 13 | 2 |
| A-4 | Backup video of a complete successful demo run | P1-8 | Oct 11 | 4 |
| A-5 | Dry Run #1 coordination: recruit a non-team observer, capture feedback | - | Oct 11 | 4 |
| A-6 | Next.js dashboard (secondary surface, never the demo surface) | P2-1 | Oct 25 | 5 |
| A-7 | Poster: architecture, FL story, measured results | P2-7 | Nov 8 | 6 |
| A-8 | Slide deck updated from the EGN 1 v3 base | P2-7 | Nov 8 | 6 |
| A-9 | Security audit writeup: what is implemented, what is roadmap, honestly | - | Nov 8 | 6 |

**A-1 has one constraint that overrides taste: it is a single HTML file with inline CSS and
vanilla JS.** No build step, no npm, no bundler. This is not a judgment about Next.js; it is
that a build toolchain at a booth with no internet is a liability with no upside. A-6 is the
real dashboard and is genuinely Next.js.

**A-9 is worth doing well.** JWT and AES-256 are unfinished. A judge who finds that gap is
much better handled by a slide that already names it as roadmap than by a team discovering
it live. Claiming security you do not have is the fastest way to lose a technical judge.

---

## 4. Sprint-by-sprint board

| Sprint | Giovanny | Mohammed | Christian | Adam |
|--------|----------|----------|-----------|------|
| 0 | G-1 | - | - | - |
| 1 | G-2, G-3 | M-1 | C-1 | A-1 |
| 2 | G-4, G-5 | M-2, M-3, M-4 | C-2 | A-2, A-3 |
| 3 | G-6, G-7 | M-5, M-6 | C-3, C-4 | - |
| 4 | G-8 | - | C-5, C-6 | A-4, A-5 |
| 5 | G-9 | M-7, M-8 | C-7 | A-6 |
| 6 | G-10 | M-9, M-10 | C-8 | A-7, A-8, A-9 |
| 7 | rehearsal | rehearsal | rehearsal | rehearsal |

Sprint 3 has no Adam items and Sprint 4 has no Mohammed items. That is deliberate slack, not
an oversight. Both land during the semester when coursework peaks, and a plan with no slack
is a plan that breaks on first contact with a midterm.

---

## 5. Cross-person dependencies

These are the handoffs where one person's slip silently blocks another. Each has a named
date by which the upstream side must deliver.

| Blocker | Blocks | Handoff by | Risk if late |
|---------|--------|------------|--------------|
| G-2 annotated stream | A-1 demo page layout | Aug 24 | Adam builds a page around an unknown stream format |
| M-1 event write path | M-2 read path, A-2 event feed | Aug 30 | Feed ships empty and looks broken at Gate A |
| G-3 enroll verified | A-2 enroll form wiring | Aug 30 | Adam wires a form to an endpoint nobody has run on the Pi |
| C-1 camera framing | G-7 lighting validation | Aug 30 | Lighting tuned for the wrong framing, redone in Sprint 4 |
| Gate A pass | Everything in Sprints 3-7 | **Sep 13** | Entire back half of the schedule shifts |
| M-4 demo branch | M-5 SD image | Sep 13 | No frozen artifact to image |
| G-8 stranger faces | A-4 backup video | Oct 11 | Backup video records a demo that only works on team faces |

---

## 6. Escalation rules

1. **Blocked more than 48 hours: say so in the group chat.** Not at the next standup.
2. **Gate A fails on Sep 13:** Sprint 3 is cancelled. Sprint 2 work continues into Sprint 3.
   Hardening a non-functional demo is wasted effort.
3. **A P2 item breaks the demo script:** revert it. Do not debug it in place. The demo path
   outranks the feature, always.
4. **Anything slips past Nov 8:** it is cut, not extended. The freeze is the deliverable.
5. **Two people idle in the same sprint:** pull work forward from the next sprint rather
   than adding scope. Finishing early is the goal; finding more to build is not.

---

## 7. Open items for Sprint 0 planning

1. **Confirm the lanes above with the team.** These are proposed from observed contribution
   and have not been agreed to by anyone but the author.
2. **Resolve the Saifeddine role conflict.** `README.md` lists his focus as Federated
   Learning; his actual contributions have been frontend architecture, documentation, and
   security audits. This plan assumes the latter. Either fix the README or reassign A-6.
3. **Confirm Nov 22 is the showcase itself.** Nov 22, 2026 is a Sunday, which is unusual for
   a campus showcase. If it is actually a submission deadline with the event on a different
   day, Sprint 7 changes shape.
4. **Confirm venue provides monitor and power**, or C-5 grows.
5. **Is the showcase judged against a rubric?** A rubric would change the balance between
   demo polish and documented engineering rigor, and therefore the weighting of Sprint 6.
6. **Agree on the sprint review time slot** before Sprint 1 starts. A recurring calendar hold
   that everyone already declined is not a cadence.
