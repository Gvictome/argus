# Prometheus Operations Manual

**How to Operate the Senior Dev AI Orchestrator**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Starting Prometheus](#starting-prometheus)
4. [Voice Commands](#voice-commands)
5. [Project Management](#project-management)
6. [Agent Control](#agent-control)
7. [Plugin System](#plugin-system)
8. [Task Execution](#task-execution)
9. [Workflows](#workflows)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
pip install SpeechRecognition pyaudio pyyaml

# 2. Navigate to argus
cd "C:\Users\colab\Dropbox\viccia ai\argus"

# 3. Start Prometheus
python -m prometheus.main

# 4. Say "Hey Prometheus" to wake
# 5. Give commands: "Create project heimdall"
```

---

## Installation

### Required Dependencies

```bash
# Core
pip install pyyaml

# Voice (optional but recommended)
pip install SpeechRecognition pyaudio

# For offline voice recognition
pip install openai-whisper

# Text-to-speech on Linux
sudo apt install espeak
```

### Windows-Specific

```powershell
# PyAudio often needs pre-built wheel
pip install pipwin
pipwin install pyaudio
```

### Verify Installation

```bash
python -c "import speech_recognition; print('Voice: OK')"
python -c "import yaml; print('YAML: OK')"
```

---

## Starting Prometheus

### Voice Mode (Default)

```bash
python -m prometheus.main
```

You'll see:
```
╔═══════════════════════════════════════════════════════════╗
║   ██████╗ ██████╗  ██████╗ ███╗   ███╗███████╗████████╗  ║
║   ...                                                     ║
║           🔥 Senior Dev AI Orchestrator v1.0 🔥           ║
╚═══════════════════════════════════════════════════════════╝

🎤 Calibrating microphone...
🎤 Microphone ready
🎤 Prometheus listening... Say 'Hey Prometheus' to wake
```

### Text Mode (No Microphone)

```bash
python -m prometheus.main --no-voice
```

You'll get a command prompt:
```
prometheus> create project heimdall
prometheus> spawn error agent
prometheus> quit
```

### Custom Workspace

```bash
python -m prometheus.main --workspace /path/to/projects
```

---

## Voice Commands

### Wake Words

Prometheus sleeps until you say one of these:
- **"Hey Prometheus"**
- **"Prometheus"**
- **"Senior Dev"**

After waking, speak your command. Prometheus stays awake until you say:
- **"Go to sleep"**
- **"Stop listening"**
- **"Goodbye"**

### Command Structure

Commands are natural language. These all work:

```
"Create a new project called heimdall"
"Create project heimdall"
"New project heimdall"
"Start project heimdall"
```

### Full Command Reference

| Category | Commands |
|----------|----------|
| **Wake** | "Hey Prometheus", "Prometheus", "Senior Dev" |
| **Sleep** | "Go to sleep", "Stop listening", "Goodbye" |
| **Projects** | "Create project X", "Switch to X", "List projects" |
| **Agents** | "Spawn X agent", "Kill X agent", "List agents" |
| **Plugins** | "Install X plugin", "Remove X plugin", "List plugins" |
| **Tasks** | "Build", "Test", "Deploy", "Fix error" |
| **Info** | "Status", "Help" |

---

## Project Management

### Create a Project

```
"Create a new project called heimdall"
"New project heimdall"
```

**What happens:**
1. Creates folder: `~/prometheus_workspace/heimdall/`
2. Creates structure: `src/`, `tests/`, `docs/`, `plugins/`
3. Creates config: `prometheus.yaml`
4. Sets as active project

### Switch Projects

```
"Switch to argus"
"Go to argus"
"Open argus"
"Focus on argus"
```

### List Projects

```
"List projects"
"Show all projects"
"What projects do I have"
```

**Response:**
```
Projects:
  • argus (active) - 3 agents, 2 plugins
  • heimdall - 0 agents, 0 plugins
  • odin - 1 agent, 1 plugin
```

### Delete Project

```
"Delete project heimdall"
```

*Note: Only removes from registry, doesn't delete files*

---

## Agent Control

### Available Agents

| Agent | Purpose |
|-------|---------|
| `error` | Error analysis and auto-fixing |
| `dependency` | pip/npm package issues |
| `network` | Port conflicts, connections |
| `syntax` | Code errors, imports |
| `hardware` | GPIO, camera, sensors |
| `camera` | Camera capture/streaming |
| `detection` | AI object/face detection |
| `automation` | Home device control |
| `security` | Auth, encryption |

### Spawn an Agent

```
"Spawn the error agent"
"Start the camera agent"
"Activate detection"
"Wake up the hardware agent"
```

**Response:**
```
💬 Spawned error agent
   agent: error_argus
   project: argus
```

### Kill an Agent

```
"Kill the error agent"
"Stop the camera agent"
"Shutdown detection"
```

### List Running Agents

```
"List agents"
"What agents are running"
"Who is active"
```

**Response:**
```
Agents:
  • error_argus (running) - started 10:30
  • camera_argus (running) - started 10:32
  • detection_heimdall (idle)
```

### Check Agent Status

```
"Error agent status"
"How is the camera agent"
"Status of detection"
```

---

## Plugin System

### Available Plugins

| Plugin | Description |
|--------|-------------|
| `camera` | Camera capture and streaming |
| `detection` | AI object/face detection |
| `automation` | Home automation control |
| `voice` | Voice recognition and TTS |
| `mqtt` | MQTT messaging protocol |
| `homeassistant` | Home Assistant integration |
| `telegram` | Telegram bot notifications |
| `discord` | Discord bot integration |
| `database` | SQLite/PostgreSQL support |
| `scheduler` | Cron-like task scheduling |
| `git` | Git version control |
| `docker` | Docker container management |

### Install a Plugin

```
"Install the camera plugin"
"Add the telegram plugin"
"Enable mqtt"
```

**Response:**
```
💬 Installed camera plugin
   description: Camera capture and streaming
```

### Remove a Plugin

```
"Remove the telegram plugin"
"Uninstall mqtt"
"Disable discord"
```

### List Plugins

```
"List plugins"
"What plugins are installed"
"Show available plugins"
```

**Response:**
```
Installed: camera, detection, automation
Available: mqtt, telegram, discord, database, scheduler...
```

---

## Task Execution

### Build Project

```
"Build"
"Build the project"
"Compile"
```

### Run Tests

```
"Run tests"
"Test the project"
"Check tests"
```

### Deploy

```
"Deploy"                    → deploys to production
"Deploy to staging"         → deploys to staging
"Deploy to raspberry pi"    → deploys to raspberry_pi target
"Ship it"                   → deploys to production
```

### Fix Errors

```
"Fix the error"
"Fix the last error"
"What went wrong"
"Debug"
```

**What happens:**
1. Spawns error agent (if not running)
2. Analyzes last error
3. Suggests or applies fix
4. Reports result

---

## Workflows

### Workflow 1: New Project Setup

```
You: "Hey Prometheus"
Pro: "🟢 Prometheus awake. Listening..."

You: "Create project heimdall"
Pro: "Created project 'heimdall'"

You: "Install camera plugin"
Pro: "Installed camera plugin"

You: "Install detection plugin"
Pro: "Installed detection plugin"

You: "Spawn camera agent"
Pro: "Spawned camera agent"

You: "Status"
Pro: "Active project: heimdall, 1 agent, 2 plugins"
```

### Workflow 2: Error Fixing

```
You: "Hey Prometheus, fix the error"
Pro: "Analyzing and fixing last error..."
Pro: "Found: Port 8000 in use"
Pro: "Fix: Killed process 12345"
Pro: "Error resolved"
```

### Workflow 3: Multi-Project Management

```
You: "List projects"
Pro: "argus (active), heimdall, odin"

You: "Switch to heimdall"
Pro: "Switched to project 'heimdall'"

You: "Spawn error agent"
Pro: "Spawned error agent"

You: "Switch to argus"
Pro: "Switched to project 'argus'"

You: "List agents"
Pro: "error_argus, camera_argus, error_heimdall"
```

### Workflow 4: Development Cycle

```
You: "Build"
Pro: "Building argus..."

You: "Test"
Pro: "Running tests for argus..."

You: "Deploy to staging"
Pro: "Deploying argus to staging..."
```

---

## Text Mode Commands

When running with `--no-voice`:

```
prometheus> create project test
prometheus> switch to argus
prometheus> spawn error agent
prometheus> list agents
prometheus> status
prometheus> help
prometheus> quit
```

---

## Configuration

### Project Config (prometheus.yaml)

```yaml
name: argus
version: 0.1.0
description: Smart home security system

plugins:
  - camera
  - detection
  - automation

agents:
  default:
    - error
    - camera
  production:
    - error
    - camera
    - detection
    - automation

build:
  command: python -m pytest && python main.py

deploy:
  raspberry_pi:
    host: 192.168.1.100
    user: pi
    path: /home/pi/argus
```

### Workspace Structure

```
~/prometheus_workspace/
├── argus/
│   ├── prometheus.yaml
│   ├── src/
│   ├── tests/
│   └── docs/
├── heimdall/
│   ├── prometheus.yaml
│   └── ...
└── odin/
    └── ...
```

---

## Troubleshooting

### "Voice disabled" on startup

**Cause:** SpeechRecognition or PyAudio not installed

**Fix:**
```bash
pip install SpeechRecognition pyaudio
```

### "No microphone found"

**Cause:** System can't access microphone

**Fix:**
1. Check microphone is connected
2. Check system permissions
3. Run: `python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"`

### "Could not understand audio"

**Cause:** Background noise or unclear speech

**Fix:**
1. Speak clearly and closer to mic
2. Reduce background noise
3. Try: `python -m prometheus.main --no-voice` (text mode)

### "API error" for voice

**Cause:** No internet (Google API) or rate limited

**Fix:**
Use offline Whisper model:
```python
# In listener.py, set:
self.use_whisper = True
```

### "Project not found"

**Cause:** Typo or project doesn't exist

**Fix:**
```
"List projects"  # See available projects
```

### Agent won't spawn

**Cause:** No active project

**Fix:**
```
"Switch to argus"  # Select a project first
"Spawn error agent"
```

---

## Keyboard Shortcuts (Text Mode)

| Key | Action |
|-----|--------|
| `Enter` | Execute command |
| `Ctrl+C` | Stop Prometheus |
| `quit` | Exit gracefully |

---

## Best Practices

1. **Always wake first** - Say "Hey Prometheus" before commands
2. **One command at a time** - Wait for response before next command
3. **Use project context** - Switch to project before agent operations
4. **Check status often** - "Status" shows current state
5. **Sleep when done** - "Go to sleep" to stop listening

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    PROMETHEUS QUICK REF                      │
├─────────────────────────────────────────────────────────────┤
│  WAKE:     "Hey Prometheus"                                 │
│  SLEEP:    "Go to sleep"                                    │
├─────────────────────────────────────────────────────────────┤
│  PROJECTS                                                    │
│    Create:  "Create project <name>"                         │
│    Switch:  "Switch to <name>"                              │
│    List:    "List projects"                                 │
├─────────────────────────────────────────────────────────────┤
│  AGENTS                                                      │
│    Start:   "Spawn <type> agent"                            │
│    Stop:    "Kill <type> agent"                             │
│    List:    "List agents"                                   │
├─────────────────────────────────────────────────────────────┤
│  PLUGINS                                                     │
│    Add:     "Install <name> plugin"                         │
│    Remove:  "Remove <name> plugin"                          │
│    List:    "List plugins"                                  │
├─────────────────────────────────────────────────────────────┤
│  TASKS                                                       │
│    "Build" | "Test" | "Deploy" | "Fix error"                │
├─────────────────────────────────────────────────────────────┤
│  INFO                                                        │
│    "Status" | "Help"                                        │
└─────────────────────────────────────────────────────────────┘
```

---

*"I am Prometheus. I bring fire to your code."*
