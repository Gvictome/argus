# PROMETHEUS

**Voice-Activated Senior Dev Orchestrator**

Prometheus is the master AI orchestrator that listens via microphone, manages all sub-agents, and controls multiple projects with plugin architectures.

---

## Overview

```
                            ┌─────────────────────┐
                            │     PROMETHEUS      │
                            │  (Senior Dev Agent) │
                            │                     │
                            │  🎤 Voice Input     │
                            │  🧠 Command Parser  │
                            │  📡 Agent Router    │
                            └──────────┬──────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
    ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
    │   PROJECT    │           │   PROJECT    │           │   PROJECT    │
    │    ARGUS     │           │   [NEW]      │           │   [NEW]      │
    │              │           │              │           │              │
    │ ┌──────────┐ │           │ ┌──────────┐ │           │ ┌──────────┐ │
    │ │Sub-Agents│ │           │ │Sub-Agents│ │           │ │Sub-Agents│ │
    │ └──────────┘ │           │ └──────────┘ │           │ └──────────┘ │
    │ ┌──────────┐ │           │ ┌──────────┐ │           │ ┌──────────┐ │
    │ │ Plugins  │ │           │ │ Plugins  │ │           │ │ Plugins  │ │
    │ └──────────┘ │           │ └──────────┘ │           │ └──────────┘ │
    └──────────────┘           └──────────────┘           └──────────────┘
```

---

## Core Components

### 1. Voice Interface

```python
# prometheus/voice/listener.py

import speech_recognition as sr
import threading
from queue import Queue
from typing import Callable, Optional

class VoiceListener:
    """
    Continuous microphone listener for voice commands

    Wake words: "Hey Prometheus", "Prometheus", "Senior Dev"
    """

    WAKE_WORDS = ["prometheus", "senior dev", "hey prometheus"]

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_queue = Queue()
        self.is_listening = False
        self.is_awake = False
        self._callbacks: list[Callable] = []

        # Calibrate for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def start(self):
        """Start listening in background thread"""
        self.is_listening = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        print("🎤 Prometheus listening... Say 'Hey Prometheus' to wake")

    def stop(self):
        """Stop listening"""
        self.is_listening = False

    def _listen_loop(self):
        """Main listening loop"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

                # Transcribe
                text = self._transcribe(audio)
                if not text:
                    continue

                text_lower = text.lower()

                # Check for wake word
                if not self.is_awake:
                    if any(wake in text_lower for wake in self.WAKE_WORDS):
                        self.is_awake = True
                        self._notify("🟢 Prometheus awake. Listening for commands...")
                        # Extract command after wake word
                        for wake in self.WAKE_WORDS:
                            if wake in text_lower:
                                command = text_lower.split(wake, 1)[-1].strip()
                                if command:
                                    self._process_command(command)
                                break
                else:
                    # Already awake - process command directly
                    if "go to sleep" in text_lower or "stop listening" in text_lower:
                        self.is_awake = False
                        self._notify("🔴 Prometheus sleeping. Say wake word to activate.")
                    else:
                        self._process_command(text)

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Listen error: {e}")

    def _transcribe(self, audio) -> Optional[str]:
        """Transcribe audio to text"""
        try:
            # Option 1: Google (free, requires internet)
            return self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            # Fallback: Whisper (local, offline)
            try:
                return self.recognizer.recognize_whisper(audio, model="base")
            except:
                return None

    def _process_command(self, command: str):
        """Queue command for processing"""
        self.command_queue.put(command)
        for callback in self._callbacks:
            callback(command)

    def _notify(self, message: str):
        """Send notification"""
        print(message)
        # TODO: Audio feedback via TTS

    def on_command(self, callback: Callable):
        """Register command callback"""
        self._callbacks.append(callback)

    def get_command(self, timeout: float = None) -> Optional[str]:
        """Get next command from queue"""
        try:
            return self.command_queue.get(timeout=timeout)
        except:
            return None
```

### 2. Command Parser

```python
# prometheus/core/parser.py

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

class CommandType(Enum):
    # Project Management
    CREATE_PROJECT = "create_project"
    SWITCH_PROJECT = "switch_project"
    LIST_PROJECTS = "list_projects"
    DELETE_PROJECT = "delete_project"

    # Agent Control
    SPAWN_AGENT = "spawn_agent"
    KILL_AGENT = "kill_agent"
    LIST_AGENTS = "list_agents"
    AGENT_STATUS = "agent_status"

    # Plugin Management
    INSTALL_PLUGIN = "install_plugin"
    REMOVE_PLUGIN = "remove_plugin"
    LIST_PLUGINS = "list_plugins"

    # Task Execution
    RUN_TASK = "run_task"
    FIX_ERROR = "fix_error"
    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"

    # System
    STATUS = "status"
    HELP = "help"
    SLEEP = "sleep"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Structured command output"""
    type: CommandType
    project: Optional[str] = None
    agent: Optional[str] = None
    plugin: Optional[str] = None
    target: Optional[str] = None
    args: Dict[str, Any] = None
    raw_text: str = ""


class CommandParser:
    """
    Natural language command parser

    Examples:
        "create a new project called heimdall"
        "switch to argus"
        "spawn the error agent"
        "install the camera plugin"
        "run tests on argus"
        "fix the last error"
        "what's the status"
    """

    # Command patterns (regex)
    PATTERNS = {
        CommandType.CREATE_PROJECT: [
            r"create (?:a )?(?:new )?project (?:called |named )?(\w+)",
            r"new project (\w+)",
            r"start (?:a )?(?:new )?project (\w+)",
        ],
        CommandType.SWITCH_PROJECT: [
            r"switch to (\w+)",
            r"go to (\w+)",
            r"open (\w+)",
            r"focus on (\w+)",
        ],
        CommandType.LIST_PROJECTS: [
            r"list (?:all )?projects",
            r"show (?:all )?projects",
            r"what projects",
        ],
        CommandType.SPAWN_AGENT: [
            r"spawn (?:the )?(\w+) agent",
            r"start (?:the )?(\w+) agent",
            r"activate (?:the )?(\w+) agent",
            r"wake up (?:the )?(\w+)",
        ],
        CommandType.KILL_AGENT: [
            r"kill (?:the )?(\w+) agent",
            r"stop (?:the )?(\w+) agent",
            r"shutdown (?:the )?(\w+)",
        ],
        CommandType.LIST_AGENTS: [
            r"list (?:all )?agents",
            r"show (?:all )?agents",
            r"what agents",
            r"who is running",
        ],
        CommandType.INSTALL_PLUGIN: [
            r"install (?:the )?(\w+) plugin",
            r"add (?:the )?(\w+) plugin",
            r"enable (\w+)",
        ],
        CommandType.REMOVE_PLUGIN: [
            r"remove (?:the )?(\w+) plugin",
            r"uninstall (?:the )?(\w+) plugin",
            r"disable (\w+)",
        ],
        CommandType.LIST_PLUGINS: [
            r"list (?:all )?plugins",
            r"show (?:all )?plugins",
            r"what plugins",
        ],
        CommandType.RUN_TASK: [
            r"run (\w+)",
            r"execute (\w+)",
            r"do (\w+)",
        ],
        CommandType.FIX_ERROR: [
            r"fix (?:the )?(?:last )?error",
            r"resolve (?:the )?(?:last )?error",
            r"debug (?:the )?(?:last )?error",
            r"what went wrong",
        ],
        CommandType.BUILD: [
            r"build (?:the )?(?:project)?",
            r"compile",
            r"make",
        ],
        CommandType.TEST: [
            r"run tests",
            r"test (?:the )?(?:project)?",
            r"check tests",
        ],
        CommandType.DEPLOY: [
            r"deploy (?:to )?(\w+)?",
            r"ship it",
            r"push to production",
        ],
        CommandType.STATUS: [
            r"status",
            r"what's (?:the )?status",
            r"how are we doing",
            r"system status",
        ],
        CommandType.HELP: [
            r"help",
            r"what can you do",
            r"commands",
        ],
        CommandType.SLEEP: [
            r"go to sleep",
            r"stop listening",
            r"shutdown",
            r"goodbye",
        ],
    }

    def parse(self, text: str) -> ParsedCommand:
        """Parse natural language command"""
        text_lower = text.lower().strip()

        for cmd_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return self._build_command(cmd_type, match, text)

        return ParsedCommand(
            type=CommandType.UNKNOWN,
            raw_text=text
        )

    def _build_command(self, cmd_type: CommandType, match, raw_text: str) -> ParsedCommand:
        """Build structured command from match"""
        groups = match.groups()

        cmd = ParsedCommand(
            type=cmd_type,
            raw_text=raw_text,
            args={}
        )

        # Extract entities based on command type
        if cmd_type in [CommandType.CREATE_PROJECT, CommandType.SWITCH_PROJECT]:
            cmd.project = groups[0] if groups else None

        elif cmd_type in [CommandType.SPAWN_AGENT, CommandType.KILL_AGENT]:
            cmd.agent = groups[0] if groups else None

        elif cmd_type in [CommandType.INSTALL_PLUGIN, CommandType.REMOVE_PLUGIN]:
            cmd.plugin = groups[0] if groups else None

        elif cmd_type == CommandType.DEPLOY:
            cmd.target = groups[0] if groups else "production"

        elif cmd_type == CommandType.RUN_TASK:
            cmd.args["task"] = groups[0] if groups else None

        return cmd
```

### 3. Agent Router

```python
# prometheus/core/router.py

import asyncio
from typing import Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from prometheus.core.parser import ParsedCommand, CommandType


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentInfo:
    """Information about a registered agent"""
    name: str
    type: str
    project: str
    status: AgentStatus
    pid: Optional[int] = None
    capabilities: list = None


@dataclass
class ProjectInfo:
    """Information about a registered project"""
    name: str
    path: Path
    agents: Dict[str, AgentInfo]
    plugins: list
    active: bool = False


class AgentRouter:
    """
    Routes commands to appropriate agents across projects

    Manages:
    - Project registry
    - Agent lifecycle
    - Plugin system
    - Cross-project communication
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.projects: Dict[str, ProjectInfo] = {}
        self.active_project: Optional[str] = None
        self.global_agents: Dict[str, AgentInfo] = {}

        # Discover existing projects
        self._discover_projects()

    def _discover_projects(self):
        """Scan workspace for existing projects"""
        if not self.workspace_root.exists():
            self.workspace_root.mkdir(parents=True)
            return

        for path in self.workspace_root.iterdir():
            if path.is_dir() and (path / "prometheus.yaml").exists():
                self._register_project(path)

    def _register_project(self, path: Path):
        """Register a project from its config"""
        import yaml

        config_path = path / "prometheus.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        project = ProjectInfo(
            name=path.name,
            path=path,
            agents={},
            plugins=config.get("plugins", []),
            active=False
        )

        self.projects[path.name] = project
        print(f"📁 Registered project: {path.name}")

    async def execute(self, command: ParsedCommand) -> Dict[str, Any]:
        """Execute a parsed command"""
        handlers = {
            CommandType.CREATE_PROJECT: self._create_project,
            CommandType.SWITCH_PROJECT: self._switch_project,
            CommandType.LIST_PROJECTS: self._list_projects,
            CommandType.SPAWN_AGENT: self._spawn_agent,
            CommandType.KILL_AGENT: self._kill_agent,
            CommandType.LIST_AGENTS: self._list_agents,
            CommandType.INSTALL_PLUGIN: self._install_plugin,
            CommandType.REMOVE_PLUGIN: self._remove_plugin,
            CommandType.LIST_PLUGINS: self._list_plugins,
            CommandType.RUN_TASK: self._run_task,
            CommandType.FIX_ERROR: self._fix_error,
            CommandType.BUILD: self._build,
            CommandType.TEST: self._test,
            CommandType.DEPLOY: self._deploy,
            CommandType.STATUS: self._status,
            CommandType.HELP: self._help,
        }

        handler = handlers.get(command.type)
        if handler:
            return await handler(command)

        return {"error": f"Unknown command: {command.raw_text}"}

    # =========================================================================
    # Project Management
    # =========================================================================

    async def _create_project(self, cmd: ParsedCommand) -> Dict:
        """Create a new project"""
        name = cmd.project
        if not name:
            return {"error": "Project name required"}

        if name in self.projects:
            return {"error": f"Project '{name}' already exists"}

        # Create project directory
        project_path = self.workspace_root / name
        project_path.mkdir(parents=True)

        # Create default structure
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "docs").mkdir()
        (project_path / "plugins").mkdir()

        # Create prometheus config
        config = {
            "name": name,
            "version": "0.1.0",
            "plugins": [],
            "agents": {
                "default": ["error", "build", "test"]
            }
        }

        import yaml
        with open(project_path / "prometheus.yaml", "w") as f:
            yaml.dump(config, f)

        # Register project
        self._register_project(project_path)
        self.active_project = name

        return {
            "success": True,
            "message": f"Created project '{name}'",
            "path": str(project_path)
        }

    async def _switch_project(self, cmd: ParsedCommand) -> Dict:
        """Switch active project"""
        name = cmd.project
        if name not in self.projects:
            return {"error": f"Project '{name}' not found"}

        # Deactivate current
        if self.active_project:
            self.projects[self.active_project].active = False

        # Activate new
        self.active_project = name
        self.projects[name].active = True

        return {
            "success": True,
            "message": f"Switched to project '{name}'",
            "project": name
        }

    async def _list_projects(self, cmd: ParsedCommand) -> Dict:
        """List all projects"""
        return {
            "projects": [
                {
                    "name": p.name,
                    "path": str(p.path),
                    "active": p.active,
                    "agents": len(p.agents),
                    "plugins": len(p.plugins)
                }
                for p in self.projects.values()
            ],
            "active": self.active_project
        }

    # =========================================================================
    # Agent Management
    # =========================================================================

    async def _spawn_agent(self, cmd: ParsedCommand) -> Dict:
        """Spawn a new agent"""
        agent_type = cmd.agent
        if not agent_type:
            return {"error": "Agent type required"}

        project = self.active_project
        if not project:
            return {"error": "No active project. Switch to a project first."}

        # Agent registry
        agent_classes = {
            "error": "ErrorRouter",
            "dependency": "DependencyAgent",
            "network": "NetworkAgent",
            "syntax": "SyntaxAgent",
            "hardware": "HardwareAgent",
            "camera": "CameraService",
            "detection": "DetectionService",
            "automation": "AutomationService",
            "build": "BuildAgent",
            "test": "TestAgent",
            "deploy": "DeployAgent",
        }

        if agent_type not in agent_classes:
            return {"error": f"Unknown agent type: {agent_type}"}

        # Create agent info
        agent = AgentInfo(
            name=f"{agent_type}_{project}",
            type=agent_type,
            project=project,
            status=AgentStatus.RUNNING,
            capabilities=[]
        )

        self.projects[project].agents[agent_type] = agent

        return {
            "success": True,
            "message": f"Spawned {agent_type} agent",
            "agent": agent.name
        }

    async def _kill_agent(self, cmd: ParsedCommand) -> Dict:
        """Kill an agent"""
        agent_type = cmd.agent
        project = self.active_project

        if not project:
            return {"error": "No active project"}

        if agent_type not in self.projects[project].agents:
            return {"error": f"Agent '{agent_type}' not running"}

        del self.projects[project].agents[agent_type]

        return {
            "success": True,
            "message": f"Killed {agent_type} agent"
        }

    async def _list_agents(self, cmd: ParsedCommand) -> Dict:
        """List all agents"""
        all_agents = []

        for project in self.projects.values():
            for agent in project.agents.values():
                all_agents.append({
                    "name": agent.name,
                    "type": agent.type,
                    "project": agent.project,
                    "status": agent.status.value
                })

        return {
            "agents": all_agents,
            "count": len(all_agents)
        }

    # =========================================================================
    # Plugin Management
    # =========================================================================

    async def _install_plugin(self, cmd: ParsedCommand) -> Dict:
        """Install a plugin"""
        plugin = cmd.plugin
        project = self.active_project

        if not project:
            return {"error": "No active project"}

        # Plugin registry
        available_plugins = {
            "camera": "Camera capture and streaming",
            "detection": "AI object/face detection",
            "automation": "Home automation control",
            "voice": "Voice recognition",
            "mqtt": "MQTT messaging",
            "homeassistant": "Home Assistant integration",
            "telegram": "Telegram notifications",
            "discord": "Discord bot integration",
            "database": "SQLite/PostgreSQL support",
            "scheduler": "Task scheduling",
        }

        if plugin not in available_plugins:
            return {
                "error": f"Unknown plugin: {plugin}",
                "available": list(available_plugins.keys())
            }

        if plugin in self.projects[project].plugins:
            return {"error": f"Plugin '{plugin}' already installed"}

        self.projects[project].plugins.append(plugin)

        return {
            "success": True,
            "message": f"Installed {plugin} plugin",
            "description": available_plugins[plugin]
        }

    async def _remove_plugin(self, cmd: ParsedCommand) -> Dict:
        """Remove a plugin"""
        plugin = cmd.plugin
        project = self.active_project

        if not project:
            return {"error": "No active project"}

        if plugin not in self.projects[project].plugins:
            return {"error": f"Plugin '{plugin}' not installed"}

        self.projects[project].plugins.remove(plugin)

        return {
            "success": True,
            "message": f"Removed {plugin} plugin"
        }

    async def _list_plugins(self, cmd: ParsedCommand) -> Dict:
        """List plugins"""
        project = self.active_project

        installed = []
        if project:
            installed = self.projects[project].plugins

        return {
            "installed": installed,
            "project": project
        }

    # =========================================================================
    # Task Execution
    # =========================================================================

    async def _run_task(self, cmd: ParsedCommand) -> Dict:
        """Run a task"""
        task = cmd.args.get("task")
        return {"message": f"Running task: {task}"}

    async def _fix_error(self, cmd: ParsedCommand) -> Dict:
        """Fix the last error"""
        # Delegate to error router
        return {"message": "Analyzing and fixing last error..."}

    async def _build(self, cmd: ParsedCommand) -> Dict:
        """Build the project"""
        project = self.active_project
        if not project:
            return {"error": "No active project"}
        return {"message": f"Building {project}..."}

    async def _test(self, cmd: ParsedCommand) -> Dict:
        """Run tests"""
        project = self.active_project
        if not project:
            return {"error": "No active project"}
        return {"message": f"Running tests for {project}..."}

    async def _deploy(self, cmd: ParsedCommand) -> Dict:
        """Deploy the project"""
        target = cmd.target or "production"
        project = self.active_project
        if not project:
            return {"error": "No active project"}
        return {"message": f"Deploying {project} to {target}..."}

    async def _status(self, cmd: ParsedCommand) -> Dict:
        """Get system status"""
        return {
            "active_project": self.active_project,
            "total_projects": len(self.projects),
            "total_agents": sum(len(p.agents) for p in self.projects.values()),
            "projects": await self._list_projects(cmd)
        }

    async def _help(self, cmd: ParsedCommand) -> Dict:
        """Show help"""
        return {
            "commands": [
                "create project <name>",
                "switch to <project>",
                "list projects",
                "spawn <type> agent",
                "kill <type> agent",
                "list agents",
                "install <name> plugin",
                "remove <name> plugin",
                "list plugins",
                "run <task>",
                "fix error",
                "build",
                "test",
                "deploy [target]",
                "status",
                "go to sleep",
            ]
        }
```

---

## 4. Prometheus Main Controller

```python
# prometheus/main.py

import asyncio
from pathlib import Path

from prometheus.voice.listener import VoiceListener
from prometheus.core.parser import CommandParser
from prometheus.core.router import AgentRouter
from prometheus.voice.speaker import TextToSpeech  # TTS for responses


class Prometheus:
    """
    PROMETHEUS - Senior Dev AI Orchestrator

    Voice-activated master controller that manages
    all projects, agents, and plugins.
    """

    def __init__(self, workspace: Path = None):
        self.workspace = workspace or Path.home() / "prometheus_workspace"
        self.listener = VoiceListener()
        self.parser = CommandParser()
        self.router = AgentRouter(self.workspace)
        self.speaker = TextToSpeech()
        self.running = False

    async def start(self):
        """Start Prometheus"""
        print("""
        ╔═══════════════════════════════════════════════════════════╗
        ║                                                           ║
        ║   ██████╗ ██████╗  ██████╗ ███╗   ███╗███████╗████████╗  ║
        ║   ██╔══██╗██╔══██╗██╔═══██╗████╗ ████║██╔════╝╚══██╔══╝  ║
        ║   ██████╔╝██████╔╝██║   ██║██╔████╔██║█████╗     ██║     ║
        ║   ██╔═══╝ ██╔══██╗██║   ██║██║╚██╔╝██║██╔══╝     ██║     ║
        ║   ██║     ██║  ██║╚██████╔╝██║ ╚═╝ ██║███████╗   ██║     ║
        ║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝   ╚═╝     ║
        ║                                                           ║
        ║           Senior Dev AI Orchestrator v1.0                 ║
        ║                                                           ║
        ╚═══════════════════════════════════════════════════════════╝
        """)

        # Register command callback
        self.listener.on_command(self._on_voice_command)

        # Start voice listener
        self.listener.start()

        self.running = True

        # Main loop
        while self.running:
            await asyncio.sleep(0.1)

    async def _on_voice_command(self, text: str):
        """Handle voice command"""
        print(f"\n🎤 Heard: {text}")

        # Parse command
        command = self.parser.parse(text)
        print(f"📝 Parsed: {command.type.value}")

        # Check for sleep command
        if command.type.value == "sleep":
            self.speaker.say("Going to sleep. Goodbye.")
            return

        # Execute command
        result = await self.router.execute(command)

        # Speak response
        if "error" in result:
            response = f"Error: {result['error']}"
        elif "message" in result:
            response = result["message"]
        else:
            response = "Command executed."

        print(f"💬 Response: {response}")
        self.speaker.say(response)

    def stop(self):
        """Stop Prometheus"""
        self.running = False
        self.listener.stop()
        print("\n👋 Prometheus shutting down...")


# Entry point
if __name__ == "__main__":
    prometheus = Prometheus()

    try:
        asyncio.run(prometheus.start())
    except KeyboardInterrupt:
        prometheus.stop()
```

---

## 5. Text-to-Speech Response

```python
# prometheus/voice/speaker.py

import platform
import subprocess
from typing import Optional

class TextToSpeech:
    """
    Cross-platform text-to-speech for Prometheus responses
    """

    def __init__(self, voice: str = None, rate: int = 175):
        self.voice = voice
        self.rate = rate
        self.system = platform.system()

    def say(self, text: str):
        """Speak text aloud"""
        try:
            if self.system == "Darwin":  # macOS
                subprocess.run(["say", text], check=False)

            elif self.system == "Windows":
                # Use PowerShell
                ps_script = f'''
                Add-Type -AssemblyName System.Speech
                $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $synth.Rate = {self.rate // 50 - 2}
                $synth.Speak("{text}")
                '''
                subprocess.run(
                    ["powershell", "-Command", ps_script],
                    check=False,
                    capture_output=True
                )

            elif self.system == "Linux":
                # Use espeak or festival
                try:
                    subprocess.run(
                        ["espeak", "-s", str(self.rate), text],
                        check=False,
                        capture_output=True
                    )
                except FileNotFoundError:
                    subprocess.run(
                        ["festival", "--tts"],
                        input=text.encode(),
                        check=False
                    )

        except Exception as e:
            print(f"TTS error: {e}")

    def say_async(self, text: str):
        """Speak text in background thread"""
        import threading
        thread = threading.Thread(target=self.say, args=(text,), daemon=True)
        thread.start()
```

---

## 6. Project Configuration Schema

```yaml
# prometheus.yaml - Project configuration

name: argus
version: 0.1.0
description: Smart home security system

# Plugins enabled for this project
plugins:
  - camera
  - detection
  - automation
  - voice

# Default agents to spawn
agents:
  default:
    - error
    - build
    - test
  production:
    - error
    - hardware
    - camera
    - detection

# Build configuration
build:
  command: python -m pytest && python main.py
  output: dist/

# Deployment targets
deploy:
  raspberry_pi:
    host: 192.168.1.100
    user: pi
    path: /home/pi/argus
  production:
    host: prod.example.com
    user: deploy
    path: /opt/argus

# Environment variables
env:
  DEBUG: false
  PORT: 8000
```

---

## 7. Available Plugins

| Plugin | Description | Agents Provided |
|--------|-------------|-----------------|
| `camera` | Camera capture, streaming, recording | CameraAgent |
| `detection` | Motion, object, face detection | DetectionAgent |
| `automation` | Smart home device control | AutomationAgent |
| `voice` | Voice recognition and TTS | VoiceAgent |
| `mqtt` | MQTT pub/sub messaging | MQTTAgent |
| `homeassistant` | Home Assistant API integration | HAAgent |
| `telegram` | Telegram bot notifications | TelegramAgent |
| `discord` | Discord bot integration | DiscordAgent |
| `database` | SQLite/PostgreSQL | DatabaseAgent |
| `scheduler` | Cron-like task scheduling | SchedulerAgent |
| `git` | Git operations | GitAgent |
| `docker` | Docker container management | DockerAgent |

---

## 8. Voice Commands Reference

### Project Commands
```
"Create a new project called [name]"
"Switch to [project]"
"List all projects"
"Delete project [name]"
```

### Agent Commands
```
"Spawn the [type] agent"
"Kill the [type] agent"
"List all agents"
"What agents are running"
```

### Plugin Commands
```
"Install the [name] plugin"
"Remove the [name] plugin"
"List plugins"
"What plugins are available"
```

### Task Commands
```
"Run [task name]"
"Build the project"
"Run tests"
"Deploy to [target]"
"Fix the last error"
```

### System Commands
```
"What's the status"
"Help"
"Go to sleep"
"Wake up"
```

---

## 9. Installation

```bash
# Install dependencies
pip install SpeechRecognition pyaudio pyttsx3 pyyaml

# For Whisper (offline voice recognition)
pip install openai-whisper

# For Linux TTS
sudo apt install espeak festival

# Create workspace
mkdir ~/prometheus_workspace
cd ~/prometheus_workspace

# Run Prometheus
python -m prometheus.main
```

---

## 10. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            PROMETHEUS                                    │
│                       Senior Dev Orchestrator                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │   Voice     │───▶│   Command   │───▶│   Agent     │                │
│   │  Listener   │    │   Parser    │    │   Router    │                │
│   └─────────────┘    └─────────────┘    └──────┬──────┘                │
│         ▲                                       │                       │
│         │                                       ▼                       │
│   ┌─────────────┐                    ┌─────────────────────┐           │
│   │    TTS      │◀───────────────────│   Project Manager   │           │
│   │  Response   │                    └─────────┬───────────┘           │
│   └─────────────┘                              │                       │
│                                                │                       │
├────────────────────────────────────────────────┼───────────────────────┤
│                                                │                       │
│   PROJECT: ARGUS          PROJECT: HEIMDALL   │   PROJECT: [NEW]      │
│   ┌──────────────────┐    ┌─────────────────┐ │   ┌─────────────────┐ │
│   │ ┌──────────────┐ │    │ ┌─────────────┐ │ │   │                 │ │
│   │ │ ErrorAgent   │ │    │ │ BuildAgent  │ │ │   │   (Empty)       │ │
│   │ ├──────────────┤ │    │ ├─────────────┤ │ │   │                 │ │
│   │ │ CameraAgent  │ │    │ │ TestAgent   │ │ │   │                 │ │
│   │ ├──────────────┤ │    │ ├─────────────┤ │ │   │                 │ │
│   │ │ DetectAgent  │ │    │ │ DeployAgent │ │ │   │                 │ │
│   │ └──────────────┘ │    │ └─────────────┘ │ │   │                 │ │
│   │                  │    │                 │ │   │                 │ │
│   │ Plugins:         │    │ Plugins:        │ │   │ Plugins:        │ │
│   │ - camera         │    │ - git           │ │   │ - (none)        │ │
│   │ - detection      │    │ - docker        │ │   │                 │ │
│   │ - automation     │    │ - scheduler     │ │   │                 │ │
│   └──────────────────┘    └─────────────────┘ │   └─────────────────┘ │
│                                                │                       │
└────────────────────────────────────────────────┴───────────────────────┘
```

---

## Next Steps

1. **Install speech recognition**: `pip install SpeechRecognition pyaudio`
2. **Test voice input**: Run the VoiceListener standalone
3. **Configure workspace**: Set up `~/prometheus_workspace`
4. **Register ARGUS**: Add `prometheus.yaml` to argus project
5. **Launch Prometheus**: `python -m prometheus.main`

---

*"I am Prometheus. I bring fire to your code."*
