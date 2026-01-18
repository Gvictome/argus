"""
Agent Router - Routes commands to agents across projects
"""

import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any, List
from datetime import datetime

from prometheus.core.parser import ParsedCommand, CommandType


class AgentStatus(Enum):
    """Status of an agent"""
    IDLE = "idle"
    RUNNING = "running"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentInfo:
    """Information about a registered agent"""
    name: str
    type: str
    project: str
    status: AgentStatus = AgentStatus.STOPPED
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    capabilities: List[str] = field(default_factory=list)
    buffs: List[str] = field(default_factory=list)  # Active buff names
    config: Dict[str, Any] = field(default_factory=dict)  # Enhanced config
    error: Optional[str] = None


@dataclass
class ProjectInfo:
    """Information about a registered project"""
    name: str
    path: Path
    industry: str = "general"  # Industry concentration
    agents: Dict[str, AgentInfo] = field(default_factory=dict)
    plugins: List[str] = field(default_factory=list)
    active: bool = False
    created_at: datetime = field(default_factory=datetime.now)


class AgentRouter:
    """
    Routes commands to appropriate agents across projects

    Manages:
    - Project registry
    - Agent lifecycle
    - Plugin system
    - Cross-project communication
    """

    # Available agent types
    AGENT_REGISTRY = {
        "error": {"class": "ErrorRouter", "module": "src.agents.error_router"},
        "dependency": {"class": "DependencyAgent", "module": "src.agents.dependency_agent"},
        "network": {"class": "NetworkAgent", "module": "src.agents.network_agent"},
        "syntax": {"class": "SyntaxAgent", "module": "src.agents.syntax_agent"},
        "hardware": {"class": "HardwareAgent", "module": "src.agents.hardware_agent"},
        "camera": {"class": "CameraService", "module": "src.camera"},
        "detection": {"class": "DetectionService", "module": "src.detection"},
        "automation": {"class": "AutomationService", "module": "src.automation"},
        "security": {"class": "SecurityService", "module": "src.security"},
    }

    # Available plugins
    PLUGIN_REGISTRY = {
        "camera": "Camera capture and streaming",
        "detection": "AI object/face detection",
        "automation": "Home automation control",
        "voice": "Voice recognition and TTS",
        "mqtt": "MQTT messaging protocol",
        "homeassistant": "Home Assistant integration",
        "telegram": "Telegram bot notifications",
        "discord": "Discord bot integration",
        "database": "SQLite/PostgreSQL support",
        "scheduler": "Task scheduling (cron-like)",
        "git": "Git version control",
        "docker": "Docker container management",
    }

    def __init__(self, workspace_root: Path = None):
        """
        Initialize router

        Args:
            workspace_root: Root directory for all projects
        """
        self.workspace_root = workspace_root or Path.home() / "prometheus_workspace"
        self.projects: Dict[str, ProjectInfo] = {}
        self.active_project: Optional[str] = None

        # Ensure workspace exists
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Discover existing projects
        self._discover_projects()

    def _discover_projects(self):
        """Scan workspace for existing projects"""
        for path in self.workspace_root.iterdir():
            if path.is_dir():
                config_path = path / "prometheus.yaml"
                if config_path.exists():
                    self._register_project(path)
                elif (path / "main.py").exists() or (path / "src").exists():
                    # Register as project even without prometheus.yaml
                    self._register_project(path, has_config=False)

    def _register_project(self, path: Path, has_config: bool = True):
        """Register a project"""
        config = {}

        if has_config:
            try:
                import yaml
                with open(path / "prometheus.yaml") as f:
                    config = yaml.safe_load(f) or {}
            except:
                pass

        project = ProjectInfo(
            name=path.name,
            path=path,
            plugins=config.get("plugins", []),
        )

        self.projects[path.name] = project
        print(f"📁 Registered project: {path.name}")

    async def execute(self, command: ParsedCommand) -> Dict[str, Any]:
        """
        Execute a parsed command

        Args:
            command: Parsed command object

        Returns:
            Result dictionary
        """
        handlers = {
            CommandType.CREATE_PROJECT: self._create_project,
            CommandType.SWITCH_PROJECT: self._switch_project,
            CommandType.LIST_PROJECTS: self._list_projects,
            CommandType.DELETE_PROJECT: self._delete_project,
            CommandType.SPAWN_AGENT: self._spawn_agent,
            CommandType.KILL_AGENT: self._kill_agent,
            CommandType.LIST_AGENTS: self._list_agents,
            CommandType.AGENT_STATUS: self._agent_status,
            CommandType.BUFF_AGENT: self._buff_agent,
            CommandType.SET_INDUSTRY: self._set_industry,
            CommandType.LIST_INDUSTRIES: self._list_industries,
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
            try:
                return await handler(command)
            except Exception as e:
                return {"error": str(e)}

        return {
            "error": f"Unknown command: {command.raw_text}",
            "type": command.type.value
        }

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

        # Create structure
        (project_path / "src").mkdir()
        (project_path / "tests").mkdir()
        (project_path / "docs").mkdir()
        (project_path / "plugins").mkdir()

        # Create config
        config = {
            "name": name,
            "version": "0.1.0",
            "plugins": [],
            "agents": {"default": ["error"]}
        }

        try:
            import yaml
            with open(project_path / "prometheus.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            # Fallback to JSON
            import json
            with open(project_path / "prometheus.json", "w") as f:
                json.dump(config, f, indent=2)

        self._register_project(project_path)
        self.active_project = name
        self.projects[name].active = True

        return {
            "success": True,
            "message": f"Created project '{name}'",
            "path": str(project_path)
        }

    async def _switch_project(self, cmd: ParsedCommand) -> Dict:
        """Switch active project"""
        name = cmd.project
        if name not in self.projects:
            # Check for partial match
            matches = [p for p in self.projects if name.lower() in p.lower()]
            if len(matches) == 1:
                name = matches[0]
            elif matches:
                return {"error": f"Ambiguous: {matches}"}
            else:
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
            "project": name,
            "path": str(self.projects[name].path)
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
                    "plugins": p.plugins
                }
                for p in self.projects.values()
            ],
            "active": self.active_project,
            "count": len(self.projects)
        }

    async def _delete_project(self, cmd: ParsedCommand) -> Dict:
        """Delete a project (removes from registry, not files)"""
        name = cmd.project
        if name not in self.projects:
            return {"error": f"Project '{name}' not found"}

        if self.active_project == name:
            self.active_project = None

        del self.projects[name]

        return {
            "success": True,
            "message": f"Removed project '{name}' from registry"
        }

    # =========================================================================
    # Agent Management
    # =========================================================================

    async def _spawn_agent(self, cmd: ParsedCommand) -> Dict:
        """Spawn a new agent with industry-specific buffs"""
        agent_type = cmd.agent
        if not agent_type:
            return {"error": "Agent type required"}

        if not self.active_project:
            return {"error": "No active project. Use 'switch to <project>' first."}

        if agent_type not in self.AGENT_REGISTRY:
            return {
                "error": f"Unknown agent: {agent_type}",
                "available": list(self.AGENT_REGISTRY.keys())
            }

        project = self.projects[self.active_project]

        # Get industry and apply buffs
        from prometheus.core.industries import Industry, agent_enhancer

        try:
            industry = Industry(project.industry)
        except ValueError:
            industry = Industry.GENERAL

        # Get enhanced spawn config with buffs applied
        enhanced_config = agent_enhancer.get_enhanced_spawn_config(
            agent_type=agent_type,
            industry=industry,
            task=None  # Could be passed from command
        )

        # Get active buff names
        buffs = agent_enhancer.get_buffs_for_agent(agent_type, industry)
        buff_names = [b.name for b in buffs]

        # Create agent with buffs
        agent = AgentInfo(
            name=f"{agent_type}_{self.active_project}",
            type=agent_type,
            project=self.active_project,
            status=AgentStatus.RUNNING,
            started_at=datetime.now(),
            capabilities=enhanced_config.get("capabilities", []),
            buffs=buff_names,
            config=enhanced_config
        )

        project.agents[agent_type] = agent

        # Build response
        response = {
            "success": True,
            "message": f"Spawned {agent_type} agent",
            "agent": agent.name,
            "project": self.active_project,
            "industry": project.industry,
        }

        if buff_names:
            response["buffs_applied"] = buff_names
            response["message"] = f"Spawned {agent_type} agent with {len(buff_names)} buff(s)"

        if agent.capabilities:
            response["capabilities"] = agent.capabilities

        return response

    async def _buff_agent(self, cmd: ParsedCommand) -> Dict:
        """Apply additional buffs to an existing agent"""
        agent_type = cmd.agent
        if not self.active_project:
            return {"error": "No active project"}

        project = self.projects[self.active_project]

        if agent_type not in project.agents:
            return {"error": f"Agent '{agent_type}' not running. Spawn it first."}

        from prometheus.core.industries import Industry, agent_enhancer

        try:
            industry = Industry(project.industry)
        except ValueError:
            industry = Industry.GENERAL

        # Re-apply buffs with current industry
        agent = project.agents[agent_type]
        enhanced_config = agent_enhancer.apply_buffs(
            agent_type=agent_type,
            agent_config=agent.config,
            industry=industry
        )

        # Update agent
        buffs = agent_enhancer.get_buffs_for_agent(agent_type, industry)
        agent.buffs = [b.name for b in buffs]
        agent.capabilities = enhanced_config.get("capabilities", [])
        agent.config = enhanced_config

        return {
            "success": True,
            "message": f"Buffed {agent_type} agent for {industry.value}",
            "buffs": agent.buffs,
            "capabilities": agent.capabilities
        }

    async def _set_industry(self, cmd: ParsedCommand) -> Dict:
        """Set the industry concentration for active project"""
        if not self.active_project:
            return {"error": "No active project"}

        from prometheus.core.industries import Industry, INDUSTRY_CONFIGS

        industry_name = cmd.args.get("industry", "").lower()

        # Try to match industry
        matched = None
        for ind in Industry:
            if ind.value == industry_name or industry_name in ind.value:
                matched = ind
                break

        if not matched:
            return {
                "error": f"Unknown industry: {industry_name}",
                "available": [i.value for i in Industry]
            }

        project = self.projects[self.active_project]
        project.industry = matched.value

        config = INDUSTRY_CONFIGS[matched]

        return {
            "success": True,
            "message": f"Set industry to {matched.value}",
            "industry": matched.value,
            "recommended_agents": config.recommended_agents,
            "recommended_plugins": config.recommended_plugins
        }

    async def _list_industries(self, cmd: ParsedCommand) -> Dict:
        """List all available industry concentrations"""
        from prometheus.core.industries import Industry, INDUSTRY_CONFIGS

        industries = []
        for ind in Industry:
            config = INDUSTRY_CONFIGS.get(ind)
            if config:
                industries.append({
                    "id": ind.value,
                    "name": config.name,
                    "description": config.description,
                    "recommended_agents": config.recommended_agents,
                    "recommended_plugins": config.recommended_plugins
                })

        current = None
        if self.active_project:
            current = self.projects[self.active_project].industry

        return {
            "industries": industries,
            "count": len(industries),
            "current": current
        }

    async def _kill_agent(self, cmd: ParsedCommand) -> Dict:
        """Kill an agent"""
        agent_type = cmd.agent
        if not self.active_project:
            return {"error": "No active project"}

        project = self.projects[self.active_project]

        if agent_type not in project.agents:
            return {"error": f"Agent '{agent_type}' not running"}

        del project.agents[agent_type]

        return {
            "success": True,
            "message": f"Killed {agent_type} agent"
        }

    async def _list_agents(self, cmd: ParsedCommand) -> Dict:
        """List all running agents"""
        all_agents = []

        for project in self.projects.values():
            for agent in project.agents.values():
                all_agents.append({
                    "name": agent.name,
                    "type": agent.type,
                    "project": agent.project,
                    "status": agent.status.value,
                    "started": agent.started_at.isoformat() if agent.started_at else None
                })

        return {
            "agents": all_agents,
            "count": len(all_agents)
        }

    async def _agent_status(self, cmd: ParsedCommand) -> Dict:
        """Get status of specific agent"""
        agent_type = cmd.agent
        if not self.active_project:
            return {"error": "No active project"}

        project = self.projects[self.active_project]

        if agent_type not in project.agents:
            return {"error": f"Agent '{agent_type}' not found"}

        agent = project.agents[agent_type]
        return {
            "agent": agent.name,
            "type": agent.type,
            "status": agent.status.value,
            "project": agent.project,
            "started": agent.started_at.isoformat() if agent.started_at else None
        }

    # =========================================================================
    # Plugin Management
    # =========================================================================

    async def _install_plugin(self, cmd: ParsedCommand) -> Dict:
        """Install a plugin"""
        plugin = cmd.plugin
        if not self.active_project:
            return {"error": "No active project"}

        if plugin not in self.PLUGIN_REGISTRY:
            return {
                "error": f"Unknown plugin: {plugin}",
                "available": list(self.PLUGIN_REGISTRY.keys())
            }

        project = self.projects[self.active_project]

        if plugin in project.plugins:
            return {"error": f"Plugin '{plugin}' already installed"}

        project.plugins.append(plugin)

        return {
            "success": True,
            "message": f"Installed {plugin} plugin",
            "description": self.PLUGIN_REGISTRY[plugin]
        }

    async def _remove_plugin(self, cmd: ParsedCommand) -> Dict:
        """Remove a plugin"""
        plugin = cmd.plugin
        if not self.active_project:
            return {"error": "No active project"}

        project = self.projects[self.active_project]

        if plugin not in project.plugins:
            return {"error": f"Plugin '{plugin}' not installed"}

        project.plugins.remove(plugin)

        return {
            "success": True,
            "message": f"Removed {plugin} plugin"
        }

    async def _list_plugins(self, cmd: ParsedCommand) -> Dict:
        """List plugins"""
        installed = []
        if self.active_project:
            installed = self.projects[self.active_project].plugins

        return {
            "installed": installed,
            "available": list(self.PLUGIN_REGISTRY.keys()),
            "project": self.active_project
        }

    # =========================================================================
    # Task Execution
    # =========================================================================

    async def _run_task(self, cmd: ParsedCommand) -> Dict:
        """Run a generic task"""
        task = cmd.args.get("task", "unknown")
        return {
            "message": f"Running task: {task}",
            "project": self.active_project
        }

    async def _fix_error(self, cmd: ParsedCommand) -> Dict:
        """Fix the last error using error agent"""
        if not self.active_project:
            return {"error": "No active project"}

        project = self.projects[self.active_project]

        if "error" not in project.agents:
            # Auto-spawn error agent
            await self._spawn_agent(ParsedCommand(
                type=CommandType.SPAWN_AGENT,
                agent="error"
            ))

        return {
            "message": "Analyzing and fixing last error...",
            "agent": "error"
        }

    async def _build(self, cmd: ParsedCommand) -> Dict:
        """Build the project"""
        if not self.active_project:
            return {"error": "No active project"}

        return {
            "message": f"Building {self.active_project}...",
            "project": self.active_project
        }

    async def _test(self, cmd: ParsedCommand) -> Dict:
        """Run tests"""
        if not self.active_project:
            return {"error": "No active project"}

        return {
            "message": f"Running tests for {self.active_project}...",
            "project": self.active_project
        }

    async def _deploy(self, cmd: ParsedCommand) -> Dict:
        """Deploy the project"""
        target = cmd.target or "production"
        if not self.active_project:
            return {"error": "No active project"}

        return {
            "message": f"Deploying {self.active_project} to {target}...",
            "project": self.active_project,
            "target": target
        }

    async def _status(self, cmd: ParsedCommand) -> Dict:
        """Get system status"""
        total_agents = sum(len(p.agents) for p in self.projects.values())

        return {
            "prometheus": "active",
            "active_project": self.active_project,
            "total_projects": len(self.projects),
            "total_agents": total_agents,
            "workspace": str(self.workspace_root)
        }

    async def _help(self, cmd: ParsedCommand) -> Dict:
        """Show help"""
        return {
            "commands": {
                "projects": [
                    "create project <name>",
                    "switch to <project>",
                    "list projects",
                ],
                "agents": [
                    "spawn <type> agent",
                    "kill <type> agent",
                    "list agents",
                    "buff <type> agent",
                ],
                "industry": [
                    "set industry <type>",
                    "list industries",
                ],
                "plugins": [
                    "install <name> plugin",
                    "remove <name> plugin",
                    "list plugins",
                ],
                "tasks": [
                    "build",
                    "test",
                    "deploy [target]",
                    "fix error",
                ],
                "system": [
                    "status",
                    "help",
                    "go to sleep",
                ]
            }
        }
