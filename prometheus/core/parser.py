"""
Command Parser - Natural language to structured commands
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class CommandType(Enum):
    """Types of commands Prometheus understands"""

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
    BUFF_AGENT = "buff_agent"

    # Industry / Concentration
    SET_INDUSTRY = "set_industry"
    LIST_INDUSTRIES = "list_industries"

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
    """Structured command output from parser"""

    type: CommandType
    project: Optional[str] = None
    agent: Optional[str] = None
    plugin: Optional[str] = None
    target: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    confidence: float = 1.0


class CommandParser:
    """
    Natural language command parser

    Converts spoken/typed commands into structured ParsedCommand objects.

    Examples:
        "create a new project called heimdall"
        "switch to argus"
        "spawn the error agent"
        "install the camera plugin"
        "run tests on argus"
        "fix the last error"
        "what's the status"
    """

    # Command patterns (regex -> CommandType)
    PATTERNS: Dict[CommandType, List[str]] = {
        CommandType.CREATE_PROJECT: [
            r"create (?:a )?(?:new )?project (?:called |named )?(\w+)",
            r"new project (\w+)",
            r"start (?:a )?(?:new )?project (\w+)",
            r"init (?:a )?(?:new )?project (\w+)",
        ],
        CommandType.SWITCH_PROJECT: [
            r"switch to (\w+)",
            r"go to (\w+)",
            r"open (\w+)",
            r"focus (?:on )?(\w+)",
            r"activate (\w+)",
            r"use (\w+)",
        ],
        CommandType.LIST_PROJECTS: [
            r"list (?:all )?projects",
            r"show (?:all )?projects",
            r"what projects",
            r"my projects",
        ],
        CommandType.DELETE_PROJECT: [
            r"delete (?:the )?project (\w+)",
            r"remove (?:the )?project (\w+)",
            r"destroy (\w+)",
        ],
        CommandType.SPAWN_AGENT: [
            r"spawn (?:the )?(\w+)(?: agent)?",
            r"start (?:the )?(\w+) agent",
            r"activate (?:the )?(\w+)(?: agent)?",
            r"wake (?:up )?(?:the )?(\w+)",
            r"run (?:the )?(\w+) agent",
        ],
        CommandType.KILL_AGENT: [
            r"kill (?:the )?(\w+)(?: agent)?",
            r"stop (?:the )?(\w+) agent",
            r"shutdown (?:the )?(\w+)",
            r"terminate (?:the )?(\w+)",
        ],
        CommandType.LIST_AGENTS: [
            r"list (?:all )?agents",
            r"show (?:all )?agents",
            r"what agents",
            r"who(?:'s| is) running",
            r"active agents",
        ],
        CommandType.AGENT_STATUS: [
            r"(\w+) agent status",
            r"status of (?:the )?(\w+)(?: agent)?",
            r"how is (?:the )?(\w+)",
        ],
        CommandType.BUFF_AGENT: [
            r"buff (?:the )?(\w+)(?: agent)?",
            r"enhance (?:the )?(\w+)(?: agent)?",
            r"improve (?:the )?(\w+)(?: agent)?",
            r"upgrade (?:the )?(\w+)(?: agent)?",
            r"boost (?:the )?(\w+)(?: agent)?",
        ],
        CommandType.SET_INDUSTRY: [
            r"set industry (?:to )?(\w+)",
            r"focus on (\w+) industry",
            r"this is (?:a |an )?(\w+) project",
            r"concentration (?:is )?(\w+)",
            r"industry (\w+)",
        ],
        CommandType.LIST_INDUSTRIES: [
            r"list industr(?:ies|y)",
            r"show industr(?:ies|y)",
            r"what industr(?:ies|y)",
            r"available industr(?:ies|y)",
            r"concentrations",
        ],
        CommandType.INSTALL_PLUGIN: [
            r"install (?:the )?(\w+)(?: plugin)?",
            r"add (?:the )?(\w+) plugin",
            r"enable (\w+)",
            r"activate (\w+) plugin",
        ],
        CommandType.REMOVE_PLUGIN: [
            r"remove (?:the )?(\w+)(?: plugin)?",
            r"uninstall (?:the )?(\w+)(?: plugin)?",
            r"disable (\w+)",
            r"deactivate (\w+) plugin",
        ],
        CommandType.LIST_PLUGINS: [
            r"list (?:all )?plugins",
            r"show (?:all )?plugins",
            r"what plugins",
            r"available plugins",
        ],
        CommandType.RUN_TASK: [
            r"run (\w+)",
            r"execute (\w+)",
            r"do (\w+)",
            r"perform (\w+)",
        ],
        CommandType.FIX_ERROR: [
            r"fix (?:the )?(?:last )?error",
            r"resolve (?:the )?(?:last )?error",
            r"debug (?:the )?(?:last )?(?:error)?",
            r"what went wrong",
            r"analyze (?:the )?error",
        ],
        CommandType.BUILD: [
            r"build(?: the)?(?: project)?",
            r"compile",
            r"make",
            r"package",
        ],
        CommandType.TEST: [
            r"(?:run )?tests?",
            r"test(?: the)?(?: project)?",
            r"check tests?",
            r"verify",
        ],
        CommandType.DEPLOY: [
            r"deploy(?: to)? ?(\w+)?",
            r"ship(?: it)?",
            r"push to (\w+)",
            r"release(?: to)? ?(\w+)?",
        ],
        CommandType.STATUS: [
            r"status",
            r"what(?:'s| is) (?:the )?status",
            r"how are we doing",
            r"system status",
            r"overview",
        ],
        CommandType.HELP: [
            r"help",
            r"what can you do",
            r"commands?",
            r"options?",
        ],
        CommandType.SLEEP: [
            r"go to sleep",
            r"stop listening",
            r"shutdown",
            r"goodbye",
            r"sleep",
            r"quiet",
        ],
    }

    # Known agent types
    AGENT_TYPES = [
        "error", "dependency", "network", "syntax", "hardware",
        "camera", "detection", "automation", "security",
        "build", "test", "deploy", "git", "docker"
    ]

    # Known plugin types
    PLUGIN_TYPES = [
        "camera", "detection", "automation", "voice",
        "mqtt", "homeassistant", "telegram", "discord",
        "database", "scheduler", "git", "docker"
    ]

    def parse(self, text: str) -> ParsedCommand:
        """
        Parse natural language command

        Args:
            text: Raw command text

        Returns:
            ParsedCommand with structured data
        """
        text_lower = text.lower().strip()

        # Try each pattern
        for cmd_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return self._build_command(cmd_type, match, text)

        # Unknown command
        return ParsedCommand(
            type=CommandType.UNKNOWN,
            raw_text=text,
            confidence=0.0
        )

    def _build_command(
        self,
        cmd_type: CommandType,
        match: re.Match,
        raw_text: str
    ) -> ParsedCommand:
        """Build structured command from regex match"""
        groups = match.groups()

        cmd = ParsedCommand(
            type=cmd_type,
            raw_text=raw_text,
            confidence=1.0
        )

        # Extract entities based on command type
        if cmd_type in [CommandType.CREATE_PROJECT, CommandType.SWITCH_PROJECT,
                        CommandType.DELETE_PROJECT]:
            cmd.project = groups[0] if groups else None

        elif cmd_type in [CommandType.SPAWN_AGENT, CommandType.KILL_AGENT,
                          CommandType.AGENT_STATUS, CommandType.BUFF_AGENT]:
            agent = groups[0] if groups else None
            # Normalize agent name
            if agent:
                agent = self._normalize_agent(agent)
            cmd.agent = agent

        elif cmd_type == CommandType.SET_INDUSTRY:
            industry = groups[0] if groups else None
            cmd.args["industry"] = industry

        elif cmd_type in [CommandType.INSTALL_PLUGIN, CommandType.REMOVE_PLUGIN]:
            plugin = groups[0] if groups else None
            # Normalize plugin name
            if plugin:
                plugin = self._normalize_plugin(plugin)
            cmd.plugin = plugin

        elif cmd_type == CommandType.DEPLOY:
            cmd.target = groups[0] if groups and groups[0] else "production"

        elif cmd_type == CommandType.RUN_TASK:
            cmd.args["task"] = groups[0] if groups else None

        return cmd

    def _normalize_agent(self, agent: str) -> str:
        """Normalize agent name to standard form"""
        agent = agent.lower().strip()

        # Remove common suffixes
        agent = re.sub(r"[ _-]?agent$", "", agent)

        # Map aliases
        aliases = {
            "err": "error",
            "dep": "dependency",
            "deps": "dependency",
            "net": "network",
            "hw": "hardware",
            "cam": "camera",
            "detect": "detection",
            "auto": "automation",
            "sec": "security",
        }

        return aliases.get(agent, agent)

    def _normalize_plugin(self, plugin: str) -> str:
        """Normalize plugin name to standard form"""
        plugin = plugin.lower().strip()

        # Remove common suffixes
        plugin = re.sub(r"[ _-]?plugin$", "", plugin)

        # Map aliases
        aliases = {
            "cam": "camera",
            "detect": "detection",
            "auto": "automation",
            "ha": "homeassistant",
            "tg": "telegram",
            "db": "database",
            "sched": "scheduler",
        }

        return aliases.get(plugin, plugin)

    def get_suggestions(self, partial: str) -> List[str]:
        """Get command suggestions for partial input"""
        suggestions = []
        partial_lower = partial.lower()

        # Check for matching patterns
        for cmd_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                # Convert regex to readable form
                readable = pattern.replace(r"(?:", "").replace(")?", "")
                readable = re.sub(r"\(\w+\)", "<name>", readable)
                readable = readable.replace("\\w+", "<name>")

                if partial_lower in readable:
                    suggestions.append(readable)

        return suggestions[:5]  # Return top 5
