"""
Core module for Prometheus

Components:
- CommandParser: Natural language command parsing
- AgentRouter: Routes commands to agents across projects
- ProjectManager: Manages project lifecycle
"""

from prometheus.core.parser import CommandParser, CommandType, ParsedCommand
from prometheus.core.router import AgentRouter

__all__ = ["CommandParser", "CommandType", "ParsedCommand", "AgentRouter"]
