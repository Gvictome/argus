"""
Agent orchestration module

Agents:
- Sentinel Prime: Master orchestrator
- Engineer-Bot: Hardware/GPIO specialist
- Code-Mage: Python/AI specialist
- Sentinel-Guard: Security specialist
- DocuBot: Documentation specialist

Error-Solving Agents:
- ErrorRouter: Main orchestrator for error handling
- DependencyAgent: pip/npm/package errors
- NetworkAgent: Port conflicts, connection errors
- SyntaxAgent: Python/JS code errors
- HardwareAgent: GPIO, camera, sensor errors
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


class AgentType(Enum):
    SENTINEL_PRIME = "sentinel_prime"
    ENGINEER_BOT = "engineer_bot"
    CODE_MAGE = "code_mage"
    SENTINEL_GUARD = "sentinel_guard"
    DOCU_BOT = "docu_bot"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    source: AgentType
    target: AgentType
    action: str
    payload: Dict[str, Any]
    priority: int = 5  # 1-10, 10 = urgent
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseAgent:
    """Base class for all agents"""

    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.active = False

    async def start(self):
        """Start the agent"""
        self.active = True

    async def stop(self):
        """Stop the agent"""
        self.active = False

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming message - override in subclass"""
        raise NotImplementedError

    async def send_message(self, target: AgentType, action: str, payload: Dict[str, Any], priority: int = 5) -> AgentMessage:
        """Create a message to send to another agent"""
        return AgentMessage(
            source=self.agent_type,
            target=target,
            action=action,
            payload=payload,
            priority=priority
        )
