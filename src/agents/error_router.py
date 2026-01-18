"""
Error Router - Main orchestrator for error-solving agents

Routes errors to specialized agents based on error type detection.
"""

import re
import subprocess
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime


class ErrorCategory(Enum):
    DEPENDENCY = "dependency"      # pip, npm, package issues
    NETWORK = "network"            # port conflicts, connection errors
    SYNTAX = "syntax"              # code errors, import errors
    HARDWARE = "hardware"          # GPIO, camera, sensors
    PERMISSION = "permission"      # file access, admin rights
    UNKNOWN = "unknown"


@dataclass
class ErrorReport:
    """Structured error information"""
    raw_message: str
    category: ErrorCategory
    subcategory: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class FixResult:
    """Result of an attempted fix"""
    success: bool
    action_taken: str
    output: Optional[str] = None
    error: Optional[str] = None


class ErrorRouter:
    """
    Main error routing system

    Analyzes errors and delegates to specialized agents
    """

    # Error pattern matchers
    PATTERNS = {
        ErrorCategory.DEPENDENCY: [
            r"pip.*error",
            r"npm.*ERR",
            r"ModuleNotFoundError",
            r"No module named",
            r"metadata-generation-failed",
            r"Could not find a version",
            r"package.*not found",
        ],
        ErrorCategory.NETWORK: [
            r"WinError 10048",
            r"address already in use",
            r"Connection refused",
            r"ECONNREFUSED",
            r"ETIMEDOUT",
            r"bind.*failed",
            r"port.*in use",
        ],
        ErrorCategory.SYNTAX: [
            r"SyntaxError",
            r"IndentationError",
            r"NameError",
            r"TypeError",
            r"AttributeError",
            r"ImportError",
            r"ValueError",
        ],
        ErrorCategory.HARDWARE: [
            r"GPIO",
            r"camera.*not found",
            r"libcamera",
            r"picamera",
            r"device.*busy",
            r"No such device",
            r"I2C",
            r"SPI",
        ],
        ErrorCategory.PERMISSION: [
            r"Permission denied",
            r"Access.*denied",
            r"EACCES",
            r"requires.*admin",
            r"Operation not permitted",
        ],
    }

    def __init__(self):
        self.agents: Dict[ErrorCategory, "BaseErrorAgent"] = {}
        self.error_history: List[ErrorReport] = []
        self._register_agents()

    def _register_agents(self):
        """Register all specialized error agents"""
        from src.agents.dependency_agent import DependencyAgent
        from src.agents.network_agent import NetworkAgent
        from src.agents.syntax_agent import SyntaxAgent
        from src.agents.hardware_agent import HardwareAgent

        self.agents[ErrorCategory.DEPENDENCY] = DependencyAgent()
        self.agents[ErrorCategory.NETWORK] = NetworkAgent()
        self.agents[ErrorCategory.SYNTAX] = SyntaxAgent()
        self.agents[ErrorCategory.HARDWARE] = HardwareAgent()

    def analyze(self, error_message: str) -> ErrorReport:
        """
        Analyze an error message and categorize it

        Args:
            error_message: Raw error text

        Returns:
            ErrorReport with categorization and suggestions
        """
        category = self._detect_category(error_message)
        subcategory = self._detect_subcategory(error_message, category)
        file_path, line_number = self._extract_location(error_message)

        report = ErrorReport(
            raw_message=error_message,
            category=category,
            subcategory=subcategory,
            file_path=file_path,
            line_number=line_number,
        )

        # Get fix suggestion from appropriate agent
        if category in self.agents:
            agent = self.agents[category]
            report.suggested_fix = agent.suggest_fix(report)
            report.auto_fixable = agent.can_auto_fix(report)

        self.error_history.append(report)
        return report

    def fix(self, error_message: str, auto_approve: bool = False) -> FixResult:
        """
        Attempt to fix an error

        Args:
            error_message: Raw error text
            auto_approve: If True, apply fix without confirmation

        Returns:
            FixResult with outcome
        """
        report = self.analyze(error_message)

        if report.category == ErrorCategory.UNKNOWN:
            return FixResult(
                success=False,
                action_taken="none",
                error="Could not categorize error"
            )

        if report.category not in self.agents:
            return FixResult(
                success=False,
                action_taken="none",
                error=f"No agent registered for {report.category.value}"
            )

        agent = self.agents[report.category]

        if not report.auto_fixable and not auto_approve:
            return FixResult(
                success=False,
                action_taken="none",
                error=f"Manual fix required: {report.suggested_fix}"
            )

        return agent.execute_fix(report)

    def _detect_category(self, error_message: str) -> ErrorCategory:
        """Detect error category from message"""
        error_lower = error_message.lower()

        for category, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    return category

        return ErrorCategory.UNKNOWN

    def _detect_subcategory(self, error_message: str, category: ErrorCategory) -> Optional[str]:
        """Detect specific error subtype"""
        if category == ErrorCategory.DEPENDENCY:
            if "numpy" in error_message.lower():
                return "numpy_compile"
            if "metadata-generation-failed" in error_message:
                return "metadata_failed"
            if "No module named" in error_message:
                return "missing_module"

        elif category == ErrorCategory.NETWORK:
            if "10048" in error_message or "address already in use" in error_message.lower():
                return "port_in_use"
            if "refused" in error_message.lower():
                return "connection_refused"

        elif category == ErrorCategory.SYNTAX:
            if "IndentationError" in error_message:
                return "indentation"
            if "ImportError" in error_message:
                return "import"

        return None

    def _extract_location(self, error_message: str) -> Tuple[Optional[str], Optional[int]]:
        """Extract file path and line number from error"""
        # Python traceback pattern: File "path", line N
        match = re.search(r'File "([^"]+)", line (\d+)', error_message)
        if match:
            return match.group(1), int(match.group(2))

        # Generic pattern: filename:line
        match = re.search(r'([^\s:]+\.py):(\d+)', error_message)
        if match:
            return match.group(1), int(match.group(2))

        return None, None

    def get_history(self, limit: int = 10) -> List[ErrorReport]:
        """Get recent error history"""
        return self.error_history[-limit:]

    def clear_history(self):
        """Clear error history"""
        self.error_history.clear()


class BaseErrorAgent:
    """Base class for specialized error agents"""

    def __init__(self, category: ErrorCategory):
        self.category = category

    def suggest_fix(self, report: ErrorReport) -> str:
        """Suggest a fix for the error - override in subclass"""
        raise NotImplementedError

    def can_auto_fix(self, report: ErrorReport) -> bool:
        """Check if error can be auto-fixed - override in subclass"""
        return False

    def execute_fix(self, report: ErrorReport) -> FixResult:
        """Execute the fix - override in subclass"""
        raise NotImplementedError

    def run_command(self, command: str, shell: bool = True) -> Tuple[bool, str]:
        """Run a shell command and return result"""
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=120
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)


# Singleton instance
error_router = ErrorRouter()
