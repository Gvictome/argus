"""
Network Agent - Handles port conflicts, connection errors
"""

import re
import platform
from typing import Optional, Tuple
from src.agents.error_router import BaseErrorAgent, ErrorCategory, ErrorReport, FixResult


class NetworkAgent(BaseErrorAgent):
    """
    Specialized agent for network-related errors

    Handles:
    - Port already in use
    - Connection refused
    - Timeout errors
    - DNS resolution failures
    """

    def __init__(self):
        super().__init__(ErrorCategory.NETWORK)
        self.is_windows = platform.system() == "Windows"

    def suggest_fix(self, report: ErrorReport) -> str:
        """Generate fix suggestion for network error"""
        error = report.raw_message.lower()

        # Port in use
        if "10048" in report.raw_message or "address already in use" in error:
            port = self._extract_port(report.raw_message)
            if port:
                if self.is_windows:
                    return f"netstat -ano | findstr :{port} to find PID, then taskkill /PID <pid> /F"
                else:
                    return f"lsof -i :{port} to find process, then kill -9 <pid>"
            return "Find and kill the process using the port, or use a different port"

        # Connection refused
        if "connection refused" in error or "econnrefused" in error:
            return "Ensure the target service is running and accepting connections"

        # Timeout
        if "timeout" in error or "etimedout" in error:
            return "Check network connectivity and firewall rules"

        # DNS
        if "getaddrinfo" in error or "name resolution" in error:
            return "Check DNS settings and network connectivity"

        return "Check network configuration and firewall settings"

    def can_auto_fix(self, report: ErrorReport) -> bool:
        """Check if this network error can be auto-fixed"""
        # Port conflicts can be auto-fixed
        return report.subcategory == "port_in_use"

    def execute_fix(self, report: ErrorReport) -> FixResult:
        """Execute the network fix"""
        if report.subcategory == "port_in_use":
            return self._fix_port_conflict(report)

        return FixResult(
            success=False,
            action_taken="none",
            error="Manual intervention required"
        )

    def _fix_port_conflict(self, report: ErrorReport) -> FixResult:
        """Fix a port-in-use error by killing the blocking process"""
        port = self._extract_port(report.raw_message)
        if not port:
            return FixResult(
                success=False,
                action_taken="none",
                error="Could not extract port number from error"
            )

        # Find the process using the port
        pid = self._find_process_on_port(port)
        if not pid:
            return FixResult(
                success=False,
                action_taken=f"Searched for process on port {port}",
                error="Could not find process using port"
            )

        # Kill the process
        if self.is_windows:
            kill_cmd = f"taskkill /PID {pid} /F"
        else:
            kill_cmd = f"kill -9 {pid}"

        success, output = self.run_command(kill_cmd)

        if success:
            return FixResult(
                success=True,
                action_taken=f"Killed process {pid} on port {port}",
                output=output
            )
        else:
            return FixResult(
                success=False,
                action_taken=f"Attempted to kill process {pid}",
                error=output
            )

    def _extract_port(self, error_message: str) -> Optional[int]:
        """Extract port number from error message"""
        patterns = [
            r"port[:\s]+(\d+)",
            r":(\d+)\)",
            r"address.*:(\d+)",
            r"bind.*:(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                port = int(match.group(1))
                if 1 <= port <= 65535:
                    return port

        # Default common ports based on error context
        if "8000" in error_message:
            return 8000
        if "3000" in error_message:
            return 3000

        return None

    def _find_process_on_port(self, port: int) -> Optional[int]:
        """Find the PID of process using a port"""
        if self.is_windows:
            cmd = f"netstat -ano | findstr :{port}"
        else:
            cmd = f"lsof -t -i:{port}"

        success, output = self.run_command(cmd)
        if not success or not output:
            return None

        if self.is_windows:
            # Parse netstat output: TCP 0.0.0.0:8000 0.0.0.0:0 LISTENING 12345
            lines = output.strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in line:
                    try:
                        return int(parts[-1])
                    except ValueError:
                        continue
        else:
            # lsof -t returns just the PID
            try:
                return int(output.strip().split('\n')[0])
            except ValueError:
                pass

        return None

    def check_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return True
        except OSError:
            return False

    def find_available_port(self, start_port: int = 8000, max_attempts: int = 100) -> Optional[int]:
        """Find an available port starting from start_port"""
        for offset in range(max_attempts):
            port = start_port + offset
            if self.check_port_available(port):
                return port
        return None


# Export
__all__ = ["NetworkAgent"]
