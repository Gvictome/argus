"""
Hardware Agent - Handles GPIO, camera, sensor errors
"""

import re
import platform
from typing import Optional, Dict, List
from src.agents.error_router import BaseErrorAgent, ErrorCategory, ErrorReport, FixResult


class HardwareAgent(BaseErrorAgent):
    """
    Specialized agent for hardware-related errors

    Handles:
    - Camera initialization failures
    - GPIO access errors
    - I2C/SPI communication issues
    - Sensor read failures
    - Device not found errors
    """

    # Hardware error patterns and fixes
    HARDWARE_FIXES = {
        "camera": {
            "not found": "Check camera ribbon cable connection and enable camera in raspi-config",
            "busy": "Another process is using the camera. Kill it with: sudo pkill -f libcamera",
            "permission": "Add user to video group: sudo usermod -a -G video $USER",
            "mmal": "Legacy camera. Enable legacy camera in raspi-config or use libcamera",
        },
        "gpio": {
            "permission": "Run with sudo or add user to gpio group: sudo usermod -a -G gpio $USER",
            "busy": "GPIO pin already in use. Check for conflicting processes.",
            "not found": "Check if running on Raspberry Pi. GPIO not available on other systems.",
        },
        "i2c": {
            "not found": "Enable I2C in raspi-config: sudo raspi-config -> Interface Options -> I2C",
            "permission": "Add user to i2c group: sudo usermod -a -G i2c $USER",
            "no device": "Check I2C device address with: i2cdetect -y 1",
        },
        "spi": {
            "not found": "Enable SPI in raspi-config: sudo raspi-config -> Interface Options -> SPI",
            "permission": "Add user to spi group: sudo usermod -a -G spi $USER",
        },
    }

    def __init__(self):
        super().__init__(ErrorCategory.HARDWARE)
        self.is_raspberry_pi = self._check_raspberry_pi()

    def _check_raspberry_pi(self) -> bool:
        """Check if running on Raspberry Pi"""
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                return "Raspberry Pi" in cpuinfo or "BCM" in cpuinfo
        except:
            return False

    def suggest_fix(self, report: ErrorReport) -> str:
        """Generate fix suggestion for hardware error"""
        error = report.raw_message.lower()

        # Camera errors
        if any(cam in error for cam in ["camera", "libcamera", "picamera", "mmal"]):
            return self._suggest_camera_fix(error)

        # GPIO errors
        if "gpio" in error:
            return self._suggest_gpio_fix(error)

        # I2C errors
        if "i2c" in error:
            return self._suggest_i2c_fix(error)

        # SPI errors
        if "spi" in error:
            return self._suggest_spi_fix(error)

        # Generic device errors
        if "device" in error:
            return self._suggest_device_fix(error)

        return "Check hardware connections and ensure drivers are installed"

    def can_auto_fix(self, report: ErrorReport) -> bool:
        """
        Most hardware errors require physical intervention
        Some service-level fixes can be automated
        """
        error = report.raw_message.lower()

        # Can auto-fix: killing processes that block hardware
        if "busy" in error or "in use" in error:
            return True

        # Can auto-fix: enabling interfaces via raspi-config non-interactively
        # Actually this requires reboot, so not fully auto
        return False

    def execute_fix(self, report: ErrorReport) -> FixResult:
        """Execute the hardware fix"""
        error = report.raw_message.lower()

        # Try to kill blocking processes
        if "camera" in error and ("busy" in error or "in use" in error):
            return self._kill_camera_processes()

        if "gpio" in error and "busy" in error:
            return self._free_gpio()

        # Most hardware issues need manual intervention
        suggestion = self.suggest_fix(report)
        diagnostics = self._run_diagnostics()

        return FixResult(
            success=False,
            action_taken="diagnostic",
            output=f"Suggestion: {suggestion}\n\nDiagnostics:\n{diagnostics}",
            error="Manual hardware intervention may be required"
        )

    def _suggest_camera_fix(self, error: str) -> str:
        """Suggest fix for camera errors"""
        if "not found" in error or "no cameras" in error:
            return self.HARDWARE_FIXES["camera"]["not found"]

        if "busy" in error or "in use" in error:
            return self.HARDWARE_FIXES["camera"]["busy"]

        if "permission" in error or "access" in error:
            return self.HARDWARE_FIXES["camera"]["permission"]

        if "mmal" in error:
            return self.HARDWARE_FIXES["camera"]["mmal"]

        return "Check camera connection: 1) Power off Pi, 2) Reseat ribbon cable, 3) Enable camera in raspi-config"

    def _suggest_gpio_fix(self, error: str) -> str:
        """Suggest fix for GPIO errors"""
        if "permission" in error or "access" in error:
            return self.HARDWARE_FIXES["gpio"]["permission"]

        if "busy" in error or "in use" in error:
            return self.HARDWARE_FIXES["gpio"]["busy"]

        return self.HARDWARE_FIXES["gpio"]["not found"]

    def _suggest_i2c_fix(self, error: str) -> str:
        """Suggest fix for I2C errors"""
        if "no such file" in error or "not found" in error:
            return self.HARDWARE_FIXES["i2c"]["not found"]

        if "permission" in error:
            return self.HARDWARE_FIXES["i2c"]["permission"]

        if "no device" in error or "no ack" in error:
            return self.HARDWARE_FIXES["i2c"]["no device"]

        return "Check I2C wiring and device address"

    def _suggest_spi_fix(self, error: str) -> str:
        """Suggest fix for SPI errors"""
        if "no such file" in error or "not found" in error:
            return self.HARDWARE_FIXES["spi"]["not found"]

        if "permission" in error:
            return self.HARDWARE_FIXES["spi"]["permission"]

        return "Check SPI wiring and enable SPI in raspi-config"

    def _suggest_device_fix(self, error: str) -> str:
        """Suggest fix for generic device errors"""
        if "not found" in error or "no such" in error:
            return "Device not detected. Check physical connection and power."

        if "busy" in error:
            return "Device in use by another process. Check running processes."

        if "permission" in error:
            return "Permission denied. Run with sudo or add user to appropriate group."

        return "Check device connection, power, and drivers"

    def _kill_camera_processes(self) -> FixResult:
        """Kill processes blocking the camera"""
        commands = [
            "sudo pkill -f libcamera",
            "sudo pkill -f raspistill",
            "sudo pkill -f raspivid",
            "sudo pkill -f picamera",
        ]

        killed = []
        for cmd in commands:
            success, output = self.run_command(cmd)
            if success:
                killed.append(cmd)

        if killed:
            return FixResult(
                success=True,
                action_taken=f"Killed camera processes: {', '.join(killed)}",
                output="Camera should now be available"
            )
        else:
            return FixResult(
                success=False,
                action_taken="Attempted to kill camera processes",
                error="No camera processes found to kill"
            )

    def _free_gpio(self) -> FixResult:
        """Attempt to free GPIO resources"""
        # This is tricky - can't really force-free GPIO without knowing the pin
        return FixResult(
            success=False,
            action_taken="diagnostic",
            error="GPIO conflict requires identifying and stopping the conflicting process"
        )

    def _run_diagnostics(self) -> str:
        """Run hardware diagnostics"""
        lines = []

        if not self.is_raspberry_pi:
            lines.append("WARNING: Not running on Raspberry Pi")
            lines.append("Hardware features require Raspberry Pi")
            return "\n".join(lines)

        lines.append("Running on Raspberry Pi")

        # Check camera
        success, output = self.run_command("libcamera-hello --list-cameras 2>&1 || echo 'libcamera not available'")
        lines.append(f"\nCamera check:\n{output[:200]}")

        # Check I2C
        success, output = self.run_command("ls /dev/i2c* 2>&1 || echo 'I2C not enabled'")
        lines.append(f"\nI2C devices: {output.strip()}")

        # Check SPI
        success, output = self.run_command("ls /dev/spidev* 2>&1 || echo 'SPI not enabled'")
        lines.append(f"\nSPI devices: {output.strip()}")

        # Check GPIO
        success, output = self.run_command("ls /dev/gpiomem 2>&1 || echo 'GPIO not available'")
        lines.append(f"\nGPIO: {output.strip()}")

        return "\n".join(lines)

    def get_system_info(self) -> Dict:
        """Get hardware system information"""
        info = {
            "is_raspberry_pi": self.is_raspberry_pi,
            "platform": platform.system(),
            "machine": platform.machine(),
        }

        if self.is_raspberry_pi:
            # Get Pi model
            try:
                with open("/proc/device-tree/model", "r") as f:
                    info["model"] = f.read().strip('\x00')
            except:
                info["model"] = "Unknown"

        return info


# Export
__all__ = ["HardwareAgent"]
