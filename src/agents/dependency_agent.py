"""
Dependency Agent - Handles pip/npm/package installation errors
"""

import re
import sys
from typing import List, Optional
from src.agents.error_router import BaseErrorAgent, ErrorCategory, ErrorReport, FixResult


class DependencyAgent(BaseErrorAgent):
    """
    Specialized agent for dependency/package errors

    Handles:
    - pip install failures
    - Missing modules
    - Version conflicts
    - Build/compile errors
    """

    # Names that may be passed to pip. Must start with a letter or digit so a
    # name can never be read as a pip option (e.g. "-e" or "--index-url").
    SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

    # Known fixes for specific packages: package -> (pip install args)
    PACKAGE_FIXES = {
        "numpy": ["numpy", "--only-binary=:all:"],
        "opencv": ["opencv-python-headless", "--only-binary=:all:"],
        "opencv-python": ["opencv-python-headless", "--only-binary=:all:"],
        "pillow": ["Pillow", "--only-binary=:all:"],
        "cryptography": ["cryptography", "--only-binary=:all:"],
    }

    def __init__(self):
        super().__init__(ErrorCategory.DEPENDENCY)

    @classmethod
    def is_safe_module_name(cls, name: Optional[str]) -> bool:
        """True only for plain package names that are safe to hand to pip."""
        return bool(name) and bool(cls.SAFE_NAME.match(name)) and name[0].isalnum()

    def suggest_fix(self, report: ErrorReport) -> str:
        """Generate fix suggestion for dependency error (text only, never run)"""
        error = report.raw_message.lower()

        install_args = self._install_args(report)
        if install_args:
            return "pip install " + " ".join(install_args)

        # Missing module
        if "no module named" in error or "modulenotfounderror" in error:
            return "Install the missing module with pip"

        # Metadata generation failed
        if "metadata-generation-failed" in error:
            return "Install a prebuilt wheel: pip install <package> --only-binary=:all:"

        # Version conflict
        if "version" in error and "conflict" in error:
            return "pip install --upgrade <package> or check requirements.txt for conflicts"

        # General pip error
        if "pip" in error:
            return "Try: pip install --upgrade pip, then retry the install"

        return "Check package name and try reinstalling"

    def can_auto_fix(self, report: ErrorReport) -> bool:
        """Check if this dependency error can be auto-fixed"""
        subcategory = report.subcategory

        # Only fixes that install a known package from the fixed table are
        # automatic. A missing module's name comes from the posted text, so it
        # is never auto-fixed (execute_fix still needs explicit approval).
        safe_fixes = ["numpy_compile", "metadata_failed"]
        return subcategory in safe_fixes and self._install_args(report) is not None

    def execute_fix(self, report: ErrorReport) -> FixResult:
        """Execute the dependency fix"""
        install_args = self._install_args(report)

        if not install_args:
            return FixResult(
                success=False,
                action_taken="none",
                error="Could not determine a safe fix command"
            )

        # Fixed argument list, no shell: the package name was validated above.
        args = [sys.executable, "-m", "pip", "install", *install_args]
        success, output = self.run_command(args)

        return FixResult(
            success=success,
            action_taken="pip install " + " ".join(install_args),
            output=output if success else None,
            error=output if not success else None
        )

    def _install_args(self, report: ErrorReport) -> Optional[List[str]]:
        """pip install arguments for this error, or None if there is no safe fix."""
        error = report.raw_message.lower()

        if "no module named" in error or "modulenotfounderror" in error:
            module = self._extract_module_name(report.raw_message)
            return [module] if self.is_safe_module_name(module) else None

        if "metadata-generation-failed" in error or report.subcategory == "numpy_compile":
            package = self._extract_package_name(report.raw_message)
            return list(self.PACKAGE_FIXES[package]) if package in self.PACKAGE_FIXES else None

        return None

    def _extract_module_name(self, error_message: str) -> Optional[str]:
        """Extract module name from import error"""
        # Pattern: No module named 'xyz' or ModuleNotFoundError: No module named 'xyz'
        patterns = [
            r"No module named ['\"]([^'\"]+)['\"]",
            r"No module named (\w+)",
            r"ModuleNotFoundError:.*['\"]([^'\"]+)['\"]",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                module = match.group(1)
                # Handle submodule imports (e.g., 'foo.bar' -> 'foo')
                return module.split('.')[0]

        return None

    def _extract_package_name(self, error_message: str) -> Optional[str]:
        """Extract package name from pip error"""
        # Look for package name near the error
        patterns = [
            r"error:.*package[:\s]+(\w+)",
            r"╰─>\s*(\w+)",
            r"Building wheel for (\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        # Check for known packages in the message
        known_packages = ["numpy", "opencv", "pillow", "cryptography", "scipy", "pandas"]
        error_lower = error_message.lower()
        for pkg in known_packages:
            if pkg in error_lower:
                return pkg

        return None


# Export
__all__ = ["DependencyAgent"]
