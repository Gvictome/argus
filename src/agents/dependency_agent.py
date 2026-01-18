"""
Dependency Agent - Handles pip/npm/package installation errors
"""

import re
from typing import Optional
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

    # Known fixes for specific packages
    PACKAGE_FIXES = {
        "numpy": {
            "metadata_failed": "pip install numpy --only-binary=all",
            "compile_error": "pip install numpy --only-binary=all",
        },
        "opencv-python": {
            "metadata_failed": "pip install opencv-python-headless --only-binary=all",
        },
        "pillow": {
            "compile_error": "pip install Pillow --only-binary=all",
        },
        "cryptography": {
            "compile_error": "pip install cryptography --only-binary=all",
        },
    }

    def __init__(self):
        super().__init__(ErrorCategory.DEPENDENCY)

    def suggest_fix(self, report: ErrorReport) -> str:
        """Generate fix suggestion for dependency error"""
        error = report.raw_message.lower()

        # Missing module
        if "no module named" in error or "modulenotfounderror" in error:
            module = self._extract_module_name(report.raw_message)
            if module:
                return f"pip install {module}"
            return "Install the missing module with pip"

        # Metadata generation failed
        if "metadata-generation-failed" in error:
            package = self._extract_package_name(report.raw_message)
            if package and package in self.PACKAGE_FIXES:
                fix = self.PACKAGE_FIXES[package].get("metadata_failed")
                if fix:
                    return fix
            return f"pip install {package} --only-binary=all"

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

        # Safe to auto-fix
        safe_fixes = ["numpy_compile", "metadata_failed", "missing_module"]
        return subcategory in safe_fixes

    def execute_fix(self, report: ErrorReport) -> FixResult:
        """Execute the dependency fix"""
        fix_command = self.suggest_fix(report)

        if not fix_command or fix_command.startswith("Check"):
            return FixResult(
                success=False,
                action_taken="none",
                error="Could not determine fix command"
            )

        # Execute the fix
        success, output = self.run_command(fix_command)

        return FixResult(
            success=success,
            action_taken=fix_command,
            output=output if success else None,
            error=output if not success else None
        )

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
