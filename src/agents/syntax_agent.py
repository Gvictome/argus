"""
Syntax Agent - Handles Python/JavaScript code errors
"""

import re
from typing import Optional, List
from src.agents.error_router import BaseErrorAgent, ErrorCategory, ErrorReport, FixResult


class SyntaxAgent(BaseErrorAgent):
    """
    Specialized agent for code syntax and runtime errors

    Handles:
    - SyntaxError
    - IndentationError
    - NameError
    - TypeError
    - AttributeError
    - ImportError
    """

    # Common fixes for error patterns
    COMMON_FIXES = {
        "unexpected EOF": "Check for missing closing brackets, parentheses, or quotes",
        "invalid syntax": "Check for typos, missing colons, or incorrect operators",
        "expected an indented block": "Add proper indentation after if/for/def/class statements",
        "unindent does not match": "Fix inconsistent indentation (spaces vs tabs)",
        "name .* is not defined": "Check variable spelling or add missing import",
        "has no attribute": "Check attribute name spelling or object type",
        "object is not callable": "Remove parentheses or check if object is a function",
        "missing .* required positional argument": "Add the required argument to the function call",
        "takes .* positional arguments but .* were given": "Remove extra arguments from function call",
        "cannot import name": "Check import path and module structure",
        "circular import": "Restructure imports or use lazy imports",
    }

    def __init__(self):
        super().__init__(ErrorCategory.SYNTAX)

    def suggest_fix(self, report: ErrorReport) -> str:
        """Generate fix suggestion for syntax/runtime error"""
        error = report.raw_message

        # Check against known patterns
        for pattern, fix in self.COMMON_FIXES.items():
            if re.search(pattern, error, re.IGNORECASE):
                location = self._format_location(report)
                return f"{fix}{location}"

        # Specific error types
        if "SyntaxError" in error:
            return self._analyze_syntax_error(error, report)

        if "IndentationError" in error:
            return self._analyze_indentation_error(error, report)

        if "ImportError" in error or "ModuleNotFoundError" in error:
            return self._analyze_import_error(error, report)

        if "TypeError" in error:
            return self._analyze_type_error(error, report)

        if "AttributeError" in error:
            return self._analyze_attribute_error(error, report)

        if "NameError" in error:
            return self._analyze_name_error(error, report)

        return f"Review the error at {self._format_location(report)}"

    def can_auto_fix(self, report: ErrorReport) -> bool:
        """
        Syntax errors generally cannot be auto-fixed safely
        They require human review of the code
        """
        # Only very specific patterns could be auto-fixed
        # For now, be conservative
        return False

    def execute_fix(self, report: ErrorReport) -> FixResult:
        """Execute the syntax fix - mostly diagnostic"""
        suggestion = self.suggest_fix(report)

        # Provide detailed diagnostic instead of auto-fix
        diagnostics = self._generate_diagnostics(report)

        return FixResult(
            success=False,
            action_taken="diagnostic",
            output=f"Suggestion: {suggestion}\n\nDiagnostics:\n{diagnostics}",
            error="Manual code fix required"
        )

    def _analyze_syntax_error(self, error: str, report: ErrorReport) -> str:
        """Analyze SyntaxError details"""
        location = self._format_location(report)

        if "EOL while scanning string" in error:
            return f"Missing closing quote for string{location}"

        if "unexpected EOF" in error:
            return f"Missing closing bracket, parenthesis, or quote{location}"

        if "invalid character" in error:
            return f"Remove invalid character (possibly copied from web/doc){location}"

        if "f-string" in error:
            return f"Check f-string syntax - no backslashes or nested quotes{location}"

        return f"Check syntax near{location}"

    def _analyze_indentation_error(self, error: str, report: ErrorReport) -> str:
        """Analyze IndentationError details"""
        location = self._format_location(report)

        if "expected an indented block" in error:
            return f"Add 4 spaces of indentation after the colon{location}"

        if "unindent does not match" in error:
            return f"Fix indentation level - ensure consistent use of spaces (not tabs){location}"

        if "unexpected indent" in error:
            return f"Remove extra indentation{location}"

        return f"Fix indentation{location} - use 4 spaces per level"

    def _analyze_import_error(self, error: str, report: ErrorReport) -> str:
        """Analyze ImportError details"""
        # Extract module name
        match = re.search(r"cannot import name ['\"]?(\w+)['\"]?", error)
        if match:
            name = match.group(1)
            return f"'{name}' doesn't exist in the module. Check spelling or module version."

        match = re.search(r"No module named ['\"]?([^'\"]+)['\"]?", error)
        if match:
            module = match.group(1)
            return f"Install missing module: pip install {module.split('.')[0]}"

        return "Check import statement and module installation"

    def _analyze_type_error(self, error: str, report: ErrorReport) -> str:
        """Analyze TypeError details"""
        if "NoneType" in error:
            return "A variable is None when it shouldn't be. Add a null check."

        if "not subscriptable" in error:
            return "Can't use [] on this type. Check the object type."

        if "not iterable" in error:
            return "Can't iterate over this object. Check if it's a list/tuple/dict."

        if "argument" in error:
            return "Wrong number or type of arguments passed to function."

        return "Type mismatch - check variable types"

    def _analyze_attribute_error(self, error: str, report: ErrorReport) -> str:
        """Analyze AttributeError details"""
        match = re.search(r"'(\w+)' object has no attribute '(\w+)'", error)
        if match:
            obj_type, attr = match.groups()
            return f"'{obj_type}' doesn't have '{attr}'. Check spelling or object type."

        return "Object doesn't have this attribute. Check object type and attribute name."

    def _analyze_name_error(self, error: str, report: ErrorReport) -> str:
        """Analyze NameError details"""
        match = re.search(r"name ['\"]?(\w+)['\"]? is not defined", error)
        if match:
            name = match.group(1)
            return f"'{name}' is not defined. Check spelling, add import, or define it first."

        return "Variable not defined. Check spelling or add import statement."

    def _format_location(self, report: ErrorReport) -> str:
        """Format file location string"""
        if report.file_path and report.line_number:
            return f" at {report.file_path}:{report.line_number}"
        elif report.file_path:
            return f" in {report.file_path}"
        elif report.line_number:
            return f" at line {report.line_number}"
        return ""

    def _generate_diagnostics(self, report: ErrorReport) -> str:
        """Generate detailed diagnostics for the error"""
        lines = []

        lines.append(f"Error Type: {report.subcategory or 'Unknown'}")
        lines.append(f"Category: {report.category.value}")

        if report.file_path:
            lines.append(f"File: {report.file_path}")
        if report.line_number:
            lines.append(f"Line: {report.line_number}")

        lines.append(f"\nRaw Error:\n{report.raw_message[:500]}")

        return "\n".join(lines)


# Export
__all__ = ["SyntaxAgent"]
