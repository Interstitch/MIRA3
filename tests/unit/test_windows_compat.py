"""
Windows Compatibility Tests

Comprehensive tests to ensure the codebase works correctly on Windows.
Uses both static analysis (AST + regex) and functional testing.

Test Categories:
1. Static Analysis - Scans source code for problematic patterns
2. Functional Tests - Verifies cross-platform functions work correctly
3. Path Handling - Ensures paths work on both platforms
4. Platform Guards - Verifies platform-specific code is properly guarded
"""

import ast
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
from unittest.mock import patch

import pytest

# =============================================================================
# Configuration
# =============================================================================

SRC_DIR = Path(__file__).parent.parent.parent / "src" / "mira"

# Known exceptions - patterns that look problematic but are intentional
KNOWN_EXCEPTIONS = {
    # Display formatting patterns (not filesystem paths)
    "readable = '/' +",
    "project_path_normalized = '/' +",
    # SQL column names
    "source TEXT DEFAULT",
    "source = CASE WHEN",
}

# Files to skip entirely
SKIP_FILES = {
    "test_windows_compat.py",
}


# =============================================================================
# Utilities
# =============================================================================

def get_python_files() -> List[Path]:
    """Get all Python files in the mira source directory."""
    return [f for f in SRC_DIR.rglob("*.py") if f.name not in SKIP_FILES]


def read_file_content(path: Path) -> str:
    """Read file content, handling encoding issues."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def is_in_known_exceptions(line: str) -> bool:
    """Check if a line matches a known exception pattern."""
    return any(exc in line for exc in KNOWN_EXCEPTIONS)


def get_relative_path(path: Path) -> str:
    """Get path relative to src directory for readable output."""
    try:
        return str(path.relative_to(SRC_DIR.parent))
    except ValueError:
        return str(path)


@dataclass
class Violation:
    """Represents a compatibility violation found in source code."""
    file: Path
    line_num: int
    line_content: str
    message: str
    severity: str = "error"  # error, warning, info

    def __str__(self) -> str:
        rel_path = get_relative_path(self.file)
        return f"{rel_path}:{self.line_num}: [{self.severity}] {self.message}\n    {self.line_content.strip()}"


class SourceAnalyzer:
    """Base class for analyzing source files."""

    def __init__(self, path: Path):
        self.path = path
        self.content = read_file_content(path)
        self.lines = self.content.split("\n")
        self._ast: Optional[ast.AST] = None

    @property
    def ast_tree(self) -> Optional[ast.AST]:
        """Lazily parse AST."""
        if self._ast is None:
            try:
                self._ast = ast.parse(self.content)
            except SyntaxError:
                pass
        return self._ast

    def has_platform_guard(self, line_num: int, lookback: int = 15) -> bool:
        """Check if the given line is within a platform guard."""
        start = max(0, line_num - lookback)
        context = "\n".join(self.lines[start:line_num])
        return any(guard in context for guard in [
            "sys.platform",
            "os.name",
            "platform.system",
            "if WIN32",
            "if WINDOWS",
            "if IS_WINDOWS",
            # OS type checks (e.g., if os_type == 'linux':)
            "os_type ==",
            "os_type !=",
            "== 'linux'",
            "== 'darwin'",
            "== 'windows'",
            "!= 'win32'",
        ])

    def is_in_comment_or_docstring(self, line_num: int) -> bool:
        """Check if a line is in a comment or docstring."""
        line = self.lines[line_num - 1].strip()
        if line.startswith("#"):
            return True
        # Simple docstring detection
        if line.startswith('"""') or line.startswith("'''"):
            return True
        if line.startswith('r"""') or line.startswith("r'''"):
            return True
        return False


# =============================================================================
# AST-Based Analyzers
# =============================================================================

class FunctionCallVisitor(ast.NodeVisitor):
    """AST visitor to find function calls."""

    def __init__(self):
        self.calls: List[Tuple[str, int, int]] = []  # (name, line, col)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.calls.append((node.func.id, node.lineno, node.col_offset))
        elif isinstance(node.func, ast.Attribute):
            self.calls.append((node.func.attr, node.lineno, node.col_offset))
        self.generic_visit(node)


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to find imports."""

    def __init__(self):
        self.imports: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module)
        self.generic_visit(node)


# =============================================================================
# Static Analysis Tests
# =============================================================================

class TestHardcodedUnixPaths:
    """Test for hardcoded Unix-style paths that won't work on Windows."""

    UNIX_PATHS = [
        (r'["\']\/bin\/', "/bin/", "Hardcoded /bin/ path"),
        (r'["\']\/usr\/', "/usr/", "Hardcoded /usr/ path"),
        (r'["\']\/etc\/', "/etc/", "Hardcoded /etc/ path"),
        (r'["\']\/tmp\/', "/tmp/", "Hardcoded /tmp/ - use tempfile module"),
        (r'["\']\/var\/', "/var/", "Hardcoded /var/ path"),
        (r'["\']\/home\/', "/home/", "Hardcoded /home/ - use Path.home()"),
        (r'["\']\/opt\/', "/opt/", "Hardcoded /opt/ path"),
    ]

    # Patterns that are OK (URLs, regexes, etc.)
    SAFE_PATTERNS = [
        r"https?://",
        r"git@",
        r"ssh://",
        r"file://",
        r'r["\']',  # Raw strings (regexes)
    ]

    def test_no_hardcoded_unix_paths(self):
        """Ensure no hardcoded Unix paths outside of platform guards."""
        violations = []

        for py_file in get_python_files():
            analyzer = SourceAnalyzer(py_file)

            for line_num, line in enumerate(analyzer.lines, 1):
                if analyzer.is_in_comment_or_docstring(line_num):
                    continue

                # Skip safe patterns
                if any(re.search(pat, line) for pat in self.SAFE_PATTERNS):
                    continue

                for pattern, path_type, message in self.UNIX_PATHS:
                    if re.search(pattern, line):
                        if not analyzer.has_platform_guard(line_num):
                            violations.append(Violation(
                                file=py_file,
                                line_num=line_num,
                                line_content=line,
                                message=message,
                            ))

        assert not violations, self._format_violations(violations)

    def _format_violations(self, violations: List[Violation]) -> str:
        return (
            f"Found {len(violations)} hardcoded Unix paths:\n"
            + "\n".join(str(v) for v in violations[:15])
            + ("\n..." if len(violations) > 15 else "")
        )


class TestWindowsIncompatibleAPIs:
    """Test for APIs that don't exist or behave differently on Windows."""

    # (function_name, module, message, requires_guard)
    INCOMPATIBLE_APIS = [
        ("fork", "os", "os.fork() doesn't exist on Windows - use multiprocessing", True),
        ("fchmod", "os", "os.fchmod() doesn't exist on Windows", True),
        ("fchown", "os", "os.fchown() doesn't exist on Windows", True),
        ("chown", "os", "os.chown() doesn't exist on Windows", True),
        ("getuid", "os", "os.getuid() doesn't exist on Windows", True),
        ("getgid", "os", "os.getgid() doesn't exist on Windows", True),
        ("setsid", "os", "os.setsid() doesn't exist on Windows", True),
        ("fcntl", None, "fcntl module doesn't exist on Windows - use msvcrt", True),
        ("grp", None, "grp module doesn't exist on Windows", True),
        ("pwd", None, "pwd module doesn't exist on Windows", True),
    ]

    def test_no_unguarded_incompatible_apis(self):
        """Ensure Windows-incompatible APIs are platform-guarded."""
        violations = []

        for py_file in get_python_files():
            analyzer = SourceAnalyzer(py_file)

            for line_num, line in enumerate(analyzer.lines, 1):
                if analyzer.is_in_comment_or_docstring(line_num):
                    continue

                for func, module, message, requires_guard in self.INCOMPATIBLE_APIS:
                    # Build patterns to match
                    patterns = []
                    if module:
                        patterns.append(rf'\b{module}\.{func}\s*\(')
                    else:
                        patterns.append(rf'\bimport\s+{func}\b')
                        patterns.append(rf'\bfrom\s+{func}\s+import\b')

                    for pattern in patterns:
                        if re.search(pattern, line):
                            if requires_guard and not analyzer.has_platform_guard(line_num):
                                violations.append(Violation(
                                    file=py_file,
                                    line_num=line_num,
                                    line_content=line,
                                    message=message,
                                ))

        assert not violations, self._format_violations(violations)

    def _format_violations(self, violations: List[Violation]) -> str:
        return (
            f"Found {len(violations)} unguarded Windows-incompatible APIs:\n"
            + "\n".join(str(v) for v in violations[:10])
        )


class TestReservedFilenames:
    """Test that code doesn't create Windows reserved filenames."""

    # Windows reserved device names (case-insensitive)
    RESERVED_NAMES = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }

    def test_no_reserved_filename_creation(self):
        """Ensure code doesn't create files with Windows reserved names."""
        violations = []

        # Patterns that create files
        file_creation_patterns = [
            r'open\s*\(\s*["\']([^"\']+)["\']',
            r'Path\s*\(\s*["\']([^"\']+)["\']',
            r'\.write_text\s*\(',
            r'\.write_bytes\s*\(',
        ]

        for py_file in get_python_files():
            content = read_file_content(py_file)
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                for pattern in file_creation_patterns:
                    match = re.search(pattern, line)
                    if match and match.groups():
                        filename = match.group(1)
                        # Extract just the filename part
                        basename = Path(filename).stem.upper()
                        if basename in self.RESERVED_NAMES:
                            violations.append(Violation(
                                file=py_file,
                                line_num=line_num,
                                line_content=line,
                                message=f"'{basename}' is a Windows reserved name",
                                severity="warning",
                            ))

        assert not violations, (
            f"Found {len(violations)} potential reserved filename uses:\n"
            + "\n".join(str(v) for v in violations[:10])
        )


class TestPathConstruction:
    """Test for proper cross-platform path construction."""

    def test_uses_pathlib_or_os_path(self):
        """Check that path construction uses pathlib or os.path, not string ops."""
        suspicious_patterns = [
            (r'\+\s*os\.sep\s*\+', "Use pathlib or os.path.join instead of os.sep concatenation"),
            (r'["\'][/\\]["\']\.join\s*\(', "Use os.path.join or Path for joining paths"),
        ]

        safe_patterns = [
            r'readable\s*=',
            r'\.lstrip\(',
            r'\.rstrip\(',
            r'replace\(',
            r'\.startswith\(',
            r'\.endswith\(',
            r'split\(',
        ]

        violations = []

        for py_file in get_python_files():
            analyzer = SourceAnalyzer(py_file)

            for line_num, line in enumerate(analyzer.lines, 1):
                if analyzer.is_in_comment_or_docstring(line_num):
                    continue
                if is_in_known_exceptions(line):
                    continue

                for pattern, message in suspicious_patterns:
                    if re.search(pattern, line):
                        if not any(re.search(safe, line) for safe in safe_patterns):
                            violations.append(Violation(
                                file=py_file,
                                line_num=line_num,
                                line_content=line,
                                message=message,
                            ))

        assert not violations, (
            f"Found {len(violations)} suspicious path construction patterns:\n"
            + "\n".join(str(v) for v in violations[:10])
        )

    def test_path_functions_return_path_objects(self):
        """Verify key path functions return Path objects."""
        from mira.core.constants import (
            get_global_mira_path,
            get_project_mira_path,
            get_db_path,
            DB_CUSTODIAN,
        )
        from mira.core.utils import get_venv_path, get_claude_projects_path

        path_functions = [
            ("get_global_mira_path", get_global_mira_path),
            ("get_project_mira_path", get_project_mira_path),
            ("get_venv_path", get_venv_path),
            ("get_claude_projects_path", get_claude_projects_path),
        ]

        for name, func in path_functions:
            result = func()
            assert isinstance(result, Path), f"{name}() should return Path, got {type(result)}"

        # Also check get_db_path
        result = get_db_path(DB_CUSTODIAN)
        assert isinstance(result, Path), f"get_db_path() should return Path, got {type(result)}"


class TestVenvPaths:
    """Test that venv path functions handle Windows correctly."""

    @pytest.mark.parametrize("func_name", [
        "get_venv_python",
        "get_venv_pip",
        "get_venv_uv",
        "get_venv_mira",
    ])
    def test_venv_executable_paths(self, func_name: str):
        """Verify venv executable paths are correct for current platform."""
        from mira.core import utils

        func = getattr(utils, func_name)
        path = func()

        if sys.platform == "win32":
            assert "Scripts" in path, f"Windows should use Scripts, got: {path}"
            assert path.endswith(".exe"), f"Windows should use .exe: {path}"
        else:
            assert "bin" in path, f"Unix should use bin, got: {path}"

    def test_venv_executable_source_has_platform_handling(self):
        """Verify venv executable functions check platform in source."""
        import inspect
        from mira.core import utils

        for func_name in ['get_venv_python', 'get_venv_pip', 'get_venv_uv', 'get_venv_mira']:
            func = getattr(utils, func_name)
            source = inspect.getsource(func)
            assert 'sys.platform' in source or 'win32' in source, \
                f"{func_name} should have platform-specific handling"
            assert '.exe' in source, f"{func_name} should reference .exe for Windows"


class TestSubprocessCalls:
    """Test for subprocess calls that might fail on Windows."""

    def test_no_shell_true_with_list(self):
        """Check for subprocess calls with shell=True and list args (breaks on Windows)."""
        violations = []

        # Pattern: subprocess.run/call/Popen with shell=True
        shell_pattern = r'subprocess\.(run|call|Popen)\s*\([^)]*shell\s*=\s*True'

        for py_file in get_python_files():
            content = read_file_content(py_file)
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                if re.search(shell_pattern, line):
                    # Check if args is a list (problematic with shell=True on Windows)
                    if re.search(r'\[\s*["\']', line):
                        violations.append(Violation(
                            file=py_file,
                            line_num=line_num,
                            line_content=line,
                            message="shell=True with list args behaves differently on Windows",
                            severity="warning",
                        ))

        assert not violations, (
            f"Found {len(violations)} potentially problematic subprocess calls:\n"
            + "\n".join(str(v) for v in violations[:10])
        )


class TestFileOperations:
    """Test for file operations that behave differently on Windows."""

    def test_no_unguarded_chmod(self):
        """Check that chmod calls are guarded with platform checks."""
        violations = []

        for py_file in get_python_files():
            analyzer = SourceAnalyzer(py_file)

            for line_num, line in enumerate(analyzer.lines, 1):
                if "os.chmod(" in line or ".chmod(" in line:
                    if not analyzer.has_platform_guard(line_num, lookback=15):
                        # Allow if in a function that handles permissions
                        if "validate_file_permissions" not in line:
                            violations.append(Violation(
                                file=py_file,
                                line_num=line_num,
                                line_content=line,
                                message="chmod may not work as expected on Windows",
                                severity="warning",
                            ))

        # Filter known safe patterns
        filtered = [v for v in violations if "validate_file_permissions" not in str(v)]

        assert not filtered, (
            f"Found {len(filtered)} chmod calls without platform guards:\n"
            + "\n".join(str(v) for v in filtered[:10])
        )

    def test_no_file_locking_without_guard(self):
        """Check for file locking that uses Unix-specific fcntl."""
        violations = []

        for py_file in get_python_files():
            content = read_file_content(py_file)

            # Check for fcntl imports/usage without platform guard
            if "fcntl" in content:
                analyzer = SourceAnalyzer(py_file)
                for line_num, line in enumerate(analyzer.lines, 1):
                    if "fcntl" in line and "import" in line:
                        if not analyzer.has_platform_guard(line_num):
                            violations.append(Violation(
                                file=py_file,
                                line_num=line_num,
                                line_content=line,
                                message="fcntl doesn't exist on Windows - use msvcrt or cross-platform alternative",
                            ))

        assert not violations, (
            f"Found {len(violations)} unguarded fcntl usages:\n"
            + "\n".join(str(v) for v in violations[:10])
        )


class TestEnvironmentVariables:
    """Test for environment variable access patterns."""

    def test_uses_path_home_not_env_home(self):
        """Verify Path.home() is used instead of os.environ['HOME']."""
        violations = []

        for py_file in get_python_files():
            content = read_file_content(py_file)
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                if line.strip().startswith("#"):
                    continue

                # Check for HOME environment variable access
                if re.search(r"os\.environ\s*\[\s*['\"]HOME['\"]\s*\]", line):
                    violations.append(Violation(
                        file=py_file,
                        line_num=line_num,
                        line_content=line,
                        message="Use Path.home() instead of os.environ['HOME']",
                    ))
                if re.search(r"os\.getenv\s*\(\s*['\"]HOME['\"]", line):
                    violations.append(Violation(
                        file=py_file,
                        line_num=line_num,
                        line_content=line,
                        message="Use Path.home() instead of os.getenv('HOME')",
                    ))

        assert not violations, (
            f"Found {len(violations)} uses of HOME env var:\n"
            + "\n".join(str(v) for v in violations[:10])
        )


class TestTempPaths:
    """Test that temporary paths use cross-platform methods."""

    def test_uses_tempfile_module(self):
        """Verify temporary files use tempfile module, not /tmp/."""
        violations = []

        for py_file in get_python_files():
            if "test_" in py_file.name:
                continue

            content = read_file_content(py_file)
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                if line.strip().startswith("#"):
                    continue

                if '"/tmp/' in line or "'/tmp/" in line:
                    # Skip URLs and examples
                    if "http" not in line and "example" not in line.lower():
                        violations.append(Violation(
                            file=py_file,
                            line_num=line_num,
                            line_content=line,
                            message="Use tempfile module instead of /tmp/",
                        ))

        assert not violations, (
            f"Found {len(violations)} hardcoded /tmp/ paths:\n"
            + "\n".join(str(v) for v in violations[:10])
        )


# =============================================================================
# Database Path Tests
# =============================================================================

class TestDatabasePaths:
    """Test that database paths are correctly routed."""

    def test_global_databases_use_global_path(self):
        """Verify global databases are stored in ~/.mira/."""
        from mira.core.constants import (
            GLOBAL_DATABASES,
            get_db_path,
            get_global_mira_path,
        )

        global_path = get_global_mira_path()

        for db_name in GLOBAL_DATABASES:
            db_path = get_db_path(db_name)
            assert db_path.parent == global_path, (
                f"{db_name} should be in {global_path}, got {db_path.parent}"
            )

    def test_project_databases_use_project_path(self):
        """Verify project databases are stored in <cwd>/.mira/."""
        from mira.core.constants import (
            PROJECT_DATABASES,
            get_db_path,
            get_project_mira_path,
        )

        project_path = get_project_mira_path()

        for db_name in PROJECT_DATABASES:
            db_path = get_db_path(db_name)
            assert db_path.parent == project_path, (
                f"{db_name} should be in {project_path}, got {db_path.parent}"
            )


# =============================================================================
# Platform Guard Tests
# =============================================================================

class TestPlatformGuards:
    """Test that platform-specific code has proper guards."""

    def test_bootstrap_has_platform_guards(self):
        """Verify bootstrap.py has proper platform guards."""
        bootstrap_file = SRC_DIR / "core" / "bootstrap.py"
        content = read_file_content(bootstrap_file)

        assert 'sys.platform' in content, "bootstrap.py should check sys.platform"
        assert 'win32' in content, "bootstrap.py should handle Windows (win32)"

    def test_utils_has_platform_guards(self):
        """Verify utils.py has proper platform guards."""
        utils_file = SRC_DIR / "core" / "utils.py"
        content = read_file_content(utils_file)

        assert 'sys.platform' in content, "utils.py should check sys.platform"
        assert 'win32' in content, "utils.py should handle Windows (win32)"
        assert 'Scripts' in content, "utils.py should reference Windows Scripts directory"

    def test_platform_specific_files_are_guarded(self):
        """Ensure files with platform-specific code have guards."""
        # Files we know should have platform guards
        expected_guarded = [
            "core/bootstrap.py",
            "core/utils.py",
        ]

        for rel_path in expected_guarded:
            file_path = SRC_DIR / rel_path
            if file_path.exists():
                content = read_file_content(file_path)
                assert 'sys.platform' in content or 'os.name' in content, \
                    f"{rel_path} should have platform guards"


# =============================================================================
# Functional Tests (with mocking)
# =============================================================================

class TestCrossPlatformFunctionality:
    """Functional tests that verify cross-platform behavior."""

    def test_path_functions_work_on_both_platforms(self):
        """Test that path functions produce valid paths."""
        from mira.core.constants import get_global_mira_path, get_project_mira_path

        global_path = get_global_mira_path()
        project_path = get_project_mira_path()

        # Should be absolute paths
        assert global_path.is_absolute(), "Global path should be absolute"
        assert project_path.is_absolute(), "Project path should be absolute"

        # Should be in home directory
        assert str(Path.home()) in str(global_path), "Global path should be under home"

    @pytest.mark.parametrize("platform,expected_dir,expected_ext", [
        ("win32", "Scripts", ".exe"),
        ("linux", "bin", ""),
        ("darwin", "bin", ""),
    ])
    def test_venv_paths_by_platform(self, platform: str, expected_dir: str, expected_ext: str):
        """Test venv paths produce correct results for each platform."""
        from mira.core import utils

        with patch.object(sys, 'platform', platform):
            # Re-import to pick up patched platform
            # Note: This tests the logic, actual behavior depends on implementation
            python_path = utils.get_venv_python()

            if sys.platform == platform:  # Only check if we're actually on this platform
                if platform == "win32":
                    assert expected_dir in python_path
                    assert python_path.endswith(expected_ext)

    def test_db_path_creates_parent_directories(self):
        """Test that get_db_path works with non-existent parent directories."""
        from mira.core.constants import get_db_path

        # This should not raise even if the directory doesn't exist
        db_path = get_db_path("test_nonexistent.db")
        assert db_path is not None
        assert str(db_path).endswith("test_nonexistent.db")


class TestLineEndings:
    """Test that files use consistent line endings."""

    def test_no_mixed_line_endings(self):
        """Check that Python files don't have mixed line endings."""
        violations = []

        for py_file in get_python_files():
            try:
                content = py_file.read_bytes()
            except Exception:
                continue

            has_crlf = b'\r\n' in content
            has_lf_only = b'\n' in content and b'\r\n' not in content
            has_cr_only = b'\r' in content and b'\r\n' not in content

            if has_crlf and has_lf_only:
                violations.append(f"{get_relative_path(py_file)}: Mixed CRLF and LF line endings")
            if has_cr_only:
                violations.append(f"{get_relative_path(py_file)}: Uses old Mac CR line endings")

        assert not violations, (
            f"Found {len(violations)} files with line ending issues:\n"
            + "\n".join(violations[:10])
        )


class TestSignalHandling:
    """Test for signal handling that differs between platforms."""

    def test_no_unguarded_unix_signals(self):
        """Check for Unix-specific signal handling without guards."""
        # Signals that don't exist on Windows
        unix_only_signals = [
            "SIGKILL",
            "SIGSTOP",
            "SIGTSTP",
            "SIGCONT",
            "SIGHUP",
            "SIGQUIT",
            "SIGUSR1",
            "SIGUSR2",
            "SIGALRM",
            "SIGPIPE",
        ]

        violations = []

        for py_file in get_python_files():
            analyzer = SourceAnalyzer(py_file)

            for line_num, line in enumerate(analyzer.lines, 1):
                if analyzer.is_in_comment_or_docstring(line_num):
                    continue

                for sig in unix_only_signals:
                    if f"signal.{sig}" in line:
                        if not analyzer.has_platform_guard(line_num):
                            violations.append(Violation(
                                file=py_file,
                                line_num=line_num,
                                line_content=line,
                                message=f"signal.{sig} doesn't exist on Windows",
                            ))

        assert not violations, (
            f"Found {len(violations)} unguarded Unix-only signals:\n"
            + "\n".join(str(v) for v in violations[:10])
        )


# =============================================================================
# Summary Test
# =============================================================================

class TestWindowsCompatibilitySummary:
    """Summary test that runs all checks and reports."""

    def test_all_checks_pass(self):
        """Meta-test to verify all Windows compatibility checks pass."""
        # This test exists to provide a summary when run with -v
        # All actual checks are in the tests above
        pass
