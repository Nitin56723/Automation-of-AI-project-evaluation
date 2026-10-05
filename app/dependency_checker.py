from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import venv
from pathlib import Path
from typing import Iterable

from .models import (
    DependencyCheckResult,
    EvaluationStatus,
    RepositoryInspectionResult,
)


class DependencyChecker:
    """
    Checks project dependencies and Python environment compatibility.

    The checker performs static dependency discovery first.

    Actual dependency installation is optional and can be performed in a
    temporary virtual environment when explicitly requested.
    """

    DEPENDENCY_FILES = (
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "Pipfile",
        "Pipfile.lock",
        "environment.yml",
        "environment.yaml",
    )

    # Python standard-library module names that should not be treated as
    # third-party dependencies.
    STANDARD_LIBRARY_MODULES = {
        "__future__",
        "abc",
        "argparse",
        "array",
        "ast",
        "asyncio",
        "base64",
        "calendar",
        "collections",
        "concurrent",
        "contextlib",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "errno",
        "fnmatch",
        "functools",
        "getopt",
        "glob",
        "gzip",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "http",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "mimetypes",
        "multiprocessing",
        "operator",
        "os",
        "pathlib",
        "pickle",
        "platform",
        "pprint",
        "queue",
        "random",
        "re",
        "shutil",
        "signal",
        "socket",
        "sqlite3",
        "statistics",
        "string",
        "subprocess",
        "sys",
        "tempfile",
        "textwrap",
        "threading",
        "time",
        "traceback",
        "typing",
        "unittest",
        "urllib",
        "uuid",
        "venv",
        "warnings",
        "weakref",
        "xml",
        "zipfile",
    }

    PACKAGE_IMPORT_ALIASES = {
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "bs4": "beautifulsoup4",
        "dotenv": "python-dotenv",
        "git": "GitPython",
    }

    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(
        self,
        project_root: Path,
        repository_result: RepositoryInspectionResult | None = None,
    ) -> DependencyCheckResult:
        """
        Perform static dependency inspection.

        No packages are installed by this method.
        """

        project_root = Path(project_root)

        result = DependencyCheckResult(
            status=EvaluationStatus.RUNNING,
            python_version_used=self._python_version(),
        )

        if not project_root.exists() or not project_root.is_dir():
            result.status = EvaluationStatus.FAILED
            result.warnings.append(
                f"Project directory does not exist: {project_root}"
            )
            return result

        try:
            requirements_file = self._find_dependency_file(
                project_root
            )

            if requirements_file:
                result.requirements_file = str(
                    requirements_file.relative_to(project_root)
                )

                result.declared_dependencies = (
                    self._parse_declared_dependencies(
                        requirements_file
                    )
                )

            if repository_result:
                result.detected_imports = list(
                    repository_result.detected_imports
                )

            # If repository inspection wasn't supplied, inspect Python
            # files directly.
            else:
                result.detected_imports = (
                    self._detect_python_imports(
                        project_root
                    )
                )

            external_imports = (
                self._filter_external_imports(
                    result.detected_imports
                )
            )

            normalized_declared = (
                self._normalize_dependency_names(
                    result.declared_dependencies
                )
            )

            normalized_imports = (
                self._normalize_dependency_names(
                    external_imports
                )
            )

            result.missing_dependencies = [
                package
                for package in normalized_imports
                if package not in normalized_declared
            ]

            # A project can legitimately import packages provided by an
            # environment or setup mechanism. Therefore this is a warning,
            # not automatically a failure.
            if result.missing_dependencies:
                result.warnings.append(
                    "Some imported third-party packages were not found "
                    "in the declared dependency files: "
                    + ", ".join(
                        result.missing_dependencies
                    )
                )

            result.status = EvaluationStatus.SUCCESS

        except Exception as exc:
            result.status = EvaluationStatus.FAILED
            result.warnings.append(
                f"Dependency inspection failed: {exc}"
            )

        return result

    def install_dependencies(
        self,
        project_root: Path,
        packages: Iterable[str] | None = None,
    ) -> DependencyCheckResult:
        """
        Install project dependencies into a temporary virtual environment.

        This does NOT install packages into our main .venv.

        It returns a DependencyCheckResult describing the outcome.
        """

        project_root = Path(project_root)

        result = DependencyCheckResult(
            status=EvaluationStatus.RUNNING,
            python_version_used=self._python_version(),
        )

        if not project_root.exists() or not project_root.is_dir():
            result.status = EvaluationStatus.FAILED
            result.warnings.append(
                "Project directory does not exist."
            )
            return result

        requirements_file = self._find_dependency_file(
            project_root
        )

        requested_packages = list(packages or [])

        if not requested_packages and requirements_file:
            if requirements_file.name.lower() in {
                "requirements.txt",
                "requirements-dev.txt",
            }:
                requested_packages = (
                    self._parse_declared_dependencies(
                        requirements_file
                    )
                )

        if not requested_packages:
            result.install_success = True
            result.status = EvaluationStatus.SUCCESS
            result.warnings.append(
                "No installable dependency list was found."
            )
            return result

        temp_env = project_root / ".evaluation_venv"

        try:
            self._create_virtual_environment(
                temp_env
            )

            python_executable = (
                self._venv_python_executable(
                    temp_env
                )
            )

            install_command = [
                str(python_executable),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *requested_packages,
            ]

            completed = subprocess.run(
                install_command,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )

            if completed.returncode == 0:
                result.install_success = True
                result.status = EvaluationStatus.SUCCESS
            else:
                result.install_success = False
                result.status = EvaluationStatus.FAILED

                result.install_failures.append(
                    self._truncate_output(
                        completed.stderr
                        or completed.stdout
                    )
                )

        except subprocess.TimeoutExpired:
            result.install_success = False
            result.status = EvaluationStatus.FAILED

            result.install_failures.append(
                "Dependency installation timed out."
            )

        except Exception as exc:
            result.install_success = False
            result.status = EvaluationStatus.FAILED

            result.install_failures.append(
                str(exc)
            )

        finally:
            self._remove_evaluation_environment(
                temp_env
            )

        return result

    # ------------------------------------------------------------------
    # Dependency file discovery
    # ------------------------------------------------------------------

    def _find_dependency_file(
        self,
        project_root: Path,
    ) -> Path | None:
        """
        Find the first supported dependency file.
        """

        # Prefer requirements.txt because it is the simplest and most
        # explicit format for our lightweight Python workshop projects.
        preferred = [
            "requirements.txt",
            "requirements-dev.txt",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "Pipfile",
            "environment.yml",
            "environment.yaml",
            "Pipfile.lock",
        ]

        for filename in preferred:
            candidate = project_root / filename

            if candidate.is_file():
                return candidate

        return None

    # ------------------------------------------------------------------
    # requirements.txt parsing
    # ------------------------------------------------------------------

    def _parse_declared_dependencies(
        self,
        dependency_file: Path,
    ) -> list[str]:
        """
        Parse common dependency names from requirements.txt-like files.

        This is intentionally conservative.
        """

        try:
            text = dependency_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return []

        dependencies: list[str] = []

        for raw_line in text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line.startswith("-r "):
                continue

            if line.startswith("--"):
                continue

            # Remove inline comments where possible.
            line = re.split(
                r"\s+#",
                line,
                maxsplit=1,
            )[0].strip()

            # Handle common environment markers/version specifiers.
            match = re.match(
                r"^([A-Za-z0-9_.-]+)",
                line,
            )

            if not match:
                continue

            package_name = match.group(1)

            # Ignore editable path syntax.
            if package_name in {".", ".."}:
                continue

            dependencies.append(
                package_name
            )

        return self._unique_preserve_order(
            dependencies
        )

    # ------------------------------------------------------------------
    # Import discovery
    # ------------------------------------------------------------------

    def _detect_python_imports(
        self,
        project_root: Path,
    ) -> list[str]:
        """
        Detect imports from all Python files in the project.
        """

        imports: list[str] = []

        for python_file in project_root.rglob("*.py"):

            # Skip common generated/environment directories.
            if any(
                part.lower()
                in {
                    ".git",
                    ".venv",
                    "venv",
                    "__pycache__",
                }
                for part in python_file.parts
            ):
                continue

            try:
                source = python_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                continue

            try:
                import ast

                tree = ast.parse(source)
            except SyntaxError:
                continue

            for node in ast.walk(tree):

                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(
                            alias.name.split(".")[0]
                        )

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(
                            node.module.split(".")[0]
                        )

        return self._unique_preserve_order(
            imports
        )

    def _filter_external_imports(
        self,
        imports: Iterable[str],
    ) -> list[str]:
        """
        Remove Python standard library modules.
        """

        external: list[str] = []

        for module in imports:

            root = module.split(".")[0]

            if root in self.STANDARD_LIBRARY_MODULES:
                continue

            external.append(root)

        return self._unique_preserve_order(
            external
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_dependency_names(
        self,
        packages: Iterable[str],
    ) -> set[str]:
        """
        Normalize package names for comparison.

        PEP 503-style normalization is sufficient for our comparison:
        lowercase and replace -, _, . with '-'.
        """

        normalized: set[str] = set()

        for package in packages:

            package = package.strip()

            if not package:
                continue

            package = self.PACKAGE_IMPORT_ALIASES.get(
                package,
                package,
            )

            normalized.add(
                re.sub(
                    r"[-_.]+",
                    "-",
                    package.lower(),
                )
            )

        return normalized

    # ------------------------------------------------------------------
    # Virtual environment helpers
    # ------------------------------------------------------------------

    def _create_virtual_environment(
        self,
        target: Path,
    ) -> None:
        """
        Create a temporary virtual environment.
        """

        if target.exists():
            self._remove_evaluation_environment(
                target
            )

        builder = venv.EnvBuilder(
            with_pip=True,
            clear=False,
        )

        builder.create(target)

    @staticmethod
    def _venv_python_executable(
        venv_path: Path,
    ) -> Path:
        """
        Return the Python executable inside a temporary venv.
        """

        if sys.platform == "win32":
            return venv_path / "Scripts" / "python.exe"

        return venv_path / "bin" / "python"

    @staticmethod
    def _remove_evaluation_environment(
        target: Path,
    ) -> None:
        """
        Remove temporary evaluation environment.
        """

        import shutil

        if target.exists():
            shutil.rmtree(
                target,
                ignore_errors=True,
            )

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _python_version() -> str:
        return (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )

    @staticmethod
    def _unique_preserve_order(
        values: Iterable[str],
    ) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []

        for value in values:

            normalized = value.strip()

            if not normalized:
                continue

            key = normalized.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(normalized)

        return result

    @staticmethod
    def _truncate_output(
        output: str,
        limit: int = 5000,
    ) -> str:
        output = output.strip()

        if len(output) <= limit:
            return output

        return (
            output[:limit]
            + "\n...[output truncated]..."
        )


def check_dependencies(
    project_root: str | Path,
    repository_result: RepositoryInspectionResult | None = None,
) -> DependencyCheckResult:
    """
    Convenience wrapper for static dependency inspection.
    """

    checker = DependencyChecker()

    return checker.inspect(
        Path(project_root),
        repository_result=repository_result,
    )