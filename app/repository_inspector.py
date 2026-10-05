from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from .models import EvaluationStatus, RepositoryInspectionResult


class RepositoryInspector:
    """
    Performs static inspection of a submitted Python project.

    This stage does not execute student code.
    It only collects structural and static evidence for later stages.
    """

    DEFAULT_LARGE_FILE_MB = 25
    DEFAULT_RESOURCE_WARNING_MB = 100

    DEPENDENCY_FILENAMES = {
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "pipfile",
        "pipfile.lock",
        "environment.yml",
        "environment.yaml",
        "setup.py",
        "setup.cfg",
    }

    ENTRY_POINT_NAMES = {
        "main.py",
        "app.py",
        "run.py",
        "cli.py",
        "server.py",
    }

    MODEL_EXTENSIONS = {
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".h5",
        ".keras",
        ".pkl",
        ".pickle",
        ".joblib",
        ".safetensors",
    }

    DATASET_EXTENSIONS = {
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".parquet",
        ".feather",
        ".xlsx",
        ".xls",
        ".npy",
        ".npz",
    }

    IGNORED_DIRECTORIES = {
        ".git",
        ".github",
        ".idea",
        ".pytest_cache",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        "node_modules",
    }

    SECRET_PATTERNS = [
        re.compile(
            r"(api[_-]?key|secret[_-]?key|access[_-]?token)"
            r"\s*[:=]\s*['\"][^'\"]+['\"]",
            re.IGNORECASE,
        ),
        re.compile(
            r"(sk-[A-Za-z0-9]{20,})",
            re.IGNORECASE,
        ),
        re.compile(
            r"(AIza[A-Za-z0-9_\-]{20,})",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        large_file_mb: int = DEFAULT_LARGE_FILE_MB,
        resource_warning_mb: int = DEFAULT_RESOURCE_WARNING_MB,
    ):
        self.large_file_bytes = large_file_mb * 1024 * 1024
        self.resource_warning_bytes = (
            resource_warning_mb * 1024 * 1024
        )

    def inspect(
        self,
        project_root: Path,
    ) -> RepositoryInspectionResult:
        """
        Inspect a project directory and return static findings.
        """

        project_root = Path(project_root)

        result = RepositoryInspectionResult(
            root_path=project_root,
            status=EvaluationStatus.RUNNING,
        )

        if not project_root.exists():
            result.status = EvaluationStatus.FAILED
            result.inspection_warnings.append(
                f"Project directory does not exist: {project_root}"
            )
            return result

        if not project_root.is_dir():
            result.status = EvaluationStatus.FAILED
            result.inspection_warnings.append(
                f"Project path is not a directory: {project_root}"
            )
            return result

        try:
            all_entries = self._walk_project(project_root)

            for path in all_entries:
                try:
                    if path.is_dir():
                        result.total_directories += 1
                        continue

                    if not path.is_file():
                        continue

                    result.total_files += 1

                    relative_path = str(
                        path.relative_to(project_root)
                    )

                    if path.suffix.lower() == ".py":
                        result.python_files.append(
                            relative_path
                        )
                    else:
                        result.other_files.append(
                            relative_path
                        )

                    self._inspect_dependency_file(
                        path,
                        relative_path,
                        result,
                    )

                    self._inspect_entry_point(
                        path,
                        relative_path,
                        result,
                    )

                    self._inspect_large_file(
                        path,
                        relative_path,
                        result,
                    )

                    self._inspect_model_file(
                        path,
                        relative_path,
                        result,
                    )

                    self._inspect_dataset_file(
                        path,
                        relative_path,
                        result,
                    )

                    self._inspect_configuration_file(
                        path,
                        relative_path,
                        result,
                    )

                    if path.suffix.lower() == ".py":
                        self._inspect_python_file(
                            path,
                            relative_path,
                            result,
                        )

                except (OSError, ValueError) as exc:
                    result.inspection_warnings.append(
                        f"Could not inspect "
                        f"{path}: {exc}"
                    )

            self._generate_resource_warnings(result)

            self._generate_general_warnings(result)

            result.status = EvaluationStatus.SUCCESS

        except Exception as exc:
            result.status = EvaluationStatus.FAILED
            result.inspection_warnings.append(
                f"Repository inspection failed: {exc}"
            )

        return result

    # ==================================================================
    # Directory traversal
    # ==================================================================

    def _walk_project(
        self,
        project_root: Path,
    ) -> list[Path]:
        """
        Walk the project while skipping irrelevant directories.
        """

        entries: list[Path] = []

        for current_root, directories, files in os.walk(
            project_root
        ):
            directories[:] = [
                directory
                for directory in directories
                if directory not in self.IGNORED_DIRECTORIES
            ]

            current_path = Path(current_root)

            for directory in directories:
                entries.append(
                    current_path / directory
                )

            for filename in files:
                entries.append(
                    current_path / filename
                )

        return entries

    # ==================================================================
    # File category inspection
    # ==================================================================

    def _inspect_dependency_file(
        self,
        path: Path,
        relative_path: str,
        result: RepositoryInspectionResult,
    ) -> None:
        if path.name.lower() in {
            name.lower()
            for name in self.DEPENDENCY_FILENAMES
        }:
            result.dependency_files.append(
                relative_path
            )

    def _inspect_entry_point(
        self,
        path: Path,
        relative_path: str,
        result: RepositoryInspectionResult,
    ) -> None:
        if path.name.lower() in {
            name.lower()
            for name in self.ENTRY_POINT_NAMES
        }:
            result.entry_point_candidates.append(
                relative_path
            )

    def _inspect_large_file(
        self,
        path: Path,
        relative_path: str,
        result: RepositoryInspectionResult,
    ) -> None:
        try:
            size_bytes = path.stat().st_size
        except OSError:
            return

        if size_bytes >= self.large_file_bytes:
            result.large_files.append(
                {
                    "path": relative_path,
                    "size_bytes": size_bytes,
                    "size_mb": round(
                        size_bytes / (1024 * 1024),
                        2,
                    ),
                }
            )

    def _inspect_model_file(
        self,
        path: Path,
        relative_path: str,
        result: RepositoryInspectionResult,
    ) -> None:
        if path.suffix.lower() in self.MODEL_EXTENSIONS:
            result.model_files.append(
                relative_path
            )

    def _inspect_dataset_file(
        self,
        path: Path,
        relative_path: str,
        result: RepositoryInspectionResult,
    ) -> None:
        if path.suffix.lower() in self.DATASET_EXTENSIONS:
            result.dataset_files.append(
                relative_path
            )

    def _inspect_configuration_file(
        self,
        path: Path,
        relative_path: str,
        result: RepositoryInspectionResult,
    ) -> None:
        filename = path.name.lower()

        if filename in {
            ".env",
            ".env.local",
            ".env.production",
            ".env.development",
        }:
            result.external_configuration.append(
                relative_path
            )

    # ==================================================================
    # Python inspection
    # ==================================================================

    def _inspect_python_file(
        self,
        path: Path,
        relative_path: str,
        result: RepositoryInspectionResult,
    ) -> None:
        try:
            source = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            result.inspection_warnings.append(
                f"Could not read Python file "
                f"{relative_path}: {exc}"
            )
            return

        self._detect_imports(
            source=source,
            result=result,
        )

        self._detect_potential_secrets(
            source=source,
            relative_path=relative_path,
            result=result,
        )

    def _detect_imports(
        self,
        source: str,
        result: RepositoryInspectionResult,
    ) -> None:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            result.inspection_warnings.append(
                f"Python syntax error detected during "
                f"static inspection: {exc}"
            )
            return

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):

                for alias in node.names:
                    root_name = alias.name.split(".")[0]

                    if root_name:
                        result.detected_imports.append(
                            root_name
                        )

            elif isinstance(node, ast.ImportFrom):

                if node.module:
                    root_name = node.module.split(".")[0]

                    if root_name:
                        result.detected_imports.append(
                            root_name
                        )

        result.detected_imports = (
            self._unique_preserve_order(
                result.detected_imports
            )
        )

    def _detect_potential_secrets(
        self,
        source: str,
        relative_path: str,
        result: RepositoryInspectionResult,
    ) -> None:
        for pattern in self.SECRET_PATTERNS:

            if pattern.search(source):
                result.potential_secrets.append(
                    relative_path
                )
                break

    # ==================================================================
    # Warnings
    # ==================================================================

    def _generate_resource_warnings(
        self,
        result: RepositoryInspectionResult,
    ) -> None:
        for file_info in result.large_files:

            size_bytes = file_info["size_bytes"]

            if size_bytes >= self.resource_warning_bytes:

                result.resource_warnings.append(
                    "Large resource detected: "
                    f"{file_info['path']} "
                    f"({file_info['size_mb']} MB)"
                )

        if result.model_files:
            result.resource_warnings.append(
                "Model artifact files detected. "
                "Runtime resource requirements should be "
                "checked before execution."
            )

        if result.dataset_files:
            result.resource_warnings.append(
                "Dataset files detected. "
                "Dataset size and runtime memory usage "
                "should be checked before execution."
            )

    def _generate_general_warnings(
        self,
        result: RepositoryInspectionResult,
    ) -> None:
        if not result.python_files:
            result.inspection_warnings.append(
                "No Python source files were detected."
            )

        if not result.entry_point_candidates:
            result.inspection_warnings.append(
                "No conventional Python entry-point file "
                "was detected."
            )

        if not result.dependency_files:
            result.inspection_warnings.append(
                "No standard dependency file was detected."
            )

        if result.potential_secrets:
            result.inspection_warnings.append(
                "Potential secrets or API credentials may "
                "be present in source files."
            )

        if result.external_configuration:
            result.inspection_warnings.append(
                "Environment/configuration files were detected. "
                "Required configuration must be checked before "
                "execution."
            )

    # ==================================================================
    # Utility
    # ==================================================================

    @staticmethod
    def _unique_preserve_order(
        values: list[str],
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


def inspect_repository(
    project_root: str | Path,
) -> RepositoryInspectionResult:
    """
    Convenience function for the main evaluation pipeline.
    """

    inspector = RepositoryInspector()

    return inspector.inspect(
        Path(project_root)
    )