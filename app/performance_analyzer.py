from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from .models import (
    CodeQualityResult,
    EvaluationStatus,
    ExecutionResult,
    PerformanceResult,
    RepositoryInspectionResult,
)


class PerformanceAnalyzer:
    """
    Evaluates project performance and pipeline efficiency.

    Runtime measurements come from ExecutionResult.
    Static efficiency observations come from repository inspection.
    """

    def __init__(
        self,
        time_warning_seconds: float = 10.0,
        memory_warning_mb: float = 500.0,
    ):
        self.time_warning_seconds = time_warning_seconds
        self.memory_warning_mb = memory_warning_mb

    def analyze(
        self,
        project_root: Path,
        execution: Optional[ExecutionResult],
        repository: Optional[RepositoryInspectionResult],
    ) -> PerformanceResult:

        result = PerformanceResult(
            status=EvaluationStatus.RUNNING
        )

        if execution is None:
            result.status = EvaluationStatus.PARTIAL
            result.resource_warnings.append(
                "Runtime execution data was not available."
            )
            return result

        result.startup_time_seconds = (
            execution.execution_time_seconds
        )

        result.total_execution_time_seconds = (
            execution.execution_time_seconds
        )

        result.peak_memory_mb = execution.peak_memory_mb

        result.measured = (
            execution.execution_time_seconds is not None
        )

        if execution.execution_time_seconds is not None:

            if execution.execution_time_seconds > self.time_warning_seconds:
                result.efficiency_findings.append(
                    f"Execution time is relatively high: "
                    f"{execution.execution_time_seconds:.2f} seconds."
                )

        if execution.peak_memory_mb is not None:

            if execution.peak_memory_mb > self.memory_warning_mb:
                result.resource_warnings.append(
                    f"Peak memory usage is high: "
                    f"{execution.peak_memory_mb:.2f} MB."
                )

        if repository:

            result.dependency_count = len(
                repository.detected_imports
            )

            if repository.model_files:
                result.efficiency_findings.append(
                    "Model artifact files were detected. "
                    "Model loading and inference cost should be reviewed."
                )

            if repository.dataset_files:
                result.efficiency_findings.append(
                    "Dataset files were detected. "
                    "Dataset loading and memory usage should be reviewed."
                )

            if repository.large_files:
                result.resource_warnings.append(
                    "Large files are present in the repository."
                )

            result.efficiency_findings.extend(
                self._detect_static_inefficiencies(
                    project_root
                )
            )

        result.status = (
            EvaluationStatus.SUCCESS
            if result.measured
            else EvaluationStatus.PARTIAL
        )

        return result

    def _detect_static_inefficiencies(
        self,
        project_root: Path,
    ) -> list[str]:

        findings: list[str] = []

        for path in project_root.rglob("*.py"):

            if any(
                part in {
                    ".git",
                    ".venv",
                    "venv",
                    "__pycache__",
                }
                for part in path.parts
            ):
                continue

            try:
                source = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                tree = ast.parse(source)
            except (OSError, SyntaxError):
                continue

            for node in ast.walk(tree):

                # Detect expensive-looking model loading inside loops.
                if isinstance(node, ast.For):

                    loop_source = ast.get_source_segment(
                        source,
                        node,
                    ) or ""

                    lowered = loop_source.lower()

                    if any(
                        keyword in lowered
                        for keyword in (
                            "from_pretrained",
                            "load_model",
                            "torch.load",
                            "joblib.load",
                            "keras.models.load",
                        )
                    ):
                        findings.append(
                            f"Potential repeated model/resource loading "
                            f"inside a loop in {path.name}."
                        )

                # Detect nested loops as a simple complexity warning.
                if isinstance(node, ast.For):

                    nested = any(
                        isinstance(child, ast.For)
                        for child in ast.walk(node)
                        if child is not node
                    )

                    if nested:
                        findings.append(
                            f"Nested loop detected in {path.name}; "
                            "review complexity for large inputs."
                        )

        return self._unique(findings)

    @staticmethod
    def _unique(values: list[str]) -> list[str]:

        seen = set()
        result = []

        for value in values:

            if value not in seen:
                seen.add(value)
                result.append(value)

        return result


def analyze_performance(
    project_root: str | Path,
    execution: Optional[ExecutionResult],
    repository: Optional[RepositoryInspectionResult],
) -> PerformanceResult:

    analyzer = PerformanceAnalyzer()

    return analyzer.analyze(
        project_root=Path(project_root),
        execution=execution,
        repository=repository,
    )