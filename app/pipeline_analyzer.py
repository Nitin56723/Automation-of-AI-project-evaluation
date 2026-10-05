from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

from .models import (
    EvaluationStatus,
    PipelineAnalysisResult,
    ProjectSpecification,
)


class PipelineAnalyzer:
    """
    Performs static analysis of a Python AI project's pipeline.

    This module does not execute student code.
    It looks for structural evidence of:

        input
          ↓
        preprocessing
          ↓
        model / inference
          ↓
        postprocessing
          ↓
        output
    """

    INPUT_KEYWORDS = {
        "input",
        "inputs",
        "prompt",
        "query",
        "text",
        "image",
        "audio",
        "file",
        "data",
        "request",
        "request_data",
        "stdin",
    }

    PREPROCESS_KEYWORDS = {
        "preprocess",
        "pre_processing",
        "preprocessing",
        "clean",
        "clean_text",
        "normalize",
        "normalise",
        "tokenize",
        "tokenise",
        "transform",
        "vectorize",
        "vectorise",
        "encode",
        "resize",
        "scale",
        "scaler",
        "feature",
        "features",
        "embedding",
    }

    MODEL_KEYWORDS = {
        "model",
        "predict",
        "prediction",
        "predictor",
        "inference",
        "infer",
        "classifier",
        "classify",
        "pipeline",
        "llm",
        "transformer",
        "regressor",
        "fit",
        "train",
        "load_model",
        "from_pretrained",
        "generate",
    }

    OUTPUT_KEYWORDS = {
        "output",
        "result",
        "results",
        "response",
        "return",
        "display",
        "print",
        "save",
        "write",
        "jsonify",
        "render",
        "prediction",
    }

    PIPELINE_FILE_KEYWORDS = {
        "main",
        "app",
        "run",
        "pipeline",
        "model",
        "predict",
        "inference",
        "preprocess",
        "preprocessing",
        "utils",
        "service",
    }

    COMMON_ENTRY_POINTS = {
        "main.py",
        "app.py",
        "run.py",
        "cli.py",
        "server.py",
    }

    def analyze(
        self,
        project_root: Path,
        specification: ProjectSpecification | None = None,
    ) -> PipelineAnalysisResult:
        """
        Analyze the Python source files in a project.

        Parameters
        ----------
        project_root:
            Root directory of the submitted project.

        specification:
            Structured project information obtained from the
            student form and README parser.
        """

        project_root = Path(project_root)

        result = PipelineAnalysisResult(
            status=EvaluationStatus.RUNNING
        )

        if not project_root.exists():
            result.status = EvaluationStatus.FAILED
            result.warnings.append(
                f"Project directory does not exist: {project_root}"
            )
            return result

        if not project_root.is_dir():
            result.status = EvaluationStatus.FAILED
            result.warnings.append(
                f"Project path is not a directory: {project_root}"
            )
            return result

        try:
            if specification:
                result.claimed_pipeline = (
                    self._build_claimed_pipeline(
                        specification
                    )
                )

                result.entry_point = (
                    specification.entry_point
                )

            python_files = self._find_python_files(
                project_root
            )

            if not python_files:
                result.status = EvaluationStatus.FAILED
                result.warnings.append(
                    "No Python source files were found."
                )
                return result

            evidence = self._collect_pipeline_evidence(
                project_root,
                python_files,
            )

            result.detected_stages = (
                self._detected_stage_names(
                    evidence
                )
            )

            result.pipeline_files = (
                evidence["pipeline_files"]
            )

            result.input_stage_found = (
                evidence["input"]
            )

            result.preprocessing_stage_found = (
                evidence["preprocessing"]
            )

            result.model_stage_found = (
                evidence["model"]
            )

            result.output_stage_found = (
                evidence["output"]
            )

            if result.entry_point is None:
                result.entry_point = (
                    self._find_entry_point(
                        python_files,
                        project_root,
                    )
                )

            self._check_pipeline_consistency(
                result,
                specification,
            )

            result.status = EvaluationStatus.SUCCESS

        except Exception as exc:
            result.status = EvaluationStatus.FAILED
            result.warnings.append(
                f"Pipeline analysis failed: {exc}"
            )

        return result

    # ==================================================================
    # Project discovery
    # ==================================================================

    def _find_python_files(
        self,
        project_root: Path,
    ) -> list[Path]:
        """
        Find Python files while ignoring common generated directories.
        """

        ignored_directories = {
            ".git",
            ".venv",
            "venv",
            "env",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "node_modules",
        }

        python_files: list[Path] = []

        for path in project_root.rglob("*.py"):

            if any(
                part in ignored_directories
                for part in path.parts
            ):
                continue

            python_files.append(path)

        return sorted(
            python_files,
            key=lambda p: str(p).lower(),
        )

    def _find_entry_point(
        self,
        python_files: list[Path],
        project_root: Path,
    ) -> str | None:
        """
        Select a likely entry point using simple heuristics.
        """

        # First preference: conventional names.
        for path in python_files:
            if path.name.lower() in self.COMMON_ENTRY_POINTS:
                return str(
                    path.relative_to(project_root)
                )

        # Second preference: file containing a __main__ block.
        for path in python_files:

            try:
                source = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                continue

            if 'if __name__ == "__main__"' in source:
                return str(
                    path.relative_to(project_root)
                )

        # Third preference: main-like filename.
        for path in python_files:

            lowered = path.stem.lower()

            if lowered in {
                "start",
                "execute",
                "predict",
                "inference",
            }:
                return str(
                    path.relative_to(project_root)
                )

        return None

    # ==================================================================
    # Evidence collection
    # ==================================================================

    def _collect_pipeline_evidence(
        self,
        project_root: Path,
        python_files: list[Path],
    ) -> dict:
        """
        Collect static evidence from Python source files.
        """

        evidence = {
            "input": False,
            "preprocessing": False,
            "model": False,
            "output": False,
            "pipeline_files": [],
            "stage_files": {
                "input": [],
                "preprocessing": [],
                "model": [],
                "output": [],
            },
            "symbols": [],
            "imports": [],
        }

        for path in python_files:

            try:
                source = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                continue

            relative_path = str(
                path.relative_to(project_root)
            )

            lowered_source = source.lower()

            tree = self._parse_ast(source)

            symbols = self._extract_symbols(
                tree
            )

            imports = self._extract_imports(
                tree
            )

            evidence["symbols"].extend(
                symbols
            )

            evidence["imports"].extend(
                imports
            )

            stage_hits = {
                "input": self._contains_stage_evidence(
                    lowered_source,
                    symbols,
                    self.INPUT_KEYWORDS,
                ),
                "preprocessing": self._contains_stage_evidence(
                    lowered_source,
                    symbols,
                    self.PREPROCESS_KEYWORDS,
                ),
                "model": self._contains_stage_evidence(
                    lowered_source,
                    symbols,
                    self.MODEL_KEYWORDS,
                ),
                "output": self._contains_stage_evidence(
                    lowered_source,
                    symbols,
                    self.OUTPUT_KEYWORDS,
                ),
            }

            for stage, detected in stage_hits.items():

                if detected:
                    evidence[stage] = True
                    evidence["stage_files"][
                        stage
                    ].append(relative_path)

            if any(stage_hits.values()) or self._is_pipeline_file(
                path
            ):
                evidence["pipeline_files"].append(
                    relative_path
                )

        return evidence

    @staticmethod
    def _parse_ast(
        source: str,
    ) -> ast.AST | None:
        """
        Parse Python source into an AST.
        """

        try:
            return ast.parse(source)
        except SyntaxError:
            return None

    @staticmethod
    def _extract_symbols(
        tree: ast.AST | None,
    ) -> list[str]:
        """
        Extract function/class/variable names.
        """

        if tree is None:
            return []

        symbols: list[str] = []

        for node in ast.walk(tree):

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                symbols.append(
                    node.name
                )

            elif isinstance(
                node,
                (
                    ast.Assign,
                    ast.AnnAssign,
                ),
            ):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )

                for target in targets:
                    if isinstance(
                        target,
                        ast.Name,
                    ):
                        symbols.append(
                            target.id
                        )

        return symbols

    @staticmethod
    def _extract_imports(
        tree: ast.AST | None,
    ) -> list[str]:
        """
        Extract top-level module names from import statements.
        """

        if tree is None:
            return []

        imports: list[str] = []

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

        return PipelineAnalyzer._unique_preserve_order(
            imports
        )

    @staticmethod
    def _contains_stage_evidence(
        source: str,
        symbols: Iterable[str],
        keywords: set[str],
    ) -> bool:
        """
        Determine whether a source file contains reasonable evidence
        for a pipeline stage.

        This intentionally uses heuristic evidence only.
        """

        lowered_symbols = [
            symbol.lower()
            for symbol in symbols
        ]

        # Check named symbols first.
        for symbol in lowered_symbols:
            for keyword in keywords:
                if keyword in symbol:
                    return True

        # Then check token-like appearances in source.
        source_tokens = set(
            re.findall(
                r"[a-zA-Z_][a-zA-Z0-9_]*",
                source,
            )
        )

        for token in source_tokens:
            if token.lower() in keywords:
                return True

        return False

    def _is_pipeline_file(
        self,
        path: Path,
    ) -> bool:
        """
        Decide whether the filename looks related to the pipeline.
        """

        stem = path.stem.lower()

        return any(
            keyword in stem
            for keyword in self.PIPELINE_FILE_KEYWORDS
        )

    # ==================================================================
    # Claimed pipeline
    # ==================================================================

    @staticmethod
    def _build_claimed_pipeline(
        specification: ProjectSpecification,
    ) -> str | None:
        """
        Build a compact representation of the pipeline from
        explicitly supplied information.

        We do not invent stages that the student did not describe.
        """

        pieces: list[str] = []

        if specification.input_type:
            pieces.append(
                specification.input_type
            )

        if specification.expected_behavior:
            pieces.append(
                "expected output"
            )

        if specification.model_info:
            pieces.insert(
                max(len(pieces) - 1, 0),
                specification.model_info,
            )

        if not pieces:
            return None

        return " -> ".join(
            pieces
        )

    # ==================================================================
    # Consistency checks
    # ==================================================================

    def _check_pipeline_consistency(
        self,
        result: PipelineAnalysisResult,
        specification: ProjectSpecification | None,
    ) -> None:
        """
        Check for obvious contradictions without pretending static
        analysis proves functional correctness.
        """

        if not result.input_stage_found:
            result.consistency_issues.append(
                "No clear input-handling stage was detected."
            )

        if not result.model_stage_found:
            result.consistency_issues.append(
                "No clear model/inference stage was detected."
            )

        if not result.output_stage_found:
            result.consistency_issues.append(
                "No clear output-generation stage was detected."
            )

        if specification:

            if specification.run_command:
                command = specification.run_command.lower()

                if ".py" in command:
                    referenced_file = self._extract_python_file(
                        command
                    )

                    if (
                        referenced_file
                        and not self._project_contains_file(
                            specification,
                            referenced_file,
                        )
                    ):
                        result.consistency_issues.append(
                            "The README run command references "
                            f"{referenced_file}, but this cannot be "
                            "confirmed from the current specification."
                        )

            if specification.model_info and not result.model_stage_found:
                result.consistency_issues.append(
                    "The project description mentions a model, "
                    "but static analysis did not find clear "
                    "model/inference evidence."
                )

            if specification.input_type and not result.input_stage_found:
                result.consistency_issues.append(
                    "An input type was documented, but clear "
                    "input-handling code was not detected."
                )

            if specification.expected_behavior and not result.output_stage_found:
                result.consistency_issues.append(
                    "Expected output/behavior was documented, "
                    "but clear output-generation code was not detected."
                )

        if (
            result.input_stage_found
            and result.model_stage_found
            and result.output_stage_found
        ):
            if not result.preprocessing_stage_found:
                result.warnings.append(
                    "No explicit preprocessing stage was detected. "
                    "This may be valid for simple projects."
                )

    @staticmethod
    def _extract_python_file(
        command: str,
    ) -> str | None:
        """
        Extract a Python filename from a run command.
        """

        match = re.search(
            r"([A-Za-z0-9_.-]+\.py)\b",
            command,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        return match.group(1)

    @staticmethod
    def _project_contains_file(
        specification: ProjectSpecification,
        filename: str,
    ) -> bool:
        """
        Placeholder-compatible check using the entry point information.

        Actual repository-level file verification belongs to the
        RepositoryInspector.
        """

        if specification.entry_point:
            return (
                Path(specification.entry_point).name.lower()
                == Path(filename).name.lower()
            )

        return False

    # ==================================================================
    # Output helpers
    # ==================================================================

    @staticmethod
    def _detected_stage_names(
        evidence: dict,
    ) -> list[str]:
        """
        Convert boolean stage evidence into a readable list.
        """

        names: list[str] = []

        mapping = [
            ("input", "Input"),
            ("preprocessing", "Preprocessing"),
            ("model", "Model / Inference"),
            ("output", "Output"),
        ]

        for key, label in mapping:

            if evidence.get(key):
                names.append(label)

        return names

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


def analyze_pipeline(
    project_root: str | Path,
    specification: ProjectSpecification | None = None,
) -> PipelineAnalysisResult:
    """
    Convenience wrapper for the main evaluation pipeline.
    """

    analyzer = PipelineAnalyzer()

    return analyzer.analyze(
        project_root=Path(project_root),
        specification=specification,
    )