from __future__ import annotations

import ast
import re
from pathlib import Path

from .models import CodeQualityResult, EvaluationStatus


class CodeQualityAnalyzer:
    """
    Performs lightweight static analysis of Python project code.

    This module does not execute student code and does not use an LLM.
    """

    def analyze(self, project_root: Path) -> CodeQualityResult:
        result = CodeQualityResult(
            status=EvaluationStatus.RUNNING
        )

        project_root = Path(project_root)

        if not project_root.exists() or not project_root.is_dir():
            result.status = EvaluationStatus.FAILED
            result.static_findings.append(
                "Project directory does not exist."
            )
            return result

        python_files = self._find_python_files(project_root)

        if not python_files:
            result.status = EvaluationStatus.FAILED
            result.static_findings.append(
                "No Python source files were found."
            )
            return result

        total_functions = 0
        total_complexity = 0
        files_with_errors = 0

        for path in python_files:
            try:
                source = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

                tree = ast.parse(source)

            except SyntaxError as exc:
                files_with_errors += 1
                result.static_findings.append(
                    f"Syntax error in {path.name}: {exc.msg}."
                )
                continue

            except OSError as exc:
                files_with_errors += 1
                result.static_findings.append(
                    f"Unable to read {path.name}: {exc}."
                )
                continue

            total_functions += self._count_functions(tree)
            total_complexity += self._estimate_complexity(tree)

            self._check_function_structure(
                tree,
                path,
                result,
            )

            self._check_unused_imports(
                tree,
                path,
                result,
            )

            self._check_error_handling(
                tree,
                path,
                result,
            )

            self._check_dangerous_or_secret_patterns(
                source,
                path,
                result,
            )

            self._check_long_functions(
                tree,
                path,
                result,
            )

            self._check_global_hard_coding(
                tree,
                path,
                result,
            )

        # ------------------------------------------------------------------
        # Basic derived scores
        # ------------------------------------------------------------------

        readability = 15.0
        maintainability = 15.0
        complexity_score = 15.0
        error_handling_score = 15.0

        # Deduct for major static issues.
        readability -= min(
            5.0,
            len(result.unused_code_findings) * 0.5,
        )

        maintainability -= min(
            5.0,
            len(result.duplicated_code_findings) * 0.75,
        )

        complexity_score -= min(
            7.0,
            max(0, total_complexity - 10) * 0.25,
        )

        error_handling_score -= min(
            7.0,
            len(result.security_findings) * 1.0,
        )

        if files_with_errors:
            readability -= 3.0

        result.readability_score = self._clamp(
            readability,
            0,
            15,
        )

        result.maintainability_score = self._clamp(
            maintainability,
            0,
            15,
        )

        result.complexity_score = self._clamp(
            complexity_score,
            0,
            15,
        )

        result.error_handling_score = self._clamp(
            error_handling_score,
            0,
            15,
        )

        if total_functions == 0:
            result.static_findings.append(
                "No functions/classes were detected. "
                "This may be appropriate for a very small project, "
                "but larger projects should be modular."
            )

        result.status = EvaluationStatus.SUCCESS

        return result

    # ==================================================================
    # Discovery
    # ==================================================================

    @staticmethod
    def _find_python_files(
        project_root: Path,
    ) -> list[Path]:
        ignored = {
            ".git",
            ".venv",
            "venv",
            "env",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        }

        files = []

        for path in project_root.rglob("*.py"):

            if any(
                part in ignored
                for part in path.parts
            ):
                continue

            files.append(path)

        return sorted(files)

    # ==================================================================
    # Structural checks
    # ==================================================================

    @staticmethod
    def _count_functions(
        tree: ast.AST,
    ) -> int:
        return sum(
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            for node in ast.walk(tree)
        )

    def _check_function_structure(
        self,
        tree: ast.AST,
        path: Path,
        result: CodeQualityResult,
    ) -> None:
        for node in ast.walk(tree):

            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            argument_count = len(node.args.args)

            if argument_count > 7:
                result.static_findings.append(
                    f"{path.name}: function '{node.name}' "
                    f"has {argument_count} parameters; consider "
                    "splitting responsibilities."
                )

    # ==================================================================
    # Complexity
    # ==================================================================

    @staticmethod
    def _estimate_complexity(
        tree: ast.AST,
    ) -> int:
        """
        Lightweight cyclomatic-style estimate.

        Starts at 1 and adds for branching constructs.
        """

        complexity = 1

        branching_nodes = (
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.IfExp,
        )

        for node in ast.walk(tree):

            if isinstance(
                node,
                branching_nodes,
            ):
                complexity += 1

            elif isinstance(node, ast.BoolOp):
                complexity += max(
                    0,
                    len(node.values) - 1,
                )

            elif isinstance(node, ast.ExceptHandler):
                complexity += 1

        return complexity

    # ==================================================================
    # Unused imports
    # ==================================================================

    def _check_unused_imports(
        self,
        tree: ast.AST,
        path: Path,
        result: CodeQualityResult,
    ) -> None:
        imported_names: list[str] = []

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):

                for alias in node.names:
                    imported_names.append(
                        alias.asname or alias.name.split(".")[0]
                    )

            elif isinstance(node, ast.ImportFrom):

                for alias in node.names:
                    imported_names.append(
                        alias.asname or alias.name
                    )

        source_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }

        for name in imported_names:

            if name not in source_names:
                result.unused_code_findings.append(
                    f"{path.name}: imported name "
                    f"'{name}' may be unused."
                )

    # ==================================================================
    # Error handling
    # ==================================================================

    def _check_error_handling(
        self,
        tree: ast.AST,
        path: Path,
        result: CodeQualityResult,
    ) -> None:
        has_try = any(
            isinstance(node, ast.Try)
            for node in ast.walk(tree)
        )

        has_raise = any(
            isinstance(node, ast.Raise)
            for node in ast.walk(tree)
        )

        has_input = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "input"
            for node in ast.walk(tree)
        )

        if has_input and not has_try:
            result.static_findings.append(
                f"{path.name}: uses interactive input without "
                "visible exception handling."
            )

        if not has_try and not has_raise:
            result.static_findings.append(
                f"{path.name}: no explicit exception handling "
                "was detected."
            )

    # ==================================================================
    # Secret/security checks
    # ==================================================================

    def _check_dangerous_or_secret_patterns(
        self,
        source: str,
        path: Path,
        result: CodeQualityResult,
    ) -> None:
        secret_patterns = [
            re.compile(
                r"(api[_-]?key|secret[_-]?key|access[_-]?token)"
                r"\s*=\s*['\"][^'\"]+['\"]",
                re.IGNORECASE,
            ),
            re.compile(
                r"sk-[A-Za-z0-9]{20,}",
                re.IGNORECASE,
            ),
        ]

        for pattern in secret_patterns:

            if pattern.search(source):
                result.security_findings.append(
                    f"{path.name}: possible hard-coded secret "
                    "or API credential detected."
                )
                break

        # Highlight unrestricted shell execution.
        if re.search(
            r"\b(os\.system|subprocess\.(run|Popen|call))\s*\(",
            source,
        ):
            result.security_findings.append(
                f"{path.name}: shell/process execution detected; "
                "execution should be reviewed for safety."
            )

    # ==================================================================
    # Long functions
    # ==================================================================

    def _check_long_functions(
        self,
        tree: ast.AST,
        path: Path,
        result: CodeQualityResult,
    ) -> None:
        for node in ast.walk(tree):

            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            if not node.body:
                continue

            start = node.lineno

            end = getattr(
                node,
                "end_lineno",
                start,
            )

            length = end - start + 1

            if length > 80:
                result.complexity_findings.append(
                    f"{path.name}: function '{node.name}' "
                    f"is approximately {length} lines long."
                )

            elif length > 50:
                result.complexity_findings.append(
                    f"{path.name}: function '{node.name}' "
                    f"is relatively long ({length} lines)."
                )

    # ==================================================================
    # Hard-coded values
    # ==================================================================

    def _check_global_hard_coding(
        self,
        tree: ast.AST,
        path: Path,
        result: CodeQualityResult,
    ) -> None:
        for node in tree.body:

            if not isinstance(node, ast.Assign):
                continue

            if not node.targets:
                continue

            target = node.targets[0]

            if not isinstance(target, ast.Name):
                continue

            name = target.id.upper()

            if any(
                keyword in name
                for keyword in (
                    "KEY",
                    "TOKEN",
                    "PASSWORD",
                    "SECRET",
                )
            ):
                if isinstance(node.value, ast.Constant):

                    result.security_findings.append(
                        f"{path.name}: configuration variable "
                        f"'{target.id}' is defined directly in source."
                    )

    # ==================================================================
    # Utilities
    # ==================================================================

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(
            minimum,
            min(
                maximum,
                round(value, 2),
            ),
        )


def analyze_code_quality(
    project_root: str | Path,
) -> CodeQualityResult:
    """
    Convenience wrapper.
    """

    analyzer = CodeQualityAnalyzer()

    return analyzer.analyze(
        Path(project_root)
    )