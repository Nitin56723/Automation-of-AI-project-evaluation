from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .execution_engine import ExecutionEngine
from .models import (
    EvaluationStatus,
    FunctionalTestResult,
    TestCase,
    TestResult,
    TestStatus,
)


class FunctionalTester:
    """
    Runs functional test cases against a submitted Python project.

    The tester uses the execution engine for controlled subprocess
    execution and compares actual output against explicitly supplied
    expected output.

    It does not invent expected outputs.
    """

    def __init__(
        self,
        execution_engine: Optional[ExecutionEngine] = None,
        default_timeout_seconds: int = 30,
    ):
        self.execution_engine = (
            execution_engine
            or ExecutionEngine(
                timeout_seconds=default_timeout_seconds
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_tests(
        self,
        project_root: Path,
        command: str,
        test_cases: list[TestCase],
        timeout_seconds: int = 30,
    ) -> FunctionalTestResult:
        """
        Execute all supplied test cases.

        Each test runs independently.
        One failed test does not prevent the remaining tests from running.
        """

        result = FunctionalTestResult(
            test_cases=test_cases,
            status=EvaluationStatus.RUNNING,
        )

        if not project_root.exists():
            result.status = EvaluationStatus.FAILED
            return result

        if not test_cases:
            result.status = EvaluationStatus.PARTIAL
            return result

        for test_case in test_cases:

            test_result = self._run_single_test(
                project_root=project_root,
                command=command,
                test_case=test_case,
                timeout_seconds=timeout_seconds,
            )

            result.test_results.append(test_result)

        self._calculate_summary(result)

        if result.error_tests == result.total_tests:
            result.status = EvaluationStatus.FAILED

        elif result.passed_tests == result.total_tests:
            result.status = EvaluationStatus.SUCCESS

        else:
            result.status = EvaluationStatus.PARTIAL

        return result

    # ------------------------------------------------------------------
    # Single test
    # ------------------------------------------------------------------

    def _run_single_test(
        self,
        project_root: Path,
        command: str,
        test_case: TestCase,
        timeout_seconds: int,
    ) -> TestResult:
        """
        Run one functional test.

        The current MVP sends the input through stdin for Python commands.
        """

        started_at = time.perf_counter()

        if test_case.expected_output is None:
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.NOT_TESTABLE,
                expected_output=None,
                source=test_case.source,
                notes=[
                    "No explicit expected output was provided."
                ],
            )

        try:
            arguments = self.execution_engine._parse_command(
                command
            )

            arguments = self.execution_engine._normalize_python_command(
                arguments,
                python_executable=sys.executable,
            )

        except ValueError as exc:
            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.ERROR,
                error_message=str(exc),
                source=test_case.source,
            )

        input_text = self._serialize_input(
            test_case.input_data
        )

        execution_result = self._run_with_input(
            project_root=project_root,
            arguments=arguments,
            input_text=input_text,
            timeout_seconds=timeout_seconds,
        )

        elapsed = time.perf_counter() - started_at

        if execution_result["timed_out"]:

            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.TIMEOUT,
                actual_output=execution_result["stdout"],
                expected_output=test_case.expected_output,
                execution_time_seconds=round(
                    elapsed,
                    4,
                ),
                error_message=(
                    execution_result["error_message"]
                ),
                source=test_case.source,
            )

        if execution_result["returncode"] != 0:

            stderr = execution_result["stderr"]

            error_message = (
                self._last_error_line(stderr)
                or "Project returned a non-zero exit code."
            )

            return TestResult(
                test_id=test_case.test_id,
                status=TestStatus.ERROR,
                actual_output=execution_result["stdout"],
                expected_output=test_case.expected_output,
                execution_time_seconds=round(
                    elapsed,
                    4,
                ),
                error_message=error_message,
                source=test_case.source,
            )

        actual_output = (
            execution_result["stdout"]
        )

        passed, comparison_note = (
            self._compare_outputs(
                actual_output,
                test_case.expected_output,
            )
        )

        return TestResult(
            test_id=test_case.test_id,
            status=(
                TestStatus.PASS
                if passed
                else TestStatus.FAIL
            ),
            actual_output=actual_output,
            expected_output=test_case.expected_output,
            execution_time_seconds=round(
                elapsed,
                4,
            ),
            source=test_case.source,
            notes=(
                [comparison_note]
                if comparison_note
                else []
            ),
        )

    # ------------------------------------------------------------------
    # Runtime with stdin
    # ------------------------------------------------------------------

    def _run_with_input(
        self,
        project_root: Path,
        arguments: list[str],
        input_text: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """
        Execute a command while supplying the test input to stdin.

        This function intentionally uses shell=False.
        """

        process: Optional[subprocess.Popen] = None

        started_at = time.perf_counter()

        try:
            process = subprocess.Popen(
                arguments,
                cwd=project_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )

            try:
                stdout, stderr = process.communicate(
                    input=input_text,
                    timeout=timeout_seconds,
                )

            except subprocess.TimeoutExpired:

                process.kill()

                stdout, stderr = process.communicate()

                return {
                    "returncode": process.returncode,
                    "stdout": stdout or "",
                    "stderr": stderr or "",
                    "timed_out": True,
                    "error_message": (
                        f"Test exceeded the "
                        f"{timeout_seconds}-second limit."
                    ),
                    "execution_time": (
                        time.perf_counter()
                        - started_at
                    ),
                }

            return {
                "returncode": process.returncode,
                "stdout": stdout or "",
                "stderr": stderr or "",
                "timed_out": False,
                "error_message": None,
                "execution_time": (
                    time.perf_counter()
                    - started_at
                ),
            }

        except FileNotFoundError as exc:

            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
                "error_message": str(exc),
                "execution_time": (
                    time.perf_counter()
                    - started_at
                ),
            }

        except Exception as exc:

            if process is not None:
                try:
                    process.kill()
                except OSError:
                    pass

            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
                "error_message": str(exc),
                "execution_time": (
                    time.perf_counter()
                    - started_at
                ),
            }

    # ------------------------------------------------------------------
    # Output comparison
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_outputs(
        actual: Any,
        expected: Any,
    ) -> tuple[bool, Optional[str]]:
        """
        Compare actual and expected output conservatively.

        Comparison order:
        1. Exact structured equality where possible.
        2. Normalized textual equality.
        3. Numeric equality with small floating-point tolerance.

        We intentionally do NOT ask an LLM to decide whether arbitrary
        outputs are semantically equivalent at this stage.
        """

        # Structured objects
        if isinstance(actual, (dict, list)) or isinstance(
            expected,
            (dict, list),
        ):
            actual_parsed = FunctionalTester._try_json_parse(
                actual
            )

            expected_parsed = FunctionalTester._try_json_parse(
                expected
            )

            if (
                actual_parsed is not None
                and expected_parsed is not None
            ):
                if actual_parsed == expected_parsed:
                    return True, "JSON/structured output matched."

                return False, "JSON/structured output differed."

        # Numeric outputs
        actual_number = FunctionalTester._try_number(
            actual
        )

        expected_number = FunctionalTester._try_number(
            expected
        )

        if (
            actual_number is not None
            and expected_number is not None
        ):
            tolerance = max(
                1e-6,
                abs(expected_number) * 1e-5,
            )

            if abs(actual_number - expected_number) <= tolerance:
                return True, "Numeric output matched within tolerance."

            return False, "Numeric output differed beyond tolerance."

        # Text comparison
        actual_text = FunctionalTester._normalize_output(
            actual
        )

        expected_text = FunctionalTester._normalize_output(
            expected
        )

        if actual_text == expected_text:
            return True, "Normalized text output matched."

        return False, "Normalized text output differed."

    @staticmethod
    def _normalize_output(
        value: Any,
    ) -> str:
        """
        Normalize text for reasonable deterministic comparison.
        """

        if value is None:
            return ""

        text = str(value)

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove trailing whitespace from each line.
        lines = [
            line.rstrip()
            for line in text.splitlines()
        ]

        normalized = "\n".join(lines).strip()

        # Collapse repeated whitespace for single-line outputs.
        if "\n" not in normalized:
            normalized = " ".join(
                normalized.split()
            )

        return normalized.casefold()

    @staticmethod
    def _try_json_parse(
        value: Any,
    ) -> Any:
        """
        Attempt JSON parsing without raising.
        """

        if isinstance(value, (dict, list)):
            return value

        if not isinstance(value, str):
            return None

        text = value.strip()

        if not text:
            return None

        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _try_number(
        value: Any,
    ) -> Optional[float]:
        """
        Attempt to convert a scalar value into a float.
        """

        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if not isinstance(value, str):
            return None

        text = value.strip()

        try:
            return float(text)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Result summary
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_summary(
        result: FunctionalTestResult,
    ) -> None:
        """
        Calculate aggregate test counts.
        """

        result.total_tests = len(
            result.test_results
        )

        result.passed_tests = sum(
            test.status == TestStatus.PASS
            for test in result.test_results
        )

        result.failed_tests = sum(
            test.status == TestStatus.FAIL
            for test in result.test_results
        )

        result.error_tests = sum(
            test.status == TestStatus.ERROR
            for test in result.test_results
        )

        result.timeout_tests = sum(
            test.status == TestStatus.TIMEOUT
            for test in result.test_results
        )

        result.not_testable_tests = sum(
            test.status == TestStatus.NOT_TESTABLE
            for test in result.test_results
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_input(
        input_data: Any,
    ) -> str:
        """
        Convert a test input to stdin text.
        """

        if input_data is None:
            return ""

        if isinstance(
            input_data,
            (dict, list),
        ):
            return json.dumps(
                input_data,
                ensure_ascii=False,
            ) + "\n"

        return str(input_data) + "\n"

    @staticmethod
    def _last_error_line(
        stderr: str,
    ) -> Optional[str]:
        """
        Extract the final useful error line.
        """

        lines = [
            line.strip()
            for line in stderr.splitlines()
            if line.strip()
        ]

        if not lines:
            return None

        return lines[-1][:2000]


def run_functional_tests(
    project_root: str | Path,
    command: str,
    test_cases: list[TestCase],
    timeout_seconds: int = 30,
) -> FunctionalTestResult:
    """
    Convenience wrapper for functional testing.
    """

    tester = FunctionalTester()

    return tester.run_tests(
        project_root=Path(project_root),
        command=command,
        test_cases=test_cases,
        timeout_seconds=timeout_seconds,
    )