from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import Optional

import psutil

from .models import EvaluationStatus, ExecutionResult


class ExecutionEngine:
    """
    Executes a submitted Python project in a controlled subprocess.

    This module is responsible only for runtime execution and measurement.
    It does not decide whether the project itself is correct.
    """

    DEFAULT_TIMEOUT_SECONDS = 60

    PYTHON_COMMANDS = {
        "python",
        "python3",
        "py",
        "python.exe",
        "python3.exe",
    }

    def __init__(
        self,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        project_root: Path,
        command: str,
        timeout_seconds: Optional[int] = None,
        python_executable: Optional[Path | str] = None,
        environment: Optional[dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute a project command.

        Parameters
        ----------
        project_root:
            Directory from which the project should run.

        command:
            Command supplied by the student, e.g.
            "python main.py"

        timeout_seconds:
            Optional per-run timeout.

        python_executable:
            Python interpreter to use. Defaults to the current interpreter.

        environment:
            Optional additional environment variables.
        """

        project_root = Path(project_root)

        result = ExecutionResult(
            status=EvaluationStatus.RUNNING,
            command=command,
            working_directory=project_root,
        )

        if not project_root.exists():
            result.status = EvaluationStatus.FAILED
            result.error_type = "missing_directory"
            result.error_message = (
                f"Project directory does not exist: {project_root}"
            )
            return result

        if not project_root.is_dir():
            result.status = EvaluationStatus.FAILED
            result.error_type = "invalid_directory"
            result.error_message = (
                f"Project path is not a directory: {project_root}"
            )
            return result

        if not command or not command.strip():
            result.status = EvaluationStatus.FAILED
            result.error_type = "missing_command"
            result.error_message = (
                "No execution command was provided."
            )
            return result

        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.timeout_seconds
        )

        if timeout <= 0:
            result.status = EvaluationStatus.FAILED
            result.error_type = "invalid_timeout"
            result.error_message = (
                "Timeout must be greater than zero."
            )
            return result

        try:
            args = self._parse_command(command)

            args = self._normalize_python_command(
                args,
                python_executable,
            )

        except ValueError as exc:
            result.status = EvaluationStatus.FAILED
            result.error_type = "command_parse_error"
            result.error_message = str(exc)
            return result

        process_environment = os.environ.copy()

        # Prevent user-site packages from unexpectedly changing the
        # evaluator environment.
        process_environment["PYTHONNOUSERSITE"] = "1"
        process_environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

        if environment:
            process_environment.update(environment)

        start_time = time.perf_counter()

        try:
            process = subprocess.Popen(
                args,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                env=process_environment,
            )

        except FileNotFoundError as exc:
            result.status = EvaluationStatus.FAILED
            result.error_type = "executable_not_found"
            result.error_message = str(exc)
            return result

        except PermissionError as exc:
            result.status = EvaluationStatus.FAILED
            result.error_type = "permission_error"
            result.error_message = str(exc)
            return result

        except OSError as exc:
            result.status = EvaluationStatus.FAILED
            result.error_type = "process_start_error"
            result.error_message = str(exc)
            return result

        # Start memory monitor.
        memory_stop_event = Event()
        memory_holder = {"peak_mb": 0.0}

        monitor_thread = Thread(
            target=self._monitor_memory,
            args=(
                process.pid,
                memory_stop_event,
                memory_holder,
            ),
            daemon=True,
        )

        monitor_thread.start()

        try:
            stdout, stderr = process.communicate(
                timeout=timeout
            )

            elapsed = time.perf_counter() - start_time

            memory_stop_event.set()
            monitor_thread.join(timeout=2)

            result.stdout = stdout or ""
            result.stderr = stderr or ""

            result.exit_code = process.returncode
            result.execution_time_seconds = round(
                elapsed,
                4,
            )
            result.peak_memory_mb = round(
                memory_holder["peak_mb"],
                2,
            )

            if process.returncode == 0:

                result.status = EvaluationStatus.SUCCESS

            else:

                result.status = EvaluationStatus.FAILED

                result.error_type = (
                    self._classify_runtime_error(
                        stderr=result.stderr,
                        stdout=result.stdout,
                    )
                )

                result.error_message = (
                    self._extract_error_message(
                        stderr=result.stderr,
                        stdout=result.stdout,
                    )
                )

                result.dependency_error = (
                    result.error_type
                    == "dependency_error"
                )

        except subprocess.TimeoutExpired:

            elapsed = time.perf_counter() - start_time

            self._terminate_process_tree(
                process.pid
            )

            try:
                stdout, stderr = process.communicate(
                    timeout=5
                )
            except subprocess.TimeoutExpired:
                process.kill()

                stdout, stderr = process.communicate()

            memory_stop_event.set()
            monitor_thread.join(timeout=2)

            result.status = EvaluationStatus.FAILED
            result.timed_out = True

            result.error_type = "timeout"
            result.error_message = (
                f"Project exceeded the "
                f"{timeout}-second execution limit."
            )

            result.stdout = stdout or ""
            result.stderr = stderr or ""

            result.exit_code = process.returncode

            result.execution_time_seconds = round(
                elapsed,
                4,
            )

            result.peak_memory_mb = round(
                memory_holder["peak_mb"],
                2,
            )

        except Exception as exc:

            memory_stop_event.set()
            monitor_thread.join(timeout=2)

            self._terminate_process_tree(
                process.pid
            )

            result.status = EvaluationStatus.FAILED
            result.error_type = "execution_error"
            result.error_message = str(exc)

            result.execution_time_seconds = round(
                time.perf_counter() - start_time,
                4,
            )

            result.peak_memory_mb = round(
                memory_holder["peak_mb"],
                2,
            )

        return result

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_command(
        command: str,
    ) -> list[str]:
        """
        Parse a command without using shell execution.

        Examples:
            python main.py
            python main.py --input hello
        """

        try:
            arguments = shlex.split(
                command.strip(),
                posix=False,
            )
        except ValueError as exc:
            raise ValueError(
                f"Unable to parse run command: {exc}"
            ) from exc

        cleaned: list[str] = []

        for argument in arguments:

            argument = argument.strip()

            if len(argument) >= 2:
                if (
                    argument[0] == '"'
                    and argument[-1] == '"'
                ) or (
                    argument[0] == "'"
                    and argument[-1] == "'"
                ):
                    argument = argument[1:-1]

            if argument:
                cleaned.append(argument)

        if not cleaned:
            raise ValueError(
                "Execution command is empty."
            )

        return cleaned

    def _normalize_python_command(
        self,
        arguments: list[str],
        python_executable: Optional[Path | str],
    ) -> list[str]:
        """
        Replace generic Python executable names with the selected
        Python interpreter.

        This avoids accidentally invoking a different Python installation.
        """

        if not arguments:
            raise ValueError(
                "Execution command contains no arguments."
            )

        first = Path(arguments[0]).name.lower()

        if first not in {
            command.lower()
            for command in self.PYTHON_COMMANDS
        }:
            return arguments

        interpreter = (
            Path(python_executable)
            if python_executable
            else Path(sys.executable)
        )

        return [
            str(interpreter),
            *arguments[1:],
        ]

    # ------------------------------------------------------------------
    # Memory monitoring
    # ------------------------------------------------------------------

    @staticmethod
    def _monitor_memory(
        pid: int,
        stop_event: Event,
        memory_holder: dict[str, float],
    ) -> None:
        """
        Monitor the process and its descendants.

        Peak memory is measured as the sum of RSS across the process tree.
        """

        try:
            root = psutil.Process(pid)
        except psutil.Error:
            return

        while not stop_event.is_set():

            try:
                processes = [root]

                try:
                    processes.extend(
                        root.children(
                            recursive=True
                        )
                    )
                except psutil.Error:
                    pass

                total_rss = 0

                for process in processes:
                    try:
                        total_rss += (
                            process.memory_info().rss
                        )
                    except (
                        psutil.NoSuchProcess,
                        psutil.AccessDenied,
                    ):
                        continue

                total_mb = (
                    total_rss
                    / (1024 * 1024)
                )

                if total_mb > memory_holder["peak_mb"]:
                    memory_holder["peak_mb"] = total_mb

            except psutil.Error:
                pass

            stop_event.wait(0.05)

    # ------------------------------------------------------------------
    # Process termination
    # ------------------------------------------------------------------

    @staticmethod
    def _terminate_process_tree(
        pid: int,
    ) -> None:
        """
        Terminate a process and its children.
        """

        try:
            root = psutil.Process(pid)
        except psutil.Error:
            return

        try:
            children = root.children(
                recursive=True
            )
        except psutil.Error:
            children = []

        # Terminate children first.
        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                pass

        try:
            root.terminate()
        except psutil.Error:
            pass

        gone, alive = psutil.wait_procs(
            children + [root],
            timeout=2,
        )

        del gone

        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                pass

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_runtime_error(
        stderr: str,
        stdout: str,
    ) -> str:
        """
        Classify common runtime failures.
        """

        combined = (
            f"{stderr}\n{stdout}"
        ).lower()

        if (
            "modulenotfounderror"
            in combined
            or "no module named"
            in combined
            or "importerror"
            in combined
        ):
            return "dependency_error"

        if "syntaxerror" in combined:
            return "syntax_error"

        if "permissionerror" in combined:
            return "permission_error"

        if "filenotfounderror" in combined:
            return "file_not_found_error"

        if "keyerror" in combined:
            return "key_error"

        if "valueerror" in combined:
            return "value_error"

        if "typeerror" in combined:
            return "type_error"

        if "attributeerror" in combined:
            return "attribute_error"

        if "memoryerror" in combined:
            return "memory_error"

        return "runtime_error"

    @staticmethod
    def _extract_error_message(
        stderr: str,
        stdout: str,
    ) -> str:
        """
        Return a compact useful runtime error message.
        """

        source = stderr.strip()

        if not source:
            source = stdout.strip()

        if not source:
            return "Project exited with a non-zero status."

        lines = [
            line.strip()
            for line in source.splitlines()
            if line.strip()
        ]

        # Prefer the final non-empty line because Python tracebacks
        # generally end with the exception type/message.
        if lines:
            return lines[-1][:2000]

        return source[:2000]


def execute_project(
    project_root: str | Path,
    command: str,
    timeout_seconds: int = 60,
    python_executable: Optional[str | Path] = None,
) -> ExecutionResult:
    """
    Convenience wrapper around ExecutionEngine.run().
    """

    engine = ExecutionEngine(
        timeout_seconds=timeout_seconds
    )

    return engine.run(
        project_root=Path(project_root),
        command=command,
        python_executable=python_executable,
    )