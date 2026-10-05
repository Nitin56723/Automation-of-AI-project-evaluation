from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class SubmissionType(str, Enum):
    """Type of project submission."""

    GITHUB = "github"
    ZIP = "zip"


class EvaluationStatus(str, Enum):
    """Status used by each evaluation stage."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class TestStatus(str, Enum):
    """Result of an individual functional test."""

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_TESTABLE = "not_testable"


@dataclass
class StudentSubmission:
    """
    Information entered by the student through the submission form.
    """

    project_name: str
    project_goal: str
    run_command: str
    input_type: str
    expected_behavior: str

    github_url: Optional[str] = None
    zip_path: Optional[Path] = None

    sample_input: Optional[str] = None
    sample_expected_output: Optional[str] = None

    python_version: Optional[str] = None
    external_api_service: Optional[str] = None
    dataset_model_info: Optional[str] = None
    additional_instructions: Optional[str] = None

    submission_type: Optional[SubmissionType] = None

    def validate(self) -> list[str]:
        """Validate mandatory submission information."""

        errors: list[str] = []

        required_fields = {
            "project_name": self.project_name,
            "project_goal": self.project_goal,
            "run_command": self.run_command,
            "input_type": self.input_type,
            "expected_behavior": self.expected_behavior,
        }

        for field_name, value in required_fields.items():
            if not value or not value.strip():
                errors.append(f"{field_name} is required.")

        if not self.github_url and not self.zip_path:
            errors.append(
                "Either a GitHub repository URL or a ZIP file must be provided."
            )

        if self.github_url and self.zip_path:
            errors.append(
                "Provide either a GitHub repository URL or a ZIP file, not both."
            )

        if self.github_url:
            self.submission_type = SubmissionType.GITHUB

        elif self.zip_path:
            self.submission_type = SubmissionType.ZIP

        return errors


@dataclass
class ProjectSpecification:
    """
    Structured understanding of what the student says the project does.
    """

    project_name: str = ""
    project_goal: str = ""

    run_command: Optional[str] = None
    python_version: Optional[str] = None

    input_type: Optional[str] = None
    expected_behavior: Optional[str] = None

    dependencies: list[str] = field(default_factory=list)

    sample_test_cases: list[dict[str, Any]] = field(default_factory=list)

    entry_point: Optional[str] = None

    external_requirements: list[str] = field(default_factory=list)

    dataset_info: Optional[str] = None
    model_info: Optional[str] = None

    extraction_warnings: list[str] = field(default_factory=list)

    source_information: dict[str, str] = field(default_factory=dict)


@dataclass
class RepositoryInspectionResult:
    """
    Results from inspecting the submitted repository without executing it.
    """

    root_path: Optional[Path] = None

    total_files: int = 0
    total_directories: int = 0

    python_files: list[str] = field(default_factory=list)
    other_files: list[str] = field(default_factory=list)

    entry_point_candidates: list[str] = field(default_factory=list)

    dependency_files: list[str] = field(default_factory=list)
    detected_imports: list[str] = field(default_factory=list)

    large_files: list[dict[str, Any]] = field(default_factory=list)

    model_files: list[str] = field(default_factory=list)
    dataset_files: list[str] = field(default_factory=list)

    potential_secrets: list[str] = field(default_factory=list)
    external_configuration: list[str] = field(default_factory=list)

    resource_warnings: list[str] = field(default_factory=list)
    inspection_warnings: list[str] = field(default_factory=list)

    status: EvaluationStatus = EvaluationStatus.NOT_STARTED


@dataclass
class DependencyCheckResult:
    """
    Dependency and Python environment validation results.
    """

    requirements_file: Optional[str] = None

    declared_dependencies: list[str] = field(default_factory=list)
    detected_imports: list[str] = field(default_factory=list)

    missing_dependencies: list[str] = field(default_factory=list)
    install_failures: list[str] = field(default_factory=list)
    incompatible_dependencies: list[str] = field(default_factory=list)

    python_version_requested: Optional[str] = None
    python_version_used: Optional[str] = None

    install_success: bool = False

    status: EvaluationStatus = EvaluationStatus.NOT_STARTED

    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineAnalysisResult:
    """
    Static analysis of the project's claimed and detected pipeline.
    """

    claimed_pipeline: Optional[str] = None

    detected_stages: list[str] = field(default_factory=list)

    entry_point: Optional[str] = None

    pipeline_files: list[str] = field(default_factory=list)

    input_stage_found: bool = False
    preprocessing_stage_found: bool = False
    model_stage_found: bool = False
    output_stage_found: bool = False

    consistency_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    status: EvaluationStatus = EvaluationStatus.NOT_STARTED


@dataclass
class ExecutionResult:
    """
    Results from actually executing the student's project.
    """

    status: EvaluationStatus = EvaluationStatus.NOT_STARTED

    command: Optional[str] = None

    exit_code: Optional[int] = None

    stdout: str = ""
    stderr: str = ""

    execution_time_seconds: Optional[float] = None
    peak_memory_mb: Optional[float] = None

    timed_out: bool = False
    dependency_error: bool = False

    error_type: Optional[str] = None
    error_message: Optional[str] = None

    working_directory: Optional[Path] = None


@dataclass
class TestCase:
    """
    A functional test case.

    source identifies where the test came from:
    - student_provided
    - readme
    - ai_generated
    """

    test_id: str
    input_data: Any

    expected_output: Optional[Any] = None

    source: str = "unknown"

    description: Optional[str] = None


@dataclass
class TestResult:
    """Result of running one functional test."""

    test_id: str

    status: TestStatus

    actual_output: Any = None
    expected_output: Any = None

    execution_time_seconds: Optional[float] = None

    error_message: Optional[str] = None

    source: str = "unknown"

    notes: list[str] = field(default_factory=list)


@dataclass
class FunctionalTestResult:
    """Aggregate results of all functional tests."""

    test_cases: list[TestCase] = field(default_factory=list)

    test_results: list[TestResult] = field(default_factory=list)

    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    timeout_tests: int = 0
    not_testable_tests: int = 0

    status: EvaluationStatus = EvaluationStatus.NOT_STARTED


@dataclass
class PerformanceResult:
    """Measured performance and efficiency information."""

    startup_time_seconds: Optional[float] = None
    total_execution_time_seconds: Optional[float] = None
    peak_memory_mb: Optional[float] = None

    dependency_count: int = 0

    efficiency_findings: list[str] = field(default_factory=list)
    resource_warnings: list[str] = field(default_factory=list)

    measured: bool = False

    status: EvaluationStatus = EvaluationStatus.NOT_STARTED


@dataclass
class CodeQualityResult:
    """Static and AI-assisted code quality findings."""

    readability_score: Optional[float] = None
    maintainability_score: Optional[float] = None
    complexity_score: Optional[float] = None
    error_handling_score: Optional[float] = None

    duplicated_code_findings: list[str] = field(default_factory=list)
    unused_code_findings: list[str] = field(default_factory=list)
    complexity_findings: list[str] = field(default_factory=list)
    security_findings: list[str] = field(default_factory=list)

    static_findings: list[str] = field(default_factory=list)
    ai_findings: list[str] = field(default_factory=list)

    status: EvaluationStatus = EvaluationStatus.NOT_STARTED


@dataclass
class EvaluationScores:
    """
    Final score out of 100.

    Maximum points:
    Functionality & Test Results        = 30
    Pipeline Correctness                = 20
    Performance & Efficiency            = 15
    Code Quality                        = 15
    Error Handling                       = 5
    Dependencies / Environment          = 5
    Documentation                        = 5
    AI Project Completeness              = 5
    """

    functionality: float = 0.0
    pipeline_correctness: float = 0.0
    performance_efficiency: float = 0.0
    code_quality: float = 0.0
    error_handling: float = 0.0
    dependencies_environment: float = 0.0
    documentation: float = 0.0
    ai_project_completeness: float = 0.0

    @property
    def total(self) -> float:
        """Return the final score out of 100."""

        return round(
            self.functionality
            + self.pipeline_correctness
            + self.performance_efficiency
            + self.code_quality
            + self.error_handling
            + self.dependencies_environment
            + self.documentation
            + self.ai_project_completeness,
            2,
        )


@dataclass
class EvaluationReport:
    """
    Complete evaluation result for one student submission.
    """

    submission: Optional[StudentSubmission] = None

    specification: Optional[ProjectSpecification] = None

    repository_inspection: Optional[RepositoryInspectionResult] = None

    dependency_check: Optional[DependencyCheckResult] = None

    pipeline_analysis: Optional[PipelineAnalysisResult] = None

    execution: Optional[ExecutionResult] = None

    functional_testing: Optional[FunctionalTestResult] = None

    performance: Optional[PerformanceResult] = None

    code_quality: Optional[CodeQualityResult] = None

    scores: EvaluationScores = field(default_factory=EvaluationScores)

    strengths: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    evaluation_id: Optional[str] = None

    status: EvaluationStatus = EvaluationStatus.NOT_STARTED

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the complete report into a JSON-compatible dictionary.
        """

        from dataclasses import asdict

        data = asdict(self)

        def normalize(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value

            if isinstance(value, Path):
                return str(value)

            if isinstance(value, dict):
                return {
                    str(key): normalize(item)
                    for key, item in value.items()
                }

            if isinstance(value, list):
                return [normalize(item) for item in value]

            return value

        return normalize(data)