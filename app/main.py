"""
main.py
-------
Complete evaluation orchestration and Gradio UI for the
NxtWave AI Project Evaluator.

Architecture:
    UI (Gradio)
      ↓
    Submission Handler
      ↓
    README Parser
      ↓
    Repository Inspector
      ↓
    Dependency Checker
      ↓
    Pipeline Analyzer
      ↓
    Execution Engine
      ↓
    Functional Tester
      ↓
    Performance Analyzer
      ↓
    Code Quality Analyzer
      ↓
    Gemini Service (selective)
      ↓
    Scoring Engine
      ↓
    Report Generator
      ↓
    UI Result
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Generator, Optional

import gradio as gr

# ---------------------------------------------------------------------------
# Set up paths so `app` package is importable when run directly
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.code_quality import CodeQualityAnalyzer
from app.dependency_checker import DependencyChecker
from app.execution_engine import ExecutionEngine
from app.functional_tester import FunctionalTester
from app.llm_service import get_llm_service
from app.models import (
    EvaluationReport,
    EvaluationStatus,
    ProjectSpecification,
    StudentSubmission,
    SubmissionType,
    TestCase,
)
from app.performance_analyzer import PerformanceAnalyzer
from app.pipeline_analyzer import PipelineAnalyzer
from app.readme_parser import ReadmeParser
from app.report_generator import ReportGenerator, _generate_id
from app.repository_inspector import RepositoryInspector
from app.scoring_engine import ScoringEngine
from app.submission_handler import SubmissionError, SubmissionHandler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_stream_handler = logging.StreamHandler()
_stream_handler.stream = open(
    _stream_handler.stream.fileno(),
    mode="w",
    encoding="utf-8",
    errors="replace",
    closefd=False,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        _stream_handler,
        logging.FileHandler(
            LOG_DIR / "evaluator.log",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-relative paths
# ---------------------------------------------------------------------------

# ROOT = .../NxtWave_AI_Project_Evaluator/  (parent of app/)
PROJECT_ROOT = ROOT
WORKSPACES_DIR = PROJECT_ROOT / "workspaces"
RESULTS_DIR = PROJECT_ROOT / "results"
WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Core evaluation orchestrator
# ===========================================================================


def run_evaluation(
    submission: StudentSubmission,
    progress_callback,   # callable(stage: str, detail: str)
) -> EvaluationReport:
    """
    Execute the complete evaluation pipeline.

    Progress is reported via progress_callback(stage, detail).
    All exceptions are caught per-stage; evaluation continues where possible.
    """

    eval_id = _generate_id()
    report = EvaluationReport(
        submission=submission,
        evaluation_id=eval_id,
        status=EvaluationStatus.RUNNING,
    )

    workspace: Optional[Path] = None

    try:
        # ----------------------------------------------------------------
        # STAGE 1: Prepare submission workspace
        # ----------------------------------------------------------------
        progress_callback("[1/10] Preparing submission...", "Cloning / extracting project")

        handler = SubmissionHandler(workspace_root=WORKSPACES_DIR)

        try:
            workspace = handler.prepare_submission(submission)
            logger.info("Workspace prepared: %s", workspace)
        except SubmissionError as exc:
            report.status = EvaluationStatus.FAILED
            report.problems.append(f"Submission preparation failed: {exc}")
            report.limitations.append(
                "Evaluation could not proceed – submission could not be prepared."
            )
            return report

        # Find actual project root (handle single-folder ZIP / clone layout)
        project_root = _resolve_project_root(workspace)

        # ----------------------------------------------------------------
        # STAGE 2: Parse README / project spec
        # ----------------------------------------------------------------
        progress_callback("[2/10] Parsing project specification...", "Reading README and form data")

        readme_path = _find_readme(project_root)
        parser = ReadmeParser()

        try:
            spec = parser.parse_file(readme_path, submission=submission)
        except Exception as exc:
            logger.warning("README parsing error: %s", exc)
            spec = ProjectSpecification(
                project_name=submission.project_name,
                project_goal=submission.project_goal,
                run_command=submission.run_command,
                input_type=submission.input_type,
                expected_behavior=submission.expected_behavior,
            )
            spec.extraction_warnings.append(str(exc))

        report.specification = spec

        # ----------------------------------------------------------------
        # STAGE 3: Repository inspection
        # ----------------------------------------------------------------
        progress_callback("[3/10] Inspecting repository...", "Analysing files and structure")

        inspector = RepositoryInspector()
        repo_result = inspector.inspect(project_root)
        report.repository_inspection = repo_result

        # Update entry point from inspection if not already set
        if not spec.entry_point and repo_result.entry_point_candidates:
            spec.entry_point = repo_result.entry_point_candidates[0]

        # ----------------------------------------------------------------
        # STAGE 4: Dependency check
        # ----------------------------------------------------------------
        progress_callback("[4/10] Checking dependencies...", "Analysing declared and detected packages")

        dep_checker = DependencyChecker()
        dep_result = dep_checker.inspect(project_root, repository_result=repo_result)
        report.dependency_check = dep_result

        # ----------------------------------------------------------------
        # STAGE 5: Pipeline analysis
        # ----------------------------------------------------------------
        progress_callback("[5/10] Analysing pipeline...", "Static inspection of AI/ML stages")

        pipeline_analyzer = PipelineAnalyzer()
        pipeline_result = pipeline_analyzer.analyze(project_root, specification=spec)
        report.pipeline_analysis = pipeline_result

        if pipeline_result.entry_point and not spec.entry_point:
            spec.entry_point = pipeline_result.entry_point

        # ----------------------------------------------------------------
        # STAGE 6: Controlled execution
        # ----------------------------------------------------------------
        progress_callback("[6/10] Executing project...", "Running with timeout and resource limits")

        execution_result = None

        # Determine the run command
        run_command = _resolve_run_command(spec, repo_result)

        if run_command:
            # Check for resource-heavy indicators before executing
            resource_issues = _check_resource_safety(repo_result)

            if resource_issues:
                report.limitations.extend(resource_issues)
                report.problems.append(
                    "Project appears resource-heavy or unsupported for execution."
                )
                progress_callback("[WARN] Stage 6/10: Execution skipped (resource concerns)", "Static evaluation only")
            else:
                engine = ExecutionEngine(timeout_seconds=60)

                # Build safe environment (no evaluator API keys)
                safe_env = _build_safe_environment()

                execution_result = engine.run(
                    project_root=project_root,
                    command=run_command,
                    environment=safe_env,
                )

                if execution_result.status == EvaluationStatus.SUCCESS:
                    progress_callback("[OK] Stage 6/10: Execution succeeded", f"Exit 0 in {execution_result.execution_time_seconds:.2f}s")
                else:
                    progress_callback(
                        "[WARN] Stage 6/10: Execution failed",
                        execution_result.error_message or "Non-zero exit",
                    )
        else:
            report.limitations.append(
                "No run command could be determined; execution was skipped."
            )
            progress_callback("[WARN] Stage 6/10: No run command found", "Execution skipped")

        report.execution = execution_result

        # ----------------------------------------------------------------
        # STAGE 7: Functional tests
        # ----------------------------------------------------------------
        progress_callback("[7/10] Running functional tests...", "Executing test cases")

        test_cases = _build_test_cases(spec)

        if not test_cases:
            # Try to generate candidate tests via Gemini
            llm = get_llm_service()
            candidate_tests = llm.generate_candidate_tests(
                project_name=submission.project_name,
                project_goal=submission.project_goal,
                input_type=submission.input_type,
                expected_behavior=submission.expected_behavior,
            )

            if candidate_tests:
                for i, ct in enumerate(candidate_tests):
                    test_cases.append(
                        TestCase(
                            test_id=f"ai_generated_{i+1}",
                            input_data=ct.get("input", ""),
                            expected_output=ct.get("expected_output"),
                            source="ai_generated",
                            description="AI-generated candidate test (not guaranteed correct)",
                        )
                    )

        functional_result = None

        if run_command and test_cases:
            tester = FunctionalTester()

            functional_result = tester.run_tests(
                project_root=project_root,
                command=run_command,
                test_cases=test_cases,
                timeout_seconds=30,
            )
        else:
            from app.models import FunctionalTestResult
            functional_result = FunctionalTestResult(
                status=EvaluationStatus.SKIPPED,
            )

        report.functional_testing = functional_result

        progress_callback(
            "[OK] Stage 7/10: Tests complete",
            _test_summary_text(functional_result),
        )

        # ----------------------------------------------------------------
        # STAGE 8: Performance analysis
        # ----------------------------------------------------------------
        progress_callback("[8/10] Measuring performance...", "Analysing runtime and efficiency")

        perf_analyzer = PerformanceAnalyzer()
        perf_result = perf_analyzer.analyze(
            project_root=project_root,
            execution=execution_result,
            repository=repo_result,
        )
        report.performance = perf_result

        # ----------------------------------------------------------------
        # STAGE 9: Code quality
        # ----------------------------------------------------------------
        progress_callback("[9/10] Evaluating code quality...", "Static analysis of Python source")

        quality_analyzer = CodeQualityAnalyzer()
        quality_result = quality_analyzer.analyze(project_root)
        report.code_quality = quality_result

        # Optional AI interpretation of quality
        llm = get_llm_service()

        code_snippet = _get_entry_point_snippet(project_root, spec)

        ai_quality = llm.interpret_code_quality(
            project_name=submission.project_name,
            static_findings=quality_result.static_findings,
            security_findings=quality_result.security_findings,
            complexity_findings=quality_result.complexity_findings,
            unused_code_findings=quality_result.unused_code_findings,
            code_snippet=code_snippet,
        )

        if ai_quality:
            quality_result.ai_findings.append(ai_quality)

        # ----------------------------------------------------------------
        # STAGE 10: Score + report
        # ----------------------------------------------------------------
        progress_callback("[10/10] Generating report...", "Calculating scores and writing report")

        # Assemble strengths / problems / recommendations
        _populate_narrative(report, llm)

        # Scoring
        scorer = ScoringEngine()
        report.scores = scorer.score(report)
        report.status = EvaluationStatus.SUCCESS

        # Write reports
        generator = ReportGenerator(results_root=RESULTS_DIR)
        report_paths = generator.generate(report)

        progress_callback(
            "[DONE] Evaluation complete!",
            f"Score: {report.scores.total:.0f}/100",
        )

        logger.info(
            "Evaluation %s complete. Score: %s/100",
            eval_id,
            report.scores.total,
        )

    except Exception as exc:
        logger.error("Unexpected evaluation error: %s", exc, exc_info=True)
        report.status = EvaluationStatus.FAILED
        report.problems.append(f"Unexpected evaluator error: {exc}")
        report.limitations.append(
            "An unexpected error interrupted the evaluation."
        )

    finally:
        # Clean up workspace
        if workspace and workspace.exists():
            try:
                SubmissionHandler.cleanup_workspace(workspace)
            except Exception:
                pass

    return report


# ===========================================================================
# Helper functions
# ===========================================================================


def _resolve_project_root(workspace: Path) -> Path:
    """
    If the workspace contains exactly one top-level directory,
    treat that as the project root (common for GitHub clones / ZIPs).
    """
    entries = [e for e in workspace.iterdir() if not e.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return workspace


def _find_readme(project_root: Path) -> Path:
    for name in ReadmeParser.README_NAMES:
        candidate = project_root / name
        if candidate.is_file():
            return candidate
    return project_root / "README.md"  # may not exist – parser handles it


def _resolve_run_command(
    spec: ProjectSpecification,
    repo: "RepositoryInspectionResult",
) -> Optional[str]:
    """Determine the best run command from available information."""

    if spec.run_command:
        cmd = spec.run_command.strip()
        # Strip markdown code-fence artefacts
        if cmd.startswith("`") and cmd.endswith("`"):
            cmd = cmd[1:-1]
        return cmd

    # Fall back to entry point candidates
    if repo.entry_point_candidates:
        return f"python {repo.entry_point_candidates[0]}"

    if spec.entry_point:
        return f"python {spec.entry_point}"

    return None


def _check_resource_safety(repo) -> list[str]:
    """Return warnings that suggest the project may be too heavy to execute."""
    issues = []

    for file_info in repo.large_files:
        if file_info["size_mb"] > 200:
            issues.append(
                f"Very large file detected ({file_info['path']}: "
                f"{file_info['size_mb']} MB). Execution skipped."
            )

    heavy_imports = {
        "torch", "tensorflow", "jax", "keras",
    }
    repo_imports_lower = {imp.lower() for imp in repo.detected_imports}
    heavy_found = heavy_imports & repo_imports_lower

    if heavy_found and repo.model_files:
        issues.append(
            "Heavy ML framework detected with local model files. "
            "Execution may require GPU or large downloads."
        )

    return issues


def _build_safe_environment() -> dict[str, str]:
    """Build an environment dict that does NOT expose evaluator secrets."""
    safe_env = {}
    # Allow basic system vars
    for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE"):
        val = os.environ.get(key)
        if val:
            safe_env[key] = val

    # Explicitly do NOT forward GEMINI_API_KEY or any API keys
    return safe_env


def _build_test_cases(spec: ProjectSpecification) -> list[TestCase]:
    """Convert sample_test_cases from spec into TestCase objects."""
    cases: list[TestCase] = []

    for i, sample in enumerate(spec.sample_test_cases):
        cases.append(
            TestCase(
                test_id=f"test_{i+1}",
                input_data=sample.get("input"),
                expected_output=sample.get("expected_output"),
                source=sample.get("source", "student_provided"),
                description=f"Test case {i+1}",
            )
        )

    return cases


def _test_summary_text(functional_result) -> str:
    if functional_result is None:
        return "No tests"
    if functional_result.total_tests == 0:
        return "No test cases available"
    return (
        f"{functional_result.passed_tests}/{functional_result.total_tests} passed"
    )


def _get_entry_point_snippet(
    project_root: Path,
    spec: ProjectSpecification,
) -> Optional[str]:
    """Read first 1500 chars of the entry point file for AI analysis."""
    if spec.entry_point:
        ep = project_root / spec.entry_point
        if ep.is_file():
            try:
                return ep.read_text(encoding="utf-8", errors="replace")[:1500]
            except OSError:
                pass
    return None


def _populate_narrative(report: EvaluationReport, llm) -> None:
    """
    Build strengths, problems, recommendations, and limitations lists
    from evaluation evidence.  Use Gemini for the summary if available.
    """

    # Deterministic problems
    if report.execution and report.execution.status != EvaluationStatus.SUCCESS:
        if report.execution.timed_out:
            report.problems.append("Project execution timed out.")
        elif report.execution.error_type:
            report.problems.append(
                f"Execution failed: {report.execution.error_type} – "
                f"{report.execution.error_message or 'no detail'}"
            )

    if report.dependency_check:
        for missing in report.dependency_check.missing_dependencies[:3]:
            report.problems.append(
                f"Dependency possibly missing from requirements: {missing}"
            )

    if report.pipeline_analysis:
        for issue in report.pipeline_analysis.consistency_issues[:3]:
            report.problems.append(issue)

    if report.code_quality:
        for finding in report.code_quality.security_findings[:2]:
            report.problems.append(finding)

    # Deterministic strengths
    if report.execution and report.execution.status == EvaluationStatus.SUCCESS:
        report.strengths.append("Project executed successfully end-to-end.")

    if (
        report.functional_testing
        and report.functional_testing.passed_tests > 0
    ):
        report.strengths.append(
            f"{report.functional_testing.passed_tests} functional test(s) passed."
        )

    if (
        report.pipeline_analysis
        and report.pipeline_analysis.model_stage_found
        and report.pipeline_analysis.input_stage_found
        and report.pipeline_analysis.output_stage_found
    ):
        report.strengths.append("Full AI pipeline structure detected (input → model → output).")

    if report.dependency_check and report.dependency_check.requirements_file:
        report.strengths.append("Requirements file present.")

    if report.specification and report.specification.run_command:
        report.strengths.append("Run command is documented.")

    # Limitations
    if not report.execution or report.execution.status != EvaluationStatus.SUCCESS:
        report.limitations.append(
            "Runtime performance could not be measured because execution did not succeed."
        )

    if not report.functional_testing or report.functional_testing.total_tests == 0:
        report.limitations.append(
            "No functional tests were available for deterministic pass/fail evaluation."
        )

    # Gemini-assisted recommendations
    if llm.is_available and report.submission:
        scores_dict = {
            "Functionality": report.scores.functionality,
            "Pipeline": report.scores.pipeline_correctness,
            "Performance": report.scores.performance_efficiency,
            "Code Quality": report.scores.code_quality,
            "Error Handling": report.scores.error_handling,
        }

        test_summary = _test_summary_text(report.functional_testing)

        ai_narrative = llm.generate_strengths_and_recommendations(
            project_name=report.submission.project_name,
            project_goal=report.submission.project_goal,
            scores=scores_dict,
            static_findings=(
                report.code_quality.static_findings[:4]
                if report.code_quality
                else []
            ),
            execution_success=(
                report.execution is not None
                and report.execution.status == EvaluationStatus.SUCCESS
            ),
            test_summary=test_summary,
        )

        if ai_narrative:
            for s in ai_narrative.get("strengths", []):
                if s not in report.strengths:
                    report.strengths.append(f"{s} [AI-ASSISTED]")
            for r in ai_narrative.get("recommendations", []):
                report.recommendations.append(f"{r} [AI-ASSISTED]")

    # Fallback deterministic recommendations
    if not report.recommendations:
        if report.code_quality and report.code_quality.security_findings:
            report.recommendations.append(
                "Remove hard-coded credentials and use environment variables."
            )
        if (
            report.functional_testing
            and report.functional_testing.failed_tests > 0
        ):
            report.recommendations.append(
                "Review failing test cases and fix output formatting."
            )
        if report.specification and not report.specification.sample_test_cases:
            report.recommendations.append(
                "Add sample input/output examples to the README for better testability."
            )


# ===========================================================================
# Gradio UI
# ===========================================================================


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio application."""

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    CUSTOM_CSS = """
    .main-header { text-align: center; padding: 20px 0 10px; }
    .main-header h1 { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .main-header p { color: #666; margin-top: 6px; }
    .section-title { font-size: 1rem; font-weight: 600; color: #1a1a2e;
                     border-bottom: 2px solid #e9ecef; padding-bottom: 6px;
                     margin-bottom: 8px; }
    .required-star { color: #dc3545; }
    .score-box { font-size: 3rem; font-weight: 800; text-align: center; }
    .stage-log { font-family: monospace; font-size: 0.85rem; }
    footer, .gradio-container footer, footer.svelte-12z0ox2 {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    """

    with gr.Blocks(
        theme=theme,
        title="NxtWave AI Project Evaluator",
        css=CUSTOM_CSS,
    ) as app:

        gr.HTML("""
        <style>
        footer, .gradio-container footer, [data-testid="footer"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        </style>
        <div class="main-header">
          <h1>🤖 AI Project Evaluator</h1>
          <p>Automated evaluation for lightweight Python AI/ML projects</p>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ----------------------------------------------------------
            # TAB 1: Submission form
            # ----------------------------------------------------------
            with gr.Tab("📝 Submit Project", id="submit_tab"):

                gr.HTML('<div class="section-title">Required Information <span class="required-star">*</span></div>')

                with gr.Row():
                    project_name = gr.Textbox(
                        label="Project Name *",
                        placeholder="e.g. Sentiment Analyser",
                        max_lines=1,
                    )

                with gr.Row():
                    github_url = gr.Textbox(
                        label="GitHub Repository URL",
                        placeholder="https://github.com/username/repo",
                        max_lines=1,
                    )
                    zip_upload = gr.File(
                        label="OR Upload ZIP",
                        file_types=[".zip"],
                    )

                gr.HTML('<p style="color:#888;font-size:.85rem;margin:-6px 0 10px">Provide either a GitHub URL <em>or</em> a ZIP file — not both.</p>')

                with gr.Row():
                    project_goal = gr.Textbox(
                        label="Project Goal *",
                        placeholder="e.g. Classify customer reviews as positive or negative",
                        lines=2,
                    )

                with gr.Row():
                    run_command = gr.Textbox(
                        label="How to Run *",
                        placeholder="e.g. python main.py",
                        max_lines=1,
                    )
                    input_type = gr.Textbox(
                        label="Input Type *",
                        placeholder="e.g. text string via stdin",
                        max_lines=1,
                    )

                expected_behavior = gr.Textbox(
                    label="Expected Output / Behavior *",
                    placeholder="e.g. Prints 'Positive' or 'Negative' label with confidence score",
                    lines=2,
                )

                gr.HTML('<div class="section-title" style="margin-top:16px">Optional Information</div>')

                with gr.Row():
                    sample_input = gr.Textbox(
                        label="Sample Input",
                        placeholder="e.g. I love this product!",
                        lines=2,
                    )
                    sample_output = gr.Textbox(
                        label="Sample Expected Output",
                        placeholder="e.g. Positive (0.92)",
                        lines=2,
                    )

                with gr.Row():
                    python_version = gr.Textbox(
                        label="Python Version",
                        placeholder="e.g. 3.10",
                        max_lines=1,
                    )
                    external_api = gr.Textbox(
                        label="External API / Service",
                        placeholder="e.g. OpenAI API (key required)",
                        max_lines=1,
                    )

                with gr.Row():
                    dataset_info = gr.Textbox(
                        label="Dataset / Model Information",
                        placeholder="e.g. Uses distilbert-base-uncased from HuggingFace",
                        lines=2,
                    )
                    additional_instructions = gr.Textbox(
                        label="Additional Run Instructions",
                        placeholder="e.g. Set OPENAI_API_KEY before running",
                        lines=2,
                    )

                evaluate_btn = gr.Button(
                    "🚀 EVALUATE PROJECT",
                    variant="primary",
                    size="lg",
                )

                validation_error = gr.Markdown(visible=False)

            # ----------------------------------------------------------
            # TAB 2: Evaluation progress
            # ----------------------------------------------------------
            with gr.Tab("⏳ Progress", id="progress_tab", visible=False) as progress_tab:

                gr.HTML('<div class="section-title">Evaluation Progress</div>')

                progress_status = gr.Textbox(
                    label="Current Stage",
                    interactive=False,
                    max_lines=1,
                )

                progress_log = gr.Textbox(
                    label="Progress Log",
                    interactive=False,
                    lines=12,
                    elem_classes=["stage-log"],
                )

            # ----------------------------------------------------------
            # TAB 3: Results
            # ----------------------------------------------------------
            with gr.Tab("📊 Results", id="results_tab", visible=False) as results_tab:

                result_score_html = gr.HTML()
                result_detail_html = gr.HTML()
                result_report_link = gr.Markdown()

        # ------------------------------------------------------------------
        # Button click handler
        # ------------------------------------------------------------------

        def on_evaluate(
            p_name, p_github_url, p_zip, p_goal, p_run,
            p_input_type, p_expected, p_sample_in, p_sample_out,
            p_py_version, p_ext_api, p_dataset, p_additional,
            progress=gr.Progress(track_tqdm=False),
        ):
            """
            Validate inputs, run evaluation, stream progress, then show results.
            """

            # Validate
            errors = []

            if not p_name or not p_name.strip():
                errors.append("Project Name is required.")
            if not p_goal or not p_goal.strip():
                errors.append("Project Goal is required.")
            if not p_run or not p_run.strip():
                errors.append("How to Run is required.")
            if not p_input_type or not p_input_type.strip():
                errors.append("Input Type is required.")
            if not p_expected or not p_expected.strip():
                errors.append("Expected Output / Behavior is required.")
            if not p_github_url and not p_zip:
                errors.append("Provide a GitHub URL or upload a ZIP file.")
            if p_github_url and p_zip:
                errors.append("Provide either a GitHub URL or a ZIP — not both.")

            if errors:
                error_md = "**⚠️ Please fix these errors:**\n" + "\n".join(
                    f"- {e}" for e in errors
                )
                yield (
                    gr.update(value=error_md, visible=True),  # validation_error
                    gr.update(visible=False),  # progress_tab
                    gr.update(visible=False),  # results_tab
                    gr.update(value=""),       # progress_status
                    gr.update(value=""),       # progress_log
                    gr.update(value=""),       # result_score_html
                    gr.update(value=""),       # result_detail_html
                    gr.update(value=""),       # result_report_link
                )
                return

            # Build submission
            zip_path = Path(p_zip.name) if p_zip else None

            submission = StudentSubmission(
                project_name=p_name.strip(),
                project_goal=p_goal.strip(),
                run_command=p_run.strip(),
                input_type=p_input_type.strip(),
                expected_behavior=p_expected.strip(),
                github_url=p_github_url.strip() if p_github_url else None,
                zip_path=zip_path,
                sample_input=p_sample_in or None,
                sample_expected_output=p_sample_out or None,
                python_version=p_py_version or None,
                external_api_service=p_ext_api or None,
                dataset_model_info=p_dataset or None,
                additional_instructions=p_additional or None,
            )

            # Set submission type
            if submission.github_url:
                submission.submission_type = SubmissionType.GITHUB
            else:
                submission.submission_type = SubmissionType.ZIP

            # Show progress tab
            log_lines: list[str] = []

            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value="Starting evaluation…"),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
            )

            # Progress callback
            def progress_cb(stage: str, detail: str) -> None:
                stamp = datetime.datetime.now().strftime("%H:%M:%S")
                line = f"[{stamp}] {stage}"
                if detail:
                    line += f"\n          {detail}"
                log_lines.append(line)

            # We can't yield from inside a callback, so we'll run synchronously
            # and update the log at the end. For Gradio streaming we use a
            # wrapper that yields partial updates.

            # Run evaluation
            report = run_evaluation(submission, progress_cb)

            # Build result HTML
            score_html = _build_score_html(report)
            detail_html = _build_detail_html(report)
            log_text = "\n\n".join(log_lines)

            # Find report path
            report_md = ""
            try:
                gen = ReportGenerator(results_root=RESULTS_DIR)
                paths = gen.generate(report)
                if "html" in paths:
                    report_md = f"📄 [View Full HTML Report]({paths['html'].as_uri()})"
            except Exception:
                pass

            yield (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(value=f"✅ Done – Score: {report.scores.total:.0f}/100"),
                gr.update(value=log_text),
                gr.update(value=score_html),
                gr.update(value=detail_html),
                gr.update(value=report_md),
            )

        evaluate_btn.click(
            fn=on_evaluate,
            inputs=[
                project_name, github_url, zip_upload,
                project_goal, run_command, input_type, expected_behavior,
                sample_input, sample_output,
                python_version, external_api, dataset_info, additional_instructions,
            ],
            outputs=[
                validation_error,
                progress_tab,
                results_tab,
                progress_status,
                progress_log,
                result_score_html,
                result_detail_html,
                result_report_link,
            ],
        )

    return app


# ===========================================================================
# Result HTML builders
# ===========================================================================


def _build_score_html(report: EvaluationReport) -> str:
    scores = report.scores
    total = scores.total

    colour = (
        "#dc3545" if total < 40 else
        "#fd7e14" if total < 60 else
        "#ffc107" if total < 75 else
        "#28a745"
    )

    exec_badge = ""
    if report.execution:
        if report.execution.status == EvaluationStatus.SUCCESS:
            exec_badge = '<span style="background:#d4edda;color:#155724;padding:4px 12px;border-radius:12px;font-weight:600;">✅ Execution: SUCCESS</span>'
        elif report.execution.timed_out:
            exec_badge = '<span style="background:#fff3cd;color:#856404;padding:4px 12px;border-radius:12px;font-weight:600;">⏱ Execution: TIMEOUT</span>'
        else:
            exec_badge = '<span style="background:#f8d7da;color:#721c24;padding:4px 12px;border-radius:12px;font-weight:600;">❌ Execution: FAILED</span>'

    test_text = ""
    if report.functional_testing and report.functional_testing.total_tests > 0:
        ft = report.functional_testing
        test_text = f"&nbsp;|&nbsp; 🧪 Tests: {ft.passed_tests}/{ft.total_tests} passed"

    time_text = ""
    if report.execution and report.execution.execution_time_seconds is not None:
        time_text = f"&nbsp;|&nbsp; ⏱ {report.execution.execution_time_seconds:.2f}s"

    mem_text = ""
    if report.execution and report.execution.peak_memory_mb is not None:
        mem_text = f"&nbsp;|&nbsp; 💾 {report.execution.peak_memory_mb:.1f} MB"

    categories = [
        ("Functionality & Tests", scores.functionality, 30),
        ("Pipeline Correctness", scores.pipeline_correctness, 20),
        ("Performance & Efficiency", scores.performance_efficiency, 15),
        ("Code Quality", scores.code_quality, 15),
        ("Error Handling", scores.error_handling, 5),
        ("Dependencies", scores.dependencies_environment, 5),
        ("Documentation", scores.documentation, 5),
        ("AI Completeness", scores.ai_project_completeness, 5),
    ]

    rows = ""
    for label, val, maxv in categories:
        pct = min(100, (val / maxv) * 100) if maxv else 0
        rows += f"""
        <tr>
          <td style="padding:6px 12px;color:#555;width:220px">{label}</td>
          <td style="padding:6px 12px;">
            <div style="background:#e9ecef;border-radius:4px;height:10px;width:100%">
              <div style="background:{colour};width:{pct:.0f}%;height:10px;border-radius:4px"></div>
            </div>
          </td>
          <td style="padding:6px 12px;font-weight:600;text-align:right;white-space:nowrap">
            {val:.1f} / {maxv}
          </td>
        </tr>"""

    return f"""
    <div style="background:white;border-radius:12px;padding:24px;
                box-shadow:0 2px 12px rgba(0,0,0,.08);margin-bottom:16px">
      <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap">
        <div style="width:110px;height:110px;border-radius:50%;
                    background:{colour};display:flex;flex-direction:column;
                    align-items:center;justify-content:center;color:white;flex-shrink:0">
          <span style="font-size:2.2rem;font-weight:800;line-height:1">{int(total)}</span>
          <span style="font-size:.8rem;opacity:.85">/ 100</span>
        </div>
        <div style="flex:1">
          <h2 style="font-size:1.3rem;margin-bottom:6px">
            {_esc_html(report.submission.project_name if report.submission else 'Project')}
          </h2>
          <div style="margin-bottom:12px;font-size:.85rem;color:#666">
            {exec_badge}{test_text}{time_text}{mem_text}
          </div>
          <table style="width:100%;border-collapse:collapse">{rows}</table>
        </div>
      </div>
    </div>
    """


def _build_detail_html(report: EvaluationReport) -> str:
    sections = []

    # Execution error detail
    if report.execution and report.execution.status != EvaluationStatus.SUCCESS:
        stderr_block = ""
        if report.execution.stderr:
            excerpt = report.execution.stderr[-1500:]
            stderr_block = f'<pre style="background:#1e1e2e;color:#cdd6f4;padding:12px;border-radius:8px;overflow:auto;max-height:200px;font-size:.8rem">{_esc_html(excerpt)}</pre>'

        sections.append(f"""
        <div style="background:white;border-radius:12px;padding:20px;
                    margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.06)">
          <h3 style="color:#dc3545;margin-bottom:10px">❌ Execution Error</h3>
          <p><strong>Type:</strong> {_esc_html(report.execution.error_type or 'Unknown')}</p>
          <p><strong>Message:</strong> {_esc_html(report.execution.error_message or 'No message')}</p>
          {stderr_block}
        </div>""")

    # Strengths & Problems
    strengths = "".join(f'<li style="padding:3px 0">✅ {_esc_html(s)}</li>' for s in report.strengths) or "<li>None identified</li>"
    problems = "".join(f'<li style="padding:3px 0">⚠️ {_esc_html(p)}</li>' for p in report.problems) or "<li>None identified</li>"
    recs = "".join(f'<li style="padding:3px 0">→ {_esc_html(r)}</li>' for r in report.recommendations) or "<li>No recommendations</li>"

    sections.append(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
      <div style="background:white;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)">
        <h3 style="margin-bottom:10px">💪 Strengths</h3>
        <ul style="list-style:none;padding:0;font-size:.9rem">{strengths}</ul>
      </div>
      <div style="background:white;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)">
        <h3 style="margin-bottom:10px">🔴 Issues</h3>
        <ul style="list-style:none;padding:0;font-size:.9rem">{problems}</ul>
      </div>
    </div>
    <div style="background:white;border-radius:12px;padding:20px;
                margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.06)">
      <h3 style="margin-bottom:10px">💡 Recommendations</h3>
      <ul style="list-style:none;padding:0;font-size:.9rem">{recs}</ul>
    </div>""")

    # Tests
    if report.functional_testing and report.functional_testing.test_results:
        test_rows = ""
        for i, tr in enumerate(report.functional_testing.test_results, 1):
            colour_map = {
                "pass": "#28a745", "fail": "#dc3545",
                "error": "#fd7e14", "timeout": "#6c757d", "not_testable": "#adb5bd",
            }
            status_colour = colour_map.get(tr.status.value, "#666")
            test_rows += f"""
            <tr>
              <td style="padding:7px 10px">{i}</td>
              <td style="padding:7px 10px"><code style="font-size:.8rem">{_esc_html(str(tr.actual_output or '')[:80])}</code></td>
              <td style="padding:7px 10px"><code style="font-size:.8rem">{_esc_html(str(tr.expected_output or 'N/A')[:80])}</code></td>
              <td style="padding:7px 10px;color:{status_colour};font-weight:600">{tr.status.value.upper()}</td>
              <td style="padding:7px 10px;font-size:.8rem;color:#888">{_esc_html(tr.source)}</td>
            </tr>"""

        sections.append(f"""
        <div style="background:white;border-radius:12px;padding:20px;
                    margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.06)">
          <h3 style="margin-bottom:12px">🧪 Functional Tests</h3>
          <table style="width:100%;border-collapse:collapse;font-size:.88rem">
            <thead>
              <tr style="background:#f8f9fa">
                <th style="padding:7px 10px;text-align:left">#</th>
                <th style="padding:7px 10px;text-align:left">Actual Output</th>
                <th style="padding:7px 10px;text-align:left">Expected</th>
                <th style="padding:7px 10px;text-align:left">Status</th>
                <th style="padding:7px 10px;text-align:left">Source</th>
              </tr>
            </thead>
            <tbody>{test_rows}</tbody>
          </table>
        </div>""")

    # Limitations
    if report.limitations:
        lim_items = "".join(f'<li style="padding:3px 0">ℹ️ {_esc_html(l)}</li>' for l in report.limitations)
        sections.append(f"""
        <div style="background:#f8f9fa;border-radius:12px;padding:20px;
                    margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.06)">
          <h3 style="margin-bottom:10px;color:#6c757d">📌 Evaluation Limitations</h3>
          <ul style="list-style:none;padding:0;font-size:.9rem;color:#666">{lim_items}</ul>
        </div>""")

    return "\n".join(sections)


def _esc_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ===========================================================================
# Entry point
# ===========================================================================


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )
