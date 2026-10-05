"""
Smoke test – runs a complete evaluation against the local demo project
without starting Gradio.

Run with:
    python tests/smoke_test.py
"""

import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from NxtWave_AI_Project_Evaluator.app.main import run_evaluation
from NxtWave_AI_Project_Evaluator.app.models import StudentSubmission, SubmissionType


def main():
    demo_path = (
        Path(__file__).parent.parent
        / "NxtWave_AI_Project_Evaluator"
        / "demo"
        / "sample_ai_project"
    )

    # Package the demo dir as a ZIP temporarily for the ZIP path
    import shutil, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "demo_project.zip"
        shutil.make_archive(
            str(zip_path.with_suffix("")),
            "zip",
            root_dir=str(demo_path.parent),
            base_dir=demo_path.name,
        )

        submission = StudentSubmission(
            project_name="Demo Sentiment Analyser",
            project_goal="Classify text input as positive, negative, or neutral",
            run_command="python main.py",
            input_type="text string via stdin",
            expected_behavior="Prints one of: positive, negative, neutral",
            zip_path=zip_path,
            sample_input="hello",
            sample_expected_output="positive",
            submission_type=SubmissionType.ZIP,
        )

        log_lines = []

        def progress(stage, detail):
            print(f"  {stage}")
            if detail:
                print(f"    → {detail}")
            log_lines.append(stage)

        print("\n=== SMOKE TEST: NxtWave AI Project Evaluator ===\n")

        report = run_evaluation(submission, progress)

        print(f"\n{'='*50}")
        print(f"  SCORE: {report.scores.total:.1f} / 100")
        print(f"{'='*50}")
        print(f"  Functionality:     {report.scores.functionality:.1f} / 30")
        print(f"  Pipeline:          {report.scores.pipeline_correctness:.1f} / 20")
        print(f"  Performance:       {report.scores.performance_efficiency:.1f} / 15")
        print(f"  Code Quality:      {report.scores.code_quality:.1f} / 15")
        print(f"  Error Handling:    {report.scores.error_handling:.1f} / 5")
        print(f"  Dependencies:      {report.scores.dependencies_environment:.1f} / 5")
        print(f"  Documentation:     {report.scores.documentation:.1f} / 5")
        print(f"  AI Completeness:   {report.scores.ai_project_completeness:.1f} / 5")

        print(f"\n  Execution: {report.execution.status.value if report.execution else 'N/A'}")

        if report.functional_testing:
            ft = report.functional_testing
            print(f"  Tests: {ft.passed_tests}/{ft.total_tests} passed")

        print("\n  Strengths:")
        for s in report.strengths:
            print(f"    ✓ {s}")

        print("\n  Problems:")
        for p in report.problems:
            print(f"    ⚠ {p}")

        print("\n  Recommendations:")
        for r in report.recommendations:
            print(f"    → {r}")

        print("\n  Limitations:")
        for l in report.limitations:
            print(f"    ℹ {l}")

        print(f"\n  Status: {report.status.value}")
        print("\n=== SMOKE TEST COMPLETE ===\n")

        # Basic assertions
        assert report.scores.total >= 0, "Score must be non-negative"
        assert report.scores.total <= 100, "Score must not exceed 100"
        assert report.specification is not None, "Specification must be set"
        assert report.repository_inspection is not None, "Repository inspection must run"
        print("✅ All assertions passed.")


if __name__ == "__main__":
    main()
