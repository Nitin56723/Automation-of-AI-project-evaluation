"""
scoring_engine.py
-----------------
Converts multi-stage evaluation evidence into a deterministic /100 score.

Scoring categories:
    Functionality & Test Results        30
    Pipeline Correctness                20
    Performance & Efficiency            15
    Code Quality                        15
    Error Handling                       5
    Dependencies / Environment           5
    Documentation                        5
    AI Project Completeness              5
    ─────────────────────────────────────
    TOTAL                              100

Rules:
- The LLM does NOT generate the total score.
- Objective evidence drives the primary score.
- Execution failure does NOT zero the whole evaluation.
- Each category is capped at its maximum.
"""

from __future__ import annotations

import logging
from typing import Optional

from .models import (
    CodeQualityResult,
    DependencyCheckResult,
    EvaluationReport,
    EvaluationScores,
    EvaluationStatus,
    ExecutionResult,
    FunctionalTestResult,
    PerformanceResult,
    PipelineAnalysisResult,
    ProjectSpecification,
    RepositoryInspectionResult,
    TestStatus,
)

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Combines evaluation evidence into a deterministic final score.
    """

    # Category maximums
    MAX_FUNCTIONALITY = 30.0
    MAX_PIPELINE = 20.0
    MAX_PERFORMANCE = 15.0
    MAX_CODE_QUALITY = 15.0
    MAX_ERROR_HANDLING = 5.0
    MAX_DEPENDENCIES = 5.0
    MAX_DOCUMENTATION = 5.0
    MAX_AI_COMPLETENESS = 5.0

    def score(self, report: EvaluationReport) -> EvaluationScores:
        """
        Calculate scores from all available evaluation evidence.

        Missing stages reduce applicable scores; they do not zero the total.
        """

        scores = EvaluationScores()

        scores.functionality = self._score_functionality(
            report.functional_testing,
            report.execution,
        )

        scores.pipeline_correctness = self._score_pipeline(
            report.pipeline_analysis,
            report.execution,
        )

        scores.performance_efficiency = self._score_performance(
            report.performance,
            report.execution,
        )

        scores.code_quality = self._score_code_quality(
            report.code_quality,
        )

        scores.error_handling = self._score_error_handling(
            report.code_quality,
            report.execution,
        )

        scores.dependencies_environment = self._score_dependencies(
            report.dependency_check,
        )

        scores.documentation = self._score_documentation(
            report.specification,
        )

        scores.ai_project_completeness = self._score_ai_completeness(
            report.repository_inspection,
            report.pipeline_analysis,
        )

        return scores

    # ------------------------------------------------------------------
    # Functionality  (max 30)
    # ------------------------------------------------------------------

    def _score_functionality(
        self,
        testing: Optional[FunctionalTestResult],
        execution: Optional[ExecutionResult],
    ) -> float:

        score = 0.0

        # Execution (10 pts)
        if execution is not None:
            if execution.status == EvaluationStatus.SUCCESS:
                score += 10.0
            elif execution.timed_out:
                score += 2.0  # reached execution, just timed out
            elif execution.status == EvaluationStatus.FAILED:
                if execution.dependency_error:
                    score += 0.0  # deps block everything
                else:
                    score += 3.0  # ran but crashed

        # Tests (20 pts)
        if (
            testing is not None
            and testing.total_tests > 0
        ):
            total = testing.total_tests
            passed = testing.passed_tests
            not_testable = testing.not_testable_tests

            # Testable subset
            testable = total - not_testable

            if testable > 0:
                pass_rate = passed / testable
                score += pass_rate * 20.0
            else:
                # All tests not testable – partial credit if execution worked
                if (
                    execution is not None
                    and execution.status == EvaluationStatus.SUCCESS
                ):
                    score += 5.0

        elif (
            execution is not None
            and execution.status == EvaluationStatus.SUCCESS
            and testing is not None
            and testing.total_tests == 0
        ):
            # Executed successfully but no tests available
            score += 8.0

        return _clamp(score, 0.0, self.MAX_FUNCTIONALITY)

    # ------------------------------------------------------------------
    # Pipeline Correctness  (max 20)
    # ------------------------------------------------------------------

    def _score_pipeline(
        self,
        pipeline: Optional[PipelineAnalysisResult],
        execution: Optional[ExecutionResult],
    ) -> float:

        score = 0.0

        if pipeline is None:
            return score

        # Static structure (12 pts across 4 stages)
        stage_score = 0.0

        if pipeline.input_stage_found:
            stage_score += 3.0

        if pipeline.preprocessing_stage_found:
            stage_score += 3.0

        if pipeline.model_stage_found:
            stage_score += 3.0

        if pipeline.output_stage_found:
            stage_score += 3.0

        score += stage_score

        # Entry point found (2 pts)
        if pipeline.entry_point:
            score += 2.0

        # Consistency (up to 6 pts)
        issues = len(pipeline.consistency_issues)

        if issues == 0:
            consistency_bonus = 6.0
        elif issues == 1:
            consistency_bonus = 4.0
        elif issues == 2:
            consistency_bonus = 2.0
        else:
            consistency_bonus = 0.0

        score += consistency_bonus

        # Execution confirms pipeline ran (bonus within cap)
        if (
            execution is not None
            and execution.status == EvaluationStatus.SUCCESS
        ):
            score = min(score + 2.0, self.MAX_PIPELINE)

        return _clamp(score, 0.0, self.MAX_PIPELINE)

    # ------------------------------------------------------------------
    # Performance & Efficiency  (max 15)
    # ------------------------------------------------------------------

    def _score_performance(
        self,
        performance: Optional[PerformanceResult],
        execution: Optional[ExecutionResult],
    ) -> float:

        score = 0.0

        if performance is None:
            return score

        if not performance.measured:
            # Static analysis only – partial credit
            if performance.efficiency_findings:
                score = 3.0  # found something but not measured
            else:
                score = 5.0  # no issues detected statically
            return _clamp(score, 0.0, self.MAX_PERFORMANCE)

        # Runtime available
        exec_time = performance.total_execution_time_seconds
        memory_mb = performance.peak_memory_mb

        # Time score (0-7)
        if exec_time is not None:
            if exec_time <= 2.0:
                score += 7.0
            elif exec_time <= 5.0:
                score += 5.5
            elif exec_time <= 10.0:
                score += 4.0
            elif exec_time <= 20.0:
                score += 2.5
            else:
                score += 1.0

        # Memory score (0-5)
        if memory_mb is not None:
            if memory_mb <= 100:
                score += 5.0
            elif memory_mb <= 300:
                score += 4.0
            elif memory_mb <= 500:
                score += 3.0
            elif memory_mb <= 1000:
                score += 1.5
            else:
                score += 0.0

        # Static efficiency deductions
        penalty = len(performance.efficiency_findings) * 0.5
        score -= min(penalty, 3.0)

        return _clamp(score, 0.0, self.MAX_PERFORMANCE)

    # ------------------------------------------------------------------
    # Code Quality  (max 15)
    # ------------------------------------------------------------------

    def _score_code_quality(
        self,
        quality: Optional[CodeQualityResult],
    ) -> float:

        score = 0.0

        if quality is None or quality.status == EvaluationStatus.FAILED:
            return score

        # Use sub-scores from the analyzer (each out of 15, averaged)
        sub_scores = []

        for attr in (
            "readability_score",
            "maintainability_score",
            "complexity_score",
        ):
            val = getattr(quality, attr, None)

            if val is not None:
                # Normalise each sub-score to the 15-point scale
                sub_scores.append(val)

        if sub_scores:
            avg = sum(sub_scores) / len(sub_scores)
            score = (avg / 15.0) * self.MAX_CODE_QUALITY
        else:
            # Fallback: count findings
            total_findings = (
                len(quality.static_findings)
                + len(quality.security_findings)
                + len(quality.complexity_findings)
                + len(quality.unused_code_findings)
            )
            score = max(0.0, 15.0 - total_findings * 1.5)

        return _clamp(score, 0.0, self.MAX_CODE_QUALITY)

    # ------------------------------------------------------------------
    # Error Handling  (max 5)
    # ------------------------------------------------------------------

    def _score_error_handling(
        self,
        quality: Optional[CodeQualityResult],
        execution: Optional[ExecutionResult],
    ) -> float:

        score = 0.0

        if quality is None:
            return score

        # Static score (up to 3 pts)
        eh_score = getattr(quality, "error_handling_score", None)

        if eh_score is not None:
            score += (eh_score / 15.0) * 3.0
        else:
            # Deduct for security issues
            score += max(
                0.0,
                3.0 - len(quality.security_findings) * 1.0,
            )

        # Runtime outcome (up to 2 pts)
        if execution is not None:
            if execution.status == EvaluationStatus.SUCCESS:
                score += 2.0
            elif execution.status == EvaluationStatus.FAILED and not execution.timed_out:
                if execution.error_type not in ("dependency_error", "syntax_error"):
                    score += 0.5  # at least ran

        return _clamp(score, 0.0, self.MAX_ERROR_HANDLING)

    # ------------------------------------------------------------------
    # Dependencies / Environment  (max 5)
    # ------------------------------------------------------------------

    def _score_dependencies(
        self,
        dep_check: Optional[DependencyCheckResult],
    ) -> float:

        score = 0.0

        if dep_check is None:
            return score

        # Has a requirements file (1 pt)
        if dep_check.requirements_file:
            score += 1.0

        # Declared deps found (1 pt)
        if dep_check.declared_dependencies:
            score += 1.0

        # No install failures (2 pts)
        if dep_check.install_success:
            score += 2.0
        elif not dep_check.install_failures:
            score += 1.0  # not tested but no failures recorded

        # No missing deps (1 pt)
        if not dep_check.missing_dependencies:
            score += 1.0

        return _clamp(score, 0.0, self.MAX_DEPENDENCIES)

    # ------------------------------------------------------------------
    # Documentation  (max 5)
    # ------------------------------------------------------------------

    def _score_documentation(
        self,
        spec: Optional[ProjectSpecification],
    ) -> float:

        score = 0.0

        if spec is None:
            return score

        # Each explicitly-provided field = 1 pt (max 5)
        fields = [
            spec.project_goal,
            spec.run_command,
            spec.input_type,
            spec.expected_behavior,
            spec.dependencies,
        ]

        for field in fields:
            if field:
                score += 1.0

        return _clamp(score, 0.0, self.MAX_DOCUMENTATION)

    # ------------------------------------------------------------------
    # AI Project Completeness  (max 5)
    # ------------------------------------------------------------------

    def _score_ai_completeness(
        self,
        inspection: Optional[RepositoryInspectionResult],
        pipeline: Optional[PipelineAnalysisResult],
    ) -> float:

        score = 0.0

        AI_ML_IMPORTS = {
            "sklearn",
            "tensorflow",
            "keras",
            "torch",
            "transformers",
            "spacy",
            "nltk",
            "gensim",
            "xgboost",
            "lightgbm",
            "catboost",
            "fastai",
            "sentence_transformers",
            "openai",
            "anthropic",
            "google",
            "langchain",
            "huggingface_hub",
            "datasets",
            "diffusers",
            "cv2",
            "PIL",
            "imageio",
        }

        if inspection is not None:
            imports_lower = {
                imp.lower()
                for imp in inspection.detected_imports
            }

            ai_found = any(
                ai_lib in imports_lower
                for ai_lib in {lib.lower() for lib in AI_ML_IMPORTS}
            )

            if ai_found:
                score += 2.0

            if inspection.model_files:
                score += 1.0

        if pipeline is not None:
            if pipeline.model_stage_found:
                score += 1.5

            if (
                pipeline.input_stage_found
                and pipeline.model_stage_found
                and pipeline.output_stage_found
            ):
                score += 0.5

        return _clamp(score, 0.0, self.MAX_AI_COMPLETENESS)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, round(value, 2)))


def calculate_scores(report: EvaluationReport) -> EvaluationScores:
    """Convenience wrapper."""
    engine = ScoringEngine()
    return engine.score(report)
