"""
report_generator.py
-------------------
Generates machine-readable JSON and human-readable HTML reports
from a completed EvaluationReport.

Output files follow the naming convention:
    results/raw/    evaluation_<ID>.json
    results/scores/ evaluation_<ID>_score.json
    results/reports/evaluation_<ID>_report.html
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import EvaluationReport, EvaluationStatus, TestStatus

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Produces JSON and HTML reports from a completed EvaluationReport.
    """

    def __init__(self, results_root: Optional[Path] = None):
        """
        Parameters
        ----------
        results_root:
            Base directory for output files.
            Defaults to results/ relative to the current working directory.
        """

        self.results_root = (
            Path(results_root)
            if results_root
            else Path("results")
        )

        self.raw_dir = self.results_root / "raw"
        self.scores_dir = self.results_root / "scores"
        self.reports_dir = self.results_root / "reports"

        for directory in (self.raw_dir, self.scores_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        report: EvaluationReport,
    ) -> dict[str, Path]:
        """
        Generate all output files for the given report.

        Returns a dict mapping file type to output path:
            "json"  -> raw JSON path
            "score" -> score JSON path
            "html"  -> HTML report path
        """

        eval_id = report.evaluation_id or _generate_id()
        report.evaluation_id = eval_id

        paths: dict[str, Path] = {}

        try:
            json_path = self._write_json(report, eval_id)
            paths["json"] = json_path
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to write JSON report: %s", exc)

        try:
            score_path = self._write_score_json(report, eval_id)
            paths["score"] = score_path
        except Exception as exc:
            logger.error("Failed to write score JSON: %s", exc)

        try:
            html_path = self._write_html(report, eval_id)
            paths["html"] = html_path
        except Exception as exc:
            logger.error("Failed to write HTML report: %s", exc)

        return paths

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def _write_json(self, report: EvaluationReport, eval_id: str) -> Path:
        path = self.raw_dir / f"evaluation_{eval_id}.json"

        data = report.to_dict()
        data["generated_at"] = _utc_now_iso()

        path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )

        logger.info("JSON report written: %s", path)

        return path

    def _write_score_json(
        self, report: EvaluationReport, eval_id: str
    ) -> Path:
        path = self.scores_dir / f"evaluation_{eval_id}_score.json"

        scores = report.scores

        data = {
            "evaluation_id": eval_id,
            "generated_at": _utc_now_iso(),
            "project_name": (
                report.submission.project_name
                if report.submission
                else "Unknown"
            ),
            "total_score": scores.total,
            "categories": {
                "functionality": scores.functionality,
                "pipeline_correctness": scores.pipeline_correctness,
                "performance_efficiency": scores.performance_efficiency,
                "code_quality": scores.code_quality,
                "error_handling": scores.error_handling,
                "dependencies_environment": scores.dependencies_environment,
                "documentation": scores.documentation,
                "ai_project_completeness": scores.ai_project_completeness,
            },
        }

        path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

        logger.info("Score JSON written: %s", path)

        return path

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _write_html(self, report: EvaluationReport, eval_id: str) -> Path:
        path = self.reports_dir / f"evaluation_{eval_id}_report.html"

        html = self._build_html(report, eval_id)

        path.write_text(html, encoding="utf-8")

        logger.info("HTML report written: %s", path)

        return path

    def _build_html(self, report: EvaluationReport, eval_id: str) -> str:
        """Assemble the HTML report string."""

        scores = report.scores
        sub = report.submission
        spec = report.specification
        execution = report.execution
        testing = report.functional_testing
        performance = report.performance
        dependency = report.dependency_check
        pipeline = report.pipeline_analysis
        quality = report.code_quality

        project_name = (sub.project_name if sub else "Unknown Project")
        project_goal = (
            spec.project_goal if spec else (sub.project_goal if sub else "")
        )

        exec_status = "N/A"
        exec_class = "badge-secondary"

        if execution:
            if execution.status == EvaluationStatus.SUCCESS:
                exec_status = "SUCCESS"
                exec_class = "badge-success"
            elif execution.timed_out:
                exec_status = "TIMEOUT"
                exec_class = "badge-warning"
            else:
                exec_status = "FAILED"
                exec_class = "badge-danger"

        exec_time = "N/A"
        peak_mem = "N/A"

        if execution and execution.execution_time_seconds is not None:
            exec_time = f"{execution.execution_time_seconds:.2f}s"

        if execution and execution.peak_memory_mb is not None:
            peak_mem = f"{execution.peak_memory_mb:.1f} MB"

        test_summary = "No tests"

        if testing and testing.total_tests > 0:
            test_summary = (
                f"{testing.passed_tests} / {testing.total_tests} passed"
            )

        # Score bar colour
        total = scores.total
        bar_colour = "#dc3545" if total < 40 else (
            "#fd7e14" if total < 60 else (
                "#ffc107" if total < 75 else "#28a745"
            )
        )

        # Category rows
        category_rows = _category_rows(scores)

        # Test rows
        test_rows = _test_rows(testing)

        # Dependency info
        dep_info = _dependency_info(dependency)

        # Pipeline info
        pipeline_info = _pipeline_info(pipeline)

        # Quality findings
        quality_info = _quality_info(quality)

        # Strengths / problems / recommendations / limitations
        strengths_html = _list_html(report.strengths, "strength-item")
        problems_html = _list_html(report.problems, "problem-item")
        recs_html = _list_html(report.recommendations, "rec-item")
        limits_html = _list_html(report.limitations, "limit-item")

        error_section = ""
        if execution and execution.status != EvaluationStatus.SUCCESS:
            error_section = f"""
            <div class="section">
                <h2>Execution Error</h2>
                <p><strong>Type:</strong> {_esc(execution.error_type or 'Unknown')}</p>
                <p><strong>Message:</strong> {_esc(execution.error_message or 'No message')}</p>
                {f'<pre class="error-output">{_esc(execution.stderr[-2000:])}</pre>'
                 if execution.stderr else ''}
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Project Evaluation – {_esc(project_name)}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f7fa;
            color: #1a1a2e; line-height: 1.6; }}
    .container {{ max-width: 1000px; margin: 0 auto; padding: 24px; }}
    .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
               color: white; padding: 32px; border-radius: 12px; margin-bottom: 24px; }}
    .header h1 {{ font-size: 1.6rem; margin-bottom: 8px; }}
    .header .meta {{ font-size: 0.85rem; opacity: 0.7; }}
    .score-hero {{ display: flex; align-items: center; gap: 24px;
                  background: white; border-radius: 12px; padding: 24px;
                  margin-bottom: 24px; box-shadow: 0 2px 12px rgba(0,0,0,.08); }}
    .score-circle {{ width: 120px; height: 120px; border-radius: 50%;
                    display: flex; flex-direction: column; align-items: center;
                    justify-content: center; color: white; flex-shrink: 0;
                    background: {bar_colour}; }}
    .score-circle .num {{ font-size: 2.2rem; font-weight: 700; }}
    .score-circle .denom {{ font-size: 0.85rem; opacity: 0.85; }}
    .score-breakdown {{ flex: 1; }}
    .score-breakdown h2 {{ font-size: 1.1rem; margin-bottom: 12px; }}
    .cat-row {{ display: flex; align-items: center; gap: 10px;
               margin-bottom: 6px; font-size: 0.9rem; }}
    .cat-label {{ width: 220px; color: #555; }}
    .cat-bar-wrap {{ flex: 1; background: #e9ecef; border-radius: 4px; height: 8px; }}
    .cat-bar {{ height: 8px; border-radius: 4px; background: {bar_colour}; }}
    .cat-score {{ width: 55px; text-align: right; font-weight: 600; }}
    .badges {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }}
    .badge {{ padding: 6px 14px; border-radius: 20px; font-size: 0.82rem;
             font-weight: 600; }}
    .badge-success {{ background: #d4edda; color: #155724; }}
    .badge-danger {{ background: #f8d7da; color: #721c24; }}
    .badge-warning {{ background: #fff3cd; color: #856404; }}
    .badge-secondary {{ background: #e2e3e5; color: #383d41; }}
    .badge-info {{ background: #d1ecf1; color: #0c5460; }}
    .section {{ background: white; border-radius: 12px; padding: 24px;
               margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,.06); }}
    .section h2 {{ font-size: 1.05rem; color: #1a1a2e; margin-bottom: 14px;
                  padding-bottom: 8px; border-bottom: 2px solid #f0f2f5; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th {{ background: #f8f9fa; color: #555; font-weight: 600;
         padding: 8px 12px; text-align: left; border-bottom: 2px solid #e9ecef; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #f0f2f5; vertical-align: top; }}
    tr:last-child td {{ border-bottom: none; }}
    .status-pass {{ color: #28a745; font-weight: 600; }}
    .status-fail {{ color: #dc3545; font-weight: 600; }}
    .status-error {{ color: #fd7e14; font-weight: 600; }}
    .status-timeout {{ color: #6c757d; font-weight: 600; }}
    .status-nt {{ color: #adb5bd; font-weight: 600; }}
    ul {{ list-style: none; padding: 0; }}
    ul li {{ padding: 4px 0; font-size: 0.9rem; }}
    .strength-item::before {{ content: "✓ "; color: #28a745; font-weight: 700; }}
    .problem-item::before {{ content: "⚠ "; color: #dc3545; }}
    .rec-item::before {{ content: "→ "; color: #0d6efd; }}
    .limit-item::before {{ content: "ℹ "; color: #6c757d; }}
    .evidence-label {{ display: inline-block; font-size: 0.72rem; padding: 2px 7px;
                      border-radius: 4px; background: #e9ecef; color: #555;
                      margin-left: 6px; vertical-align: middle; }}
    pre.error-output {{ background: #1e1e2e; color: #cdd6f4; padding: 14px;
                       border-radius: 8px; overflow-x: auto; font-size: 0.8rem;
                       max-height: 300px; overflow-y: auto; margin-top: 10px; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    @media (max-width: 640px) {{ .two-col {{ grid-template-columns: 1fr; }}
      .score-hero {{ flex-direction: column; }} .cat-label {{ width: 160px; }} }}
    footer {{ text-align: center; font-size: 0.8rem; color: #aaa;
             margin-top: 32px; padding-bottom: 24px; }}
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🤖 AI Project Evaluation Report</h1>
    <div class="meta">
      Evaluation ID: {_esc(eval_id)} &nbsp;|&nbsp;
      Generated: {_utc_now_iso()} &nbsp;|&nbsp;
      NxtWave AI Project Evaluator
    </div>
  </div>

  <!-- Score hero -->
  <div class="score-hero">
    <div class="score-circle">
      <span class="num">{int(total)}</span>
      <span class="denom">/ 100</span>
    </div>
    <div class="score-breakdown">
      <h2>{_esc(project_name)}</h2>
      <p style="color:#666;font-size:.9rem;margin-bottom:12px;">
        {_esc(project_goal[:120]) if project_goal else ''}
      </p>
      {category_rows}
    </div>
  </div>

  <!-- Status badges -->
  <div class="badges">
    <span class="badge {exec_class}">Execution: {exec_status}</span>
    <span class="badge badge-info">Tests: {test_summary}</span>
    <span class="badge badge-info">Time: {exec_time}</span>
    <span class="badge badge-info">Memory: {peak_mem}</span>
  </div>

  {error_section}

  <!-- Strengths & Problems -->
  <div class="two-col">
    <div class="section">
      <h2>Strengths</h2>
      {strengths_html or '<p style="color:#aaa">None identified.</p>'}
    </div>
    <div class="section">
      <h2>Issues Found</h2>
      {problems_html or '<p style="color:#aaa">None identified.</p>'}
    </div>
  </div>

  <!-- Recommendations -->
  <div class="section">
    <h2>Recommendations</h2>
    {recs_html or '<p style="color:#aaa">No recommendations.</p>'}
  </div>

  <!-- Tests -->
  <div class="section">
    <h2>Functional Test Results</h2>
    {test_rows}
  </div>

  <!-- Pipeline & Dependencies -->
  <div class="two-col">
    <div class="section">
      <h2>Pipeline Analysis</h2>
      {pipeline_info}
    </div>
    <div class="section">
      <h2>Dependencies</h2>
      {dep_info}
    </div>
  </div>

  <!-- Code Quality -->
  <div class="section">
    <h2>Code Quality</h2>
    {quality_info}
  </div>

  <!-- Limitations -->
  <div class="section">
    <h2>Evaluation Limitations</h2>
    {limits_html or '<p style="color:#aaa">None noted.</p>'}
  </div>

  <footer>
    Prototype evaluation environment. Not a production-grade hostile-code sandbox.
    Results are based on static analysis and controlled execution of student code.
  </footer>
</div>
</body>
</html>"""

        return html


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _category_rows(scores) -> str:
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

    rows = []

    for label, value, maximum in categories:
        pct = min(100, (value / maximum) * 100) if maximum else 0

        rows.append(
            f'<div class="cat-row">'
            f'<span class="cat-label">{label}</span>'
            f'<div class="cat-bar-wrap">'
            f'<div class="cat-bar" style="width:{pct:.0f}%"></div>'
            f'</div>'
            f'<span class="cat-score">{value:.1f} / {maximum}</span>'
            f'</div>'
        )

    return "\n".join(rows)


def _test_rows(testing) -> str:
    if testing is None or not testing.test_results:
        return '<p style="color:#aaa">No functional tests were run.</p>'

    rows = ["<table><thead><tr>"
            "<th>#</th><th>Input</th><th>Expected</th>"
            "<th>Actual</th><th>Status</th><th>Source</th>"
            "</tr></thead><tbody>"]

    for i, tr in enumerate(testing.test_results, 1):
        status_class = {
            TestStatus.PASS: "status-pass",
            TestStatus.FAIL: "status-fail",
            TestStatus.ERROR: "status-error",
            TestStatus.TIMEOUT: "status-timeout",
            TestStatus.NOT_TESTABLE: "status-nt",
        }.get(tr.status, "")

        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><code>{_esc(str(tr.expected_output or '')[:60])}</code></td>"
            f"<td><code>{_esc(str(tr.expected_output or 'N/A')[:60])}</code></td>"
            f"<td><code>{_esc(str(tr.actual_output or '')[:60])}</code></td>"
            f'<td class="{status_class}">{tr.status.value.upper()}</td>'
            f"<td>{_esc(tr.source)}</td>"
            f"</tr>"
        )

    rows.append("</tbody></table>")

    return "\n".join(rows)


def _dependency_info(dep) -> str:
    if dep is None:
        return '<p style="color:#aaa">Not checked.</p>'

    lines = []

    if dep.requirements_file:
        lines.append(
            f"<p><strong>File:</strong> {_esc(dep.requirements_file)}</p>"
        )

    if dep.declared_dependencies:
        deps_str = ", ".join(dep.declared_dependencies[:15])
        lines.append(
            f"<p><strong>Declared:</strong> {_esc(deps_str)}</p>"
        )

    if dep.missing_dependencies:
        missing_str = ", ".join(dep.missing_dependencies[:10])
        lines.append(
            f"<p style='color:#dc3545'><strong>Possibly missing:</strong> "
            f"{_esc(missing_str)}</p>"
        )

    if dep.install_success:
        lines.append(
            '<p class="status-pass">Dependencies installed successfully.</p>'
        )
    elif dep.install_failures:
        lines.append(
            '<p class="status-fail">Installation failed.</p>'
        )

    if dep.warnings:
        for w in dep.warnings[:3]:
            lines.append(f'<p style="color:#856404">⚠ {_esc(w)}</p>')

    return "\n".join(lines) if lines else '<p>No issues detected.</p>'


def _pipeline_info(pipeline) -> str:
    if pipeline is None:
        return '<p style="color:#aaa">Not analysed.</p>'

    lines = []

    stages = pipeline.detected_stages

    if stages:
        stages_str = " → ".join(stages)
        lines.append(f"<p><strong>Detected stages:</strong> {_esc(stages_str)}</p>")
    else:
        lines.append('<p style="color:#dc3545">No pipeline stages detected.</p>')

    if pipeline.entry_point:
        lines.append(
            f"<p><strong>Entry point:</strong> <code>{_esc(pipeline.entry_point)}</code></p>"
        )

    if pipeline.consistency_issues:
        for issue in pipeline.consistency_issues[:4]:
            lines.append(f'<p style="color:#856404">⚠ {_esc(issue)}</p>')

    return "\n".join(lines)


def _quality_info(quality) -> str:
    if quality is None:
        return '<p style="color:#aaa">Not analysed.</p>'

    lines = []

    all_findings = (
        quality.static_findings[:4]
        + quality.security_findings[:2]
        + quality.complexity_findings[:2]
        + quality.unused_code_findings[:2]
    )

    if all_findings:
        lines.append("<ul>")
        for finding in all_findings:
            lines.append(f"  <li>{_esc(finding)}</li>")
        lines.append("</ul>")
    else:
        lines.append("<p>No significant static findings.</p>")

    if quality.ai_findings:
        lines.append("<p><strong>AI interpretation:</strong></p><ul>")
        for f in quality.ai_findings[:3]:
            lines.append(f"  <li>{_esc(f)}</li>")
        lines.append("</ul>")

    return "\n".join(lines)


def _list_html(items: list[str], item_class: str) -> str:
    if not items:
        return ""

    lines = ["<ul>"]

    for item in items:
        lines.append(f'  <li class="{item_class}">{_esc(item)}</li>')

    lines.append("</ul>")

    return "\n".join(lines)


def _esc(text: str) -> str:
    """HTML-escape a string."""

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_id() -> str:
    from datetime import datetime, timezone
    import random

    now = datetime.now(timezone.utc)
    suffix = random.randint(100, 999)

    return now.strftime(f"%Y%m%d_%H%M%S_{suffix}")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def generate_report(
    report: EvaluationReport,
    results_root: Optional[Path] = None,
) -> dict[str, Path]:
    """Generate all reports and return output file paths."""

    generator = ReportGenerator(results_root=results_root)

    return generator.generate(report)
