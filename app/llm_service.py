"""
llm_service.py
--------------
Selective Gemini API integration.

Uses google-genai (official Google Python SDK).
API key is read from the GEMINI_API_KEY environment variable or .env file.

If the API key is unavailable or any call fails, all methods return None
and the caller must handle the absence of AI-generated content gracefully.

Gemini is used for:
  - Interpreting README / project intent when deterministic parsing fails.
  - Summarising pipeline structure.
  - Qualitative code-quality interpretation.
  - Explaining runtime / static errors.
  - Generating candidate functional test inputs when documentation is thin.

Gemini is NOT used for:
  - Counting files / dependencies.
  - Timing or memory measurement.
  - Basic PASS / FAIL determination where deterministic comparison works.
  - Generating the final numerical score.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK import
# ---------------------------------------------------------------------------

try:
    from google import genai  # type: ignore[import]
    from google.genai import types as genai_types  # type: ignore[import]
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Model selection – use a current, low-cost Flash model
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gemini-2.0-flash"
_MAX_OUTPUT_TOKENS = 2048


class LLMService:
    """
    Thin wrapper around the Gemini API for selective AI-assisted analysis.

    All methods return None when the API is unavailable or a call fails.
    """

    def __init__(self, model: str = _DEFAULT_MODEL):
        self.model = model
        self._client: Any = None
        self._available: bool = False

        self._initialise()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self) -> None:
        """Attempt to create a Gemini client from the environment."""

        if not _SDK_AVAILABLE:
            logger.warning(
                "google-genai SDK is not installed. "
                "AI-assisted analysis will be disabled."
            )
            return

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()

        if not api_key:
            logger.warning(
                "GEMINI_API_KEY is not set. "
                "AI-assisted analysis will be disabled."
            )
            return

        try:
            self._client = genai.Client(api_key=api_key)
            self._available = True
            logger.info(
                "Gemini client initialised with model: %s",
                self.model,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Failed to initialise Gemini client: %s",
                exc,
            )

    @property
    def is_available(self) -> bool:
        """Return True if the Gemini API is ready to use."""
        return self._available and self._client is not None

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def _call(self, prompt: str) -> Optional[str]:
        """
        Make a single Gemini request and return the text response.

        Returns None on any failure.
        """

        if not self.is_available:
            return None

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=_MAX_OUTPUT_TOKENS,
                    temperature=0.2,
                ),
            )

            text = response.text

            if not text:
                return None

            return text.strip()

        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Gemini API call failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

    def interpret_project_intent(
        self,
        project_name: str,
        project_goal: str,
        readme_text: str,
    ) -> Optional[str]:
        """
        Ask Gemini to summarise the project intent in 2-3 sentences.

        Used when the deterministic README parser returns insufficient
        structured information.
        """

        if not self.is_available:
            return None

        truncated_readme = readme_text[:3000] if readme_text else ""

        prompt = (
            f"You are evaluating a lightweight student Python AI project.\n\n"
            f"Project name: {project_name}\n"
            f"Student-stated goal: {project_goal}\n\n"
            f"README excerpt:\n{truncated_readme}\n\n"
            "In 2-3 sentences, summarise what this project is intended to do "
            "and what kind of AI/ML task it performs. "
            "Only describe what is explicitly stated. "
            "Do not invent features or capabilities."
        )

        return self._call(prompt)

    def summarise_pipeline(
        self,
        project_name: str,
        project_goal: str,
        detected_stages: list[str],
        pipeline_files: list[str],
        consistency_issues: list[str],
    ) -> Optional[str]:
        """
        Ask Gemini to provide a qualitative pipeline summary.
        """

        if not self.is_available:
            return None

        prompt = (
            f"You are evaluating a lightweight student Python AI project.\n\n"
            f"Project name: {project_name}\n"
            f"Goal: {project_goal}\n\n"
            f"Pipeline stages detected by static analysis: "
            f"{', '.join(detected_stages) or 'None'}\n"
            f"Pipeline-relevant files: "
            f"{', '.join(pipeline_files[:10]) or 'None'}\n"
            f"Pipeline consistency issues: "
            f"{'; '.join(consistency_issues) or 'None'}\n\n"
            "In 2-3 sentences, briefly describe the pipeline structure "
            "evident in this project. Note any clear gaps or strengths. "
            "Only describe what is stated above. Do not invent information."
        )

        return self._call(prompt)

    def interpret_code_quality(
        self,
        project_name: str,
        static_findings: list[str],
        security_findings: list[str],
        complexity_findings: list[str],
        unused_code_findings: list[str],
        code_snippet: Optional[str] = None,
    ) -> Optional[str]:
        """
        Ask Gemini for a brief qualitative interpretation of code quality.

        Only a limited snippet of code is sent – never the whole repository.
        """

        if not self.is_available:
            return None

        findings_text = "\n".join(
            [
                f"- {finding}"
                for finding in (
                    static_findings[:5]
                    + security_findings[:3]
                    + complexity_findings[:3]
                    + unused_code_findings[:3]
                )
            ]
        ) or "No static findings."

        snippet_text = ""
        if code_snippet:
            snippet_text = (
                f"\n\nCode snippet (first 1500 chars):\n"
                f"{code_snippet[:1500]}"
            )

        prompt = (
            f"You are evaluating a lightweight student Python AI project.\n\n"
            f"Project: {project_name}\n\n"
            f"Static analysis findings:\n{findings_text}"
            f"{snippet_text}\n\n"
            "In 2-3 sentences, briefly comment on the code quality of this "
            "project. Be constructive. Only comment on what the findings "
            "indicate – do not invent issues."
        )

        return self._call(prompt)

    def explain_runtime_error(
        self,
        project_name: str,
        error_type: Optional[str],
        error_message: Optional[str],
        stderr_excerpt: str,
    ) -> Optional[str]:
        """
        Ask Gemini to explain a runtime error in plain language.
        """

        if not self.is_available:
            return None

        stderr_short = stderr_excerpt[-1500:] if stderr_excerpt else ""

        prompt = (
            f"A student Python AI project failed during evaluation.\n\n"
            f"Project: {project_name}\n"
            f"Error type: {error_type or 'unknown'}\n"
            f"Error message: {error_message or 'unknown'}\n\n"
            f"Stderr (last 1500 chars):\n{stderr_short}\n\n"
            "In 2-3 sentences, explain what likely caused this error and "
            "what the student should do to fix it. Be specific and practical."
        )

        return self._call(prompt)

    def generate_candidate_tests(
        self,
        project_name: str,
        project_goal: str,
        input_type: Optional[str],
        expected_behavior: Optional[str],
    ) -> Optional[list[dict[str, str]]]:
        """
        Ask Gemini to suggest 2-3 candidate functional test inputs.

        Returns a list of dicts with keys: input, expected_output (can be None).
        These are CANDIDATES only – they are not guaranteed correct ground truth.

        Returns None if the API is unavailable or the response cannot be parsed.
        """

        if not self.is_available:
            return None

        prompt = (
            f"You are helping evaluate a lightweight student Python AI project.\n\n"
            f"Project name: {project_name}\n"
            f"Project goal: {project_goal}\n"
            f"Input type: {input_type or 'not specified'}\n"
            f"Expected behavior: {expected_behavior or 'not specified'}\n\n"
            "Suggest exactly 2 simple test inputs that could be used to test "
            "this project by passing them to stdin. "
            "For each test, also suggest what a reasonable expected output "
            "might look like (or null if the output is unpredictable).\n\n"
            "Respond ONLY with a valid JSON array. Example format:\n"
            '[\n'
            '  {"input": "sample text", "expected_output": "positive"},\n'
            '  {"input": "other text", "expected_output": null}\n'
            "]\n\n"
            "Do not include any explanation outside the JSON array."
        )

        raw = self._call(prompt)

        if not raw:
            return None

        # Extract JSON from response (model may include markdown fences)
        json_text = _extract_json_from_text(raw)

        if not json_text:
            return None

        try:
            parsed = json.loads(json_text)

            if not isinstance(parsed, list):
                return None

            tests: list[dict[str, str]] = []

            for item in parsed:
                if not isinstance(item, dict):
                    continue

                tests.append(
                    {
                        "input": str(item.get("input", "")),
                        "expected_output": item.get("expected_output"),
                    }
                )

            return tests if tests else None

        except (json.JSONDecodeError, TypeError):
            return None

    def generate_strengths_and_recommendations(
        self,
        project_name: str,
        project_goal: str,
        scores: dict[str, float],
        static_findings: list[str],
        execution_success: bool,
        test_summary: str,
    ) -> Optional[dict[str, list[str]]]:
        """
        Ask Gemini to generate a short list of strengths and recommendations.

        Returns a dict: {"strengths": [...], "recommendations": [...]}
        or None on failure.
        """

        if not self.is_available:
            return None

        score_lines = "\n".join(
            f"  {k}: {v}"
            for k, v in scores.items()
        )

        findings_text = "\n".join(
            f"  - {f}"
            for f in static_findings[:6]
        ) or "  None"

        prompt = (
            f"You are summarising an automated evaluation of a student Python "
            f"AI project for constructive feedback.\n\n"
            f"Project: {project_name}\n"
            f"Goal: {project_goal}\n\n"
            f"Scores:\n{score_lines}\n\n"
            f"Execution: {'SUCCESS' if execution_success else 'FAILED'}\n"
            f"Test summary: {test_summary}\n\n"
            f"Key static findings:\n{findings_text}\n\n"
            "Produce a JSON object with exactly two keys:\n"
            '  "strengths": a list of 2-3 brief strengths (strings)\n'
            '  "recommendations": a list of 2-3 actionable improvements '
            "(strings)\n\n"
            "Respond ONLY with a valid JSON object. No extra text."
        )

        raw = self._call(prompt)

        if not raw:
            return None

        json_text = _extract_json_from_text(raw)

        if not json_text:
            return None

        try:
            parsed = json.loads(json_text)

            if not isinstance(parsed, dict):
                return None

            return {
                "strengths": [
                    str(s) for s in parsed.get("strengths", [])
                ][:3],
                "recommendations": [
                    str(r) for r in parsed.get("recommendations", [])
                ][:3],
            }

        except (json.JSONDecodeError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract a JSON object or array from text that may include markdown fences.
    """

    # Try the whole text first.
    stripped = text.strip()

    if stripped.startswith(("{", "[")):
        return stripped

    # Try to find a fenced JSON block.
    import re

    match = re.search(
        r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
        stripped,
        re.DOTALL,
    )

    if match:
        return match.group(1)

    # Try to find a raw JSON object or array.
    match = re.search(
        r"(\{.*\}|\[.*\])",
        stripped,
        re.DOTALL,
    )

    if match:
        return match.group(1)

    return None


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_default_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Return the shared LLM service instance."""

    global _default_service  # pylint: disable=global-statement

    if _default_service is None:
        _default_service = LLMService()

    return _default_service
