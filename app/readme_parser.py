from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .models import ProjectSpecification, StudentSubmission


class ReadmeParser:
    """
    Deterministic parser for student README files.

    The parser extracts information when it is explicitly documented.
    It does not invent missing information.
    """

    README_NAMES = (
        "README.md",
        "README.MD",
        "README.markdown",
        "README.txt",
        "readme.md",
        "readme.txt",
    )

    # Common heading/label variations that students may use.
    FIELD_PATTERNS = {
        "project_goal": [
            r"^\s*(?:project\s+)?goal\s*[:\-]\s*(.+)$",
            r"^\s*(?:project\s+)?objective\s*[:\-]\s*(.+)$",
            r"^\s*purpose\s*[:\-]\s*(.+)$",
            r"^\s*description\s*[:\-]\s*(.+)$",
        ],
        "run_command": [
            r"^\s*(?:how\s+to\s+)?run(?:\s+the\s+project)?\s*[:\-]\s*(.+)$",
            r"^\s*execution\s+command\s*[:\-]\s*(.+)$",
        ],
        "python_version": [
            r"^\s*python\s+version\s*[:\-]\s*(.+)$",
            r"^\s*python\s*[:\-]\s*(.+)$",
            r"^\s*requires\s+python\s*[:\-]\s*(.+)$",
        ],
        "input_type": [
            r"^\s*input\s+type\s*[:\-]\s*(.+)$",
            r"^\s*input\s*[:\-]\s*(.+)$",
        ],
        "expected_behavior": [
            r"^\s*expected\s+output\s*[:\-]\s*(.+)$",
            r"^\s*expected\s+behavior\s*[:\-]\s*(.+)$",
            r"^\s*output\s*[:\-]\s*(.+)$",
        ],
        "external_requirements": [
            r"^\s*(?:external\s+)?(?:api|service|services)\s*[:\-]\s*(.+)$",
            r"^\s*external\s+requirements?\s*[:\-]\s*(.+)$",
        ],
        "dataset_info": [
            r"^\s*dataset\s*[:\-]\s*(.+)$",
            r"^\s*data\s*[:\-]\s*(.+)$",
        ],
        "model_info": [
            r"^\s*model\s*[:\-]\s*(.+)$",
            r"^\s*model\s+used\s*[:\-]\s*(.+)$",
        ],
    }

    SECTION_ALIASES = {
        "project_goal": {
            "goal",
            "project goal",
            "objective",
            "project objective",
            "purpose",
            "description",
            "about",
        },
        "run_command": {
            "how to run",
            "running",
            "run",
            "execution",
            "usage",
        },
        "input_type": {
            "input",
            "input format",
            "input type",
        },
        "expected_behavior": {
            "output",
            "expected output",
            "expected behavior",
            "result",
        },
        "dependencies": {
            "dependencies",
            "requirements",
            "requirements and installation",
            "installation",
            "setup",
        },
        "python_version": {
            "python version",
            "environment",
            "requirements",
        },
        "dataset_info": {
            "dataset",
            "data",
        },
        "model_info": {
            "model",
            "model architecture",
            "machine learning model",
        },
        "external_requirements": {
            "api",
            "apis",
            "external api",
            "external services",
            "environment variables",
        },
        "sample_input": {
            "sample input",
            "example input",
            "input example",
        },
        "sample_output": {
            "sample output",
            "example output",
            "output example",
            "expected output example",
        },
    }

    def parse_file(
        self,
        readme_path: Path,
        submission: Optional[StudentSubmission] = None,
    ) -> ProjectSpecification:
        """
        Parse a README file and optionally merge student-provided fields.
        """

        readme_path = Path(readme_path)

        if not readme_path.exists():
            specification = self._from_submission(submission)

            specification.extraction_warnings.append(
                "README file was not found."
            )

            return specification

        if not readme_path.is_file():
            specification = self._from_submission(submission)

            specification.extraction_warnings.append(
                "README path is not a file."
            )

            return specification

        try:
            text = readme_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            specification = self._from_submission(submission)

            specification.extraction_warnings.append(
                f"Unable to read README: {exc}"
            )

            return specification

        return self.parse_text(
            text=text,
            submission=submission,
        )

    def parse_text(
        self,
        text: str,
        submission: Optional[StudentSubmission] = None,
    ) -> ProjectSpecification:
        """
        Parse README text into a ProjectSpecification.

        Only explicit information is extracted.
        Missing information remains None/empty.
        """

        specification = self._from_submission(submission)

        if not text or not text.strip():
            specification.extraction_warnings.append(
                "README is empty."
            )
            return specification

        normalized_text = self._normalize_text(text)

        sections = self._extract_sections(normalized_text)

        # --------------------------------------------------------------
        # Structured sections
        # --------------------------------------------------------------

        self._apply_section(
            specification,
            sections,
            "project_goal",
        )

        self._apply_section(
            specification,
            sections,
            "run_command",
        )

        self._apply_section(
            specification,
            sections,
            "input_type",
        )

        self._apply_section(
            specification,
            sections,
            "expected_behavior",
        )

        self._apply_section(
            specification,
            sections,
            "python_version",
        )

        self._apply_section(
            specification,
            sections,
            "dataset_info",
        )

        self._apply_section(
            specification,
            sections,
            "model_info",
        )

        self._apply_section(
            specification,
            sections,
            "external_requirements",
        )

        # --------------------------------------------------------------
        # Label-based extraction
        # --------------------------------------------------------------

        self._extract_labeled_fields(
            normalized_text,
            specification,
        )

        # --------------------------------------------------------------
        # Dependencies
        # --------------------------------------------------------------

        specification.dependencies.extend(
            self._extract_dependency_lines(
                sections.get("dependencies", [])
            )
        )

        specification.dependencies = self._unique_preserve_order(
            specification.dependencies
        )

        # --------------------------------------------------------------
        # Test cases
        # --------------------------------------------------------------

        sample_inputs = self._extract_sample_values(
            sections.get("sample_input", [])
        )

        sample_outputs = self._extract_sample_values(
            sections.get("sample_output", [])
        )

        self._add_sample_test_cases(
            specification,
            sample_inputs,
            sample_outputs,
        )

        # --------------------------------------------------------------
        # Warnings
        # --------------------------------------------------------------

        self._add_missing_information_warnings(
            specification
        )

        # Record where the information came from.
        specification.source_information.setdefault(
            "parser",
            "deterministic_readme_parser",
        )

        return specification

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize line endings without destroying Markdown structure.
        """

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        return text

    def _from_submission(
        self,
        submission: Optional[StudentSubmission],
    ) -> ProjectSpecification:
        """
        Build an initial specification from explicit student form data.
        """

        if submission is None:
            return ProjectSpecification()

        specification = ProjectSpecification(
            project_name=submission.project_name,
            project_goal=submission.project_goal,
            run_command=submission.run_command,
            python_version=submission.python_version,
            input_type=submission.input_type,
            expected_behavior=submission.expected_behavior,
            dataset_info=submission.dataset_model_info,
        )

        if submission.external_api_service:
            specification.external_requirements.append(
                submission.external_api_service
            )

        if submission.sample_input:
            specification.sample_test_cases.append(
                {
                    "input": submission.sample_input,
                    "expected_output": (
                        submission.sample_expected_output
                    ),
                    "source": "student_provided",
                }
            )

        specification.source_information.update(
            {
                "project_name": "student_form",
                "project_goal": "student_form",
                "run_command": "student_form",
                "input_type": "student_form",
                "expected_behavior": "student_form",
            }
        )

        return specification

    def _extract_sections(
        self,
        text: str,
    ) -> dict[str, list[str]]:
        """
        Parse Markdown headings and assign content to known sections.
        """

        sections: dict[str, list[str]] = {}

        current_section: Optional[str] = None

        lines = text.splitlines()

        for line in lines:

            heading_match = re.match(
                r"^\s{0,3}#{1,6}\s+(.+?)\s*$",
                line,
            )

            if heading_match:
                heading = self._clean_heading(
                    heading_match.group(1)
                )

                current_section = self._section_key(
                    heading
                )

                if current_section:
                    sections.setdefault(
                        current_section,
                        [],
                    )

                continue

            if current_section:
                sections[current_section].append(
                    line.rstrip()
                )

        return sections

    @staticmethod
    def _clean_heading(heading: str) -> str:
        """
        Clean Markdown heading text.
        """

        heading = re.sub(
            r"`",
            "",
            heading,
        )

        heading = re.sub(
            r"[*_]",
            "",
            heading,
        )

        heading = re.sub(
            r"\s+",
            " ",
            heading,
        )

        return heading.strip().lower()

    def _section_key(
        self,
        heading: str,
    ) -> Optional[str]:
        """
        Map a README heading to one of our known section names.
        """

        normalized = heading.strip().lower()

        for key, aliases in self.SECTION_ALIASES.items():
            if normalized in aliases:
                return key

        return None

    def _apply_section(
        self,
        specification: ProjectSpecification,
        sections: dict[str, list[str]],
        field_name: str,
    ) -> None:
        """
        Apply the first meaningful section content to a specification field.
        """

        lines = sections.get(field_name, [])

        value = self._section_to_value(lines)

        if not value:
            return

        current_value = getattr(
            specification,
            field_name,
            None,
        )

        # Explicit student input wins over README extraction.
        if current_value:
            return

        setattr(
            specification,
            field_name,
            value,
        )

        specification.source_information[
            field_name
        ] = "readme_section"

    @staticmethod
    def _section_to_value(
        lines: list[str],
    ) -> Optional[str]:
        """
        Convert section lines into a compact value.
        """

        meaningful_lines = []

        for line in lines:
            cleaned = line.strip()

            if not cleaned:
                continue

            cleaned = re.sub(
                r"^[-*+]\s+",
                "",
                cleaned,
            )

            meaningful_lines.append(cleaned)

        if not meaningful_lines:
            return None

        return "\n".join(meaningful_lines)

    def _extract_labeled_fields(
        self,
        text: str,
        specification: ProjectSpecification,
    ) -> None:
        """
        Extract fields written as:

        Goal: ...
        Input: ...
        Expected Output: ...
        """

        lines = text.splitlines()

        for field_name, patterns in self.FIELD_PATTERNS.items():

            # Respect explicit student-submitted data.
            existing_value = getattr(
                specification,
                field_name,
                None,
            )

            if existing_value:
                continue

            for pattern in patterns:

                for line in lines:
                    match = re.match(
                        pattern,
                        line,
                        flags=re.IGNORECASE,
                    )

                    if not match:
                        continue

                    value = match.group(1).strip()

                    if not value:
                        continue

                    setattr(
                        specification,
                        field_name,
                        value,
                    )

                    specification.source_information[
                        field_name
                    ] = "readme_label"

                    break

                if getattr(specification, field_name, None):
                    break

    @staticmethod
    def _extract_dependency_lines(
        lines: list[str],
    ) -> list[str]:
        """
        Extract likely dependency names from a Dependencies section.
        """

        dependencies: list[str] = []

        for line in lines:

            cleaned = line.strip()

            if not cleaned:
                continue

            # Remove common list markers.
            cleaned = re.sub(
                r"^[-*+]\s+",
                "",
                cleaned,
            )

            # Ignore common explanatory lines.
            lowered = cleaned.lower()

            if lowered in {
                "python",
                "packages",
                "libraries",
                "required packages",
            }:
                continue

            # Handle:
            # pandas
            # pandas==2.2.0
            # pandas>=2.0
            # pandas 2.0
            match = re.match(
                r"^([A-Za-z0-9_.-]+)"
                r"(?:\s*(?:==|>=|<=|~=|>|<)\s*.*)?$",
                cleaned,
            )

            if match:
                dependencies.append(
                    match.group(1)
                )
            else:
                # Keep the original line if it looks useful.
                if len(cleaned) <= 120:
                    dependencies.append(cleaned)

        return dependencies

    @staticmethod
    def _extract_sample_values(
        lines: list[str],
    ) -> list[str]:
        """
        Extract sample values from a README section.

        Supports plain text and fenced code blocks.
        """

        values: list[str] = []

        inside_code_block = False
        code_lines: list[str] = []

        for line in lines:

            stripped = line.strip()

            if stripped.startswith("```"):
                if inside_code_block:

                    value = "\n".join(
                        code_lines
                    ).strip()

                    if value:
                        values.append(value)

                    code_lines = []
                    inside_code_block = False

                else:
                    inside_code_block = True

                continue

            if inside_code_block:
                code_lines.append(line)

                continue

            if stripped:
                cleaned = re.sub(
                    r"^[-*+]\s+",
                    "",
                    stripped,
                )

                cleaned = re.sub(
                    r"^\*\*.*?\*\*:\s*",
                    "",
                    cleaned,
                )

                values.append(cleaned)

        if inside_code_block and code_lines:
            value = "\n".join(
                code_lines
            ).strip()

            if value:
                values.append(value)

        return ReadmeParser._unique_preserve_order(
            values
        )

    @staticmethod
    def _add_sample_test_cases(
        specification: ProjectSpecification,
        sample_inputs: list[str],
        sample_outputs: list[str],
    ) -> None:
        """
        Combine sample inputs and outputs into test cases.

        If the number of inputs and outputs differs, we preserve the
        input but leave expected_output as None rather than inventing it.
        """

        existing_pairs = {
            (
                case.get("input"),
                case.get("expected_output"),
            )
            for case in specification.sample_test_cases
        }

        for index, sample_input in enumerate(sample_inputs):

            expected_output = (
                sample_outputs[index]
                if index < len(sample_outputs)
                else None
            )

            pair = (
                sample_input,
                expected_output,
            )

            if pair in existing_pairs:
                continue

            specification.sample_test_cases.append(
                {
                    "input": sample_input,
                    "expected_output": expected_output,
                    "source": "readme",
                }
            )

            existing_pairs.add(pair)

    @staticmethod
    def _add_missing_information_warnings(
        specification: ProjectSpecification,
    ) -> None:
        """
        Record missing information without guessing.
        """

        if not specification.project_goal:
            specification.extraction_warnings.append(
                "Project goal was not explicitly provided."
            )

        if not specification.run_command:
            specification.extraction_warnings.append(
                "Run command was not explicitly provided."
            )

        if not specification.input_type:
            specification.extraction_warnings.append(
                "Input type was not explicitly provided."
            )

        if not specification.expected_behavior:
            specification.extraction_warnings.append(
                "Expected behavior/output was not explicitly provided."
            )

        if not specification.sample_test_cases:
            specification.extraction_warnings.append(
                "No explicit sample test case was found."
            )

    @staticmethod
    def _unique_preserve_order(
        values: list[str],
    ) -> list[str]:
        """
        Remove duplicates while preserving original order.
        """

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


def find_readme(project_root: Path) -> Optional[Path]:
    """
    Find a README file in the project root.
    """

    project_root = Path(project_root)

    for filename in ReadmeParser.README_NAMES:

        candidate = project_root / filename

        if candidate.is_file():
            return candidate

    return None


def parse_project_readme(
    project_root: Path,
    submission: Optional[StudentSubmission] = None,
) -> ProjectSpecification:
    """
    Convenience function used by the evaluation pipeline.
    """

    project_root = Path(project_root)

    readme_path = find_readme(project_root)

    parser = ReadmeParser()

    if readme_path is None:

        specification = parser.parse_text(
            text="",
            submission=submission,
        )

        specification.extraction_warnings.append(
            "No README file was found in the project root."
        )

        return specification

    return parser.parse_file(
        readme_path=readme_path,
        submission=submission,
    )