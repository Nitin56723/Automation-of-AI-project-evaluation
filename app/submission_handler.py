from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from git import Repo
from git.exc import GitCommandError

from .models import StudentSubmission, SubmissionType


class SubmissionError(Exception):
    """Raised when a project submission cannot be prepared."""


class SubmissionHandler:
    """
    Handles GitHub and ZIP-based project submissions.

    Responsibilities:
    - Validate the submission.
    - Download/clone GitHub repositories.
    - Extract ZIP submissions.
    - Prepare a temporary workspace.
    - Prevent obvious path traversal during ZIP extraction.
    """

    GITHUB_URL_PATTERN = re.compile(
        r"^https?://github\.com/[^/\s]+/[^/\s]+/?(?:\.git)?(?:\?.*)?$",
        re.IGNORECASE,
    )

    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Parameters
        ----------
        workspace_root:
            Optional base directory for temporary workspaces.
            If not supplied, the system temporary directory is used.
        """

        if workspace_root is None:
            self.workspace_root = Path(tempfile.gettempdir()) / (
                "nxtwave_ai_project_evaluator"
            )
        else:
            self.workspace_root = Path(workspace_root)

        self.workspace_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def is_valid_github_url(url: str) -> bool:
        """Return True if the supplied URL looks like a GitHub repository URL."""

        if not url:
            return False

        url = url.strip()

        if not SubmissionHandler.GITHUB_URL_PATTERN.match(url):
            return False

        parsed = urlparse(url)

        return (
            parsed.scheme in {"http", "https"}
            and parsed.netloc.lower() == "github.com"
        )

    @staticmethod
    def validate_zip_file(zip_path: Path) -> None:
        """
        Validate basic ZIP properties before extraction.
        """

        if not zip_path.exists():
            raise SubmissionError(f"ZIP file does not exist: {zip_path}")

        if not zip_path.is_file():
            raise SubmissionError(f"ZIP path is not a file: {zip_path}")

        if zip_path.suffix.lower() != ".zip":
            raise SubmissionError("Uploaded project must be a .zip file.")

        if not zipfile.is_zipfile(zip_path):
            raise SubmissionError("The uploaded file is not a valid ZIP archive.")

    @staticmethod
    def validate_submission(submission: StudentSubmission) -> list[str]:
        """
        Validate mandatory student fields and source selection.
        """

        errors = submission.validate()

        if submission.github_url:
            if not SubmissionHandler.is_valid_github_url(submission.github_url):
                errors.append("Invalid GitHub repository URL.")

        if submission.zip_path:
            try:
                SubmissionHandler.validate_zip_file(
                    Path(submission.zip_path)
                )
            except SubmissionError as exc:
                errors.append(str(exc))

        return errors

    # ------------------------------------------------------------------
    # Workspace creation
    # ------------------------------------------------------------------

    def create_workspace(self, project_name: str) -> Path:
        """
        Create a unique temporary workspace for one evaluation.
        """

        safe_name = self._safe_project_name(project_name)

        workspace = Path(
            tempfile.mkdtemp(
                prefix=f"{safe_name}_",
                dir=self.workspace_root,
            )
        )

        return workspace

    @staticmethod
    def _safe_project_name(project_name: str) -> str:
        """
        Convert a project name into a filesystem-safe name.
        """

        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", project_name.strip())

        cleaned = cleaned.strip("_")

        if not cleaned:
            cleaned = "project"

        return cleaned[:80]

    # ------------------------------------------------------------------
    # GitHub handling
    # ------------------------------------------------------------------

    def clone_github_repository(
        self,
        github_url: str,
        destination: Path,
    ) -> Path:
        """
        Clone a public GitHub repository.

        Returns the actual project root.

        If the repository itself contains a single top-level directory,
        the caller can later decide whether to treat that directory as the
        project root.
        """

        if not self.is_valid_github_url(github_url):
            raise SubmissionError("Invalid GitHub repository URL.")

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        try:
            Repo.clone_from(
                github_url,
                destination,
                depth=1,
            )
        except GitCommandError as exc:
            raise SubmissionError(
                f"Unable to clone GitHub repository: {exc}"
            ) from exc

        return destination

    # ------------------------------------------------------------------
    # ZIP handling
    # ------------------------------------------------------------------

    def extract_zip_submission(
        self,
        zip_path: Path,
        destination: Path,
    ) -> Path:
        """
        Extract a ZIP submission safely.

        Includes a basic ZIP path traversal check so entries cannot escape
        the intended extraction directory.
        """

        zip_path = Path(zip_path)
        destination = Path(destination)

        self.validate_zip_file(zip_path)

        destination.mkdir(parents=True, exist_ok=True)

        destination_resolved = destination.resolve()

        try:
            with zipfile.ZipFile(zip_path, "r") as archive:

                for member in archive.infolist():
                    member_path = destination / member.filename

                    try:
                        member_resolved = member_path.resolve()
                    except OSError as exc:
                        raise SubmissionError(
                            f"Unable to resolve ZIP entry: {member.filename}"
                        ) from exc

                    if not self._is_within_directory(
                        member_resolved,
                        destination_resolved,
                    ):
                        raise SubmissionError(
                            f"Unsafe ZIP entry detected: {member.filename}"
                        )

                archive.extractall(destination)

        except zipfile.BadZipFile as exc:
            raise SubmissionError(
                "The uploaded ZIP archive is corrupted or invalid."
            ) from exc

        return destination

    @staticmethod
    def _is_within_directory(
        target: Path,
        directory: Path,
    ) -> bool:
        """Return True if target is inside directory."""

        try:
            target.relative_to(directory)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Unified preparation
    # ------------------------------------------------------------------

    def prepare_submission(
        self,
        submission: StudentSubmission,
    ) -> Path:
        """
        Validate and prepare either a GitHub or ZIP submission.

        The returned directory is a temporary workspace containing the
        student's project.
        """

        errors = self.validate_submission(submission)

        if errors:
            raise SubmissionError(
                "Submission validation failed:\n- "
                + "\n- ".join(errors)
            )

        workspace = self.create_workspace(submission.project_name)

        try:
            if submission.submission_type == SubmissionType.GITHUB:

                if not submission.github_url:
                    raise SubmissionError(
                        "GitHub submission type selected but no URL provided."
                    )

                self.clone_github_repository(
                    submission.github_url,
                    workspace,
                )

            elif submission.submission_type == SubmissionType.ZIP:

                if not submission.zip_path:
                    raise SubmissionError(
                        "ZIP submission type selected but no ZIP provided."
                    )

                self.extract_zip_submission(
                    Path(submission.zip_path),
                    workspace,
                )

            else:
                raise SubmissionError(
                    "Unknown or missing submission type."
                )

        except Exception:
            # Clean up a partially prepared workspace if preparation fails.
            self.cleanup_workspace(workspace)
            raise

        return workspace

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def cleanup_workspace(workspace: Path) -> None:
        """
        Remove a temporary evaluation workspace.
        """

        workspace = Path(workspace)

        if not workspace.exists():
            return

        shutil.rmtree(workspace, ignore_errors=True)


def create_handler(workspace_root: Optional[str] = None) -> SubmissionHandler:
    """
    Convenience factory for other modules.
    """

    root = Path(workspace_root) if workspace_root else None

    return SubmissionHandler(workspace_root=root)