"""
utils/validators.py
====================
Small, dependency-free validation helpers used before any data is written
to the database or filesystem. Centralising validation here (rather than
inline in every command) is what keeps VALIDATION & ERROR HANDLING
consistent across the application (a technical expectation of the
project brief) and protects data integrity (a SECURITY concern - malformed
IDs / path traversal in student names, etc.).
"""

import re
from pathlib import Path

STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{2,30}$")
NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,60}$")
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ValidationError(ValueError):
    """Raised whenever user-supplied input fails validation."""


def validate_student_id(student_id: str) -> str:
    student_id = (student_id or "").strip()
    if not STUDENT_ID_PATTERN.match(student_id):
        raise ValidationError(
            "Student ID must be 2-30 characters long and contain only "
            "letters, digits, underscores or hyphens (e.g. 'CS2023001')."
        )
    return student_id


def validate_name(name: str) -> str:
    name = (name or "").strip()
    if not NAME_PATTERN.match(name):
        raise ValidationError(
            "Name must start with a letter and contain only letters, "
            "spaces, apostrophes, periods or hyphens."
        )
    return name


def validate_directory_exists(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise ValidationError(f"Directory does not exist: {p}")
    return p


def validate_image_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise ValidationError(f"Image file does not exist: {p}")
    if p.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        raise ValidationError(
            f"Unsupported image format '{p.suffix}'. "
            f"Supported: {', '.join(sorted(VALID_IMAGE_EXTENSIONS))}"
        )
    return p


def list_image_files(directory: Path):
    """Return sorted list of valid image files inside a directory."""
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
