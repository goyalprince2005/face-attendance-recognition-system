import pytest

from src.utils.validators import (
    ValidationError,
    validate_directory_exists,
    validate_name,
    validate_student_id,
)


class TestStudentIdValidation:
    def test_accepts_valid_ids(self):
        assert validate_student_id("CS2023001") == "CS2023001"
        assert validate_student_id("a_b-c") == "a_b-c"

    def test_rejects_too_short(self):
        with pytest.raises(ValidationError):
            validate_student_id("a")

    def test_rejects_special_characters(self):
        with pytest.raises(ValidationError):
            validate_student_id("CS/2023*001")

    def test_rejects_empty(self):
        with pytest.raises(ValidationError):
            validate_student_id("")

    def test_strips_whitespace(self):
        assert validate_student_id("  CS001  ") == "CS001"


class TestNameValidation:
    def test_accepts_valid_names(self):
        assert validate_name("Amit Kumar") == "Amit Kumar"
        assert validate_name("O'Neil") == "O'Neil"

    def test_rejects_names_starting_with_digit(self):
        with pytest.raises(ValidationError):
            validate_name("123Amit")

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            validate_name("")


class TestDirectoryValidation:
    def test_accepts_existing_directory(self, tmp_path):
        result = validate_directory_exists(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_rejects_missing_directory(self, tmp_path):
        with pytest.raises(ValidationError):
            validate_directory_exists(str(tmp_path / "does_not_exist"))
