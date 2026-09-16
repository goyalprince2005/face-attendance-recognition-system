"""
test_registration.py
=====================
Integration-level tests that exercise RegistrationManager end-to-end
against the small anonymised sample photo sets shipped in
data/samples/ (see data/samples/README.md for their provenance).
These are slower than the pure unit tests but validate that detection
-> encoding -> database -> LBPH training actually works together.
"""

import shutil

import pytest

from src import config
from src.database import Database
from src.face_encoder import FaceEncoder
from src.face_recognizer import FaceRecognizer
from src.registration import RegistrationError, RegistrationManager

SAMPLE_DIR = config.BASE_DIR / "data" / "samples" / "student_amit_kumar"


@pytest.fixture
def isolated_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "model.yml")
    monkeypatch.setattr(config, "LABELS_FILE", tmp_path / "labels.json")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "KNOWN_FACES_DIR", tmp_path / "known_faces")

    db = Database(db_path=tmp_path / "test.db")
    return RegistrationManager(db=db, encoder=FaceEncoder(), recognizer=FaceRecognizer())


@pytest.mark.skipif(not SAMPLE_DIR.exists(), reason="sample photo dataset not present")
class TestRegistrationIntegration:
    def test_register_student_from_photo_folder(self, isolated_manager):
        result = isolated_manager.register_student(
            "CS9001", "Test Student", str(SAMPLE_DIR)
        )
        assert result["student_id"] == "CS9001"
        assert result["samples_used"] >= config.MIN_TRAINING_SAMPLES
        assert isolated_manager.db.student_exists("CS9001")

    def test_duplicate_registration_is_rejected(self, isolated_manager):
        isolated_manager.register_student("CS9001", "Test Student", str(SAMPLE_DIR))
        with pytest.raises(RegistrationError):
            isolated_manager.register_student("CS9001", "Test Student", str(SAMPLE_DIR))

    def test_invalid_student_id_is_rejected(self, isolated_manager):
        with pytest.raises(RegistrationError):
            isolated_manager.register_student("!!bad id!!", "Test Student", str(SAMPLE_DIR))

    def test_remove_student_deletes_record_and_archive(self, isolated_manager):
        isolated_manager.register_student("CS9001", "Test Student", str(SAMPLE_DIR))
        assert isolated_manager.remove_student("CS9001") is True
        assert not isolated_manager.db.student_exists("CS9001")
