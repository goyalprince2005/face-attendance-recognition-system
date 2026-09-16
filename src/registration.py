"""
registration.py
================
Orchestrates the "student management" functional module: turns a folder
of a person's photos into a registered, trainable identity.

This module is the glue between validators -> face_encoder ->
face_recognizer -> database, and is intentionally kept free of any
OpenCV/SQLite specifics of its own (Single Responsibility Principle).
"""

import shutil
from pathlib import Path
from typing import Optional

from src import config
from src.database import Database
from src.face_encoder import FaceEncoder
from src.face_recognizer import FaceRecognizer
from src.utils.logger import get_logger
from src.utils.validators import (
    ValidationError,
    validate_directory_exists,
    validate_name,
    validate_student_id,
)

logger = get_logger(__name__)


class RegistrationError(Exception):
    pass


class RegistrationManager:
    def __init__(self, db: Optional[Database] = None,
                 encoder: Optional[FaceEncoder] = None,
                 recognizer: Optional[FaceRecognizer] = None):
        self.db = db or Database()
        self.encoder = encoder or FaceEncoder()
        self.recognizer = recognizer or FaceRecognizer()

    def register_student(self, student_id: str, name: str, photos_dir: str,
                          copy_photos: bool = True) -> dict:
        """Register a new student.

        1. Validates identity fields and photo directory.
        2. Extracts + augments face samples from the supplied photos.
        3. Persists the student record in the database (gets a label_id).
        4. Trains/updates the LBPH model incrementally with the new samples.
        5. Optionally archives the source photos under data/known_faces/<id>/.
        """
        try:
            student_id = validate_student_id(student_id)
            name = validate_name(name)
            src_dir = validate_directory_exists(photos_dir)
        except ValidationError as exc:
            raise RegistrationError(str(exc)) from exc

        if self.db.student_exists(student_id):
            raise RegistrationError(f"Student ID '{student_id}' is already registered.")

        logger.info("Building training samples for %s (%s) from %s", name, student_id, src_dir)
        samples = self.encoder.build_training_set(src_dir, augment=True)

        if len(samples) < config.MIN_TRAINING_SAMPLES:
            raise RegistrationError(
                f"Only {len(samples)} usable face samples found (need at least "
                f"{config.MIN_TRAINING_SAMPLES}). Provide clearer, front-facing photos."
            )

        # Persist student -> obtain integer label id required by LBPH
        label_id = self.db.add_student(student_id, name, sample_count=len(samples))

        labels = [label_id] * len(samples)
        self.recognizer.train(
            faces=samples,
            labels=labels,
            label_names={label_id: student_id},
            incremental=True,
        )

        if copy_photos:
            dest_dir = config.KNOWN_FACES_DIR / student_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest_dir / f.name)

        logger.info("Successfully registered %s (%s) with %d samples", name, student_id, len(samples))
        return {
            "student_id": student_id,
            "name": name,
            "label_id": label_id,
            "samples_used": len(samples),
        }

    def remove_student(self, student_id: str) -> bool:
        """Remove a student's DB record + archived photos.

        Note: the LBPH model itself is not selectively "un-trained"
        (a documented limitation) - operators should retrain from
        scratch via `cli.py retrain-all` after deletions if strict
        removal from the recognizer is required.
        """
        deleted = self.db.delete_student(student_id)
        archive_dir = config.KNOWN_FACES_DIR / student_id
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        return deleted
