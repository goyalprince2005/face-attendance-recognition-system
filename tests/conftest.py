"""
conftest.py
===========
Shared pytest fixtures. Every fixture that touches the filesystem/DB
points at a temporary directory so tests never mutate the real
attendance.db / models used for actual demo/registration data.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.database import Database


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """A fresh, isolated Database instance backed by a temp SQLite file."""
    db_path = tmp_path / "test_attendance.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    return Database(db_path=db_path)


@pytest.fixture
def sample_face():
    """A deterministic, correctly-shaped fake 'face' array (not a real
    photo) - sufficient for exercising the recognizer/encoder logic
    without depending on real image files."""
    import numpy as np
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, size=config.FACE_IMG_SIZE, dtype="uint8")
