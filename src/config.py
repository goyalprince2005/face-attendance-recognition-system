"""
config.py
=========
Centralised, single-source-of-truth configuration for the whole system.

Keeping every tunable value in one module (instead of scattering magic
numbers across files) is what makes the system MAINTAINABLE and lets
non-functional requirements such as PERFORMANCE and RESOURCE EFFICIENCY
be tuned without touching business logic.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"      # raw registration images, per student
SAMPLES_DIR = DATA_DIR / "samples"              # sample images used for demo/testing

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
EXPORTS_DIR = BASE_DIR / "exports"              # generated CSV / PDF attendance reports

DB_PATH = BASE_DIR / "attendance.db"
MODEL_FILE = MODELS_DIR / "lbph_model.yml"
LABELS_FILE = MODELS_DIR / "labels.json"

for _dir in (DATA_DIR, KNOWN_FACES_DIR, SAMPLES_DIR, MODELS_DIR, LOGS_DIR, EXPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Face detection / recognition parameters
# ---------------------------------------------------------------------------
HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"

# Detector tuning (OpenCV Haar cascade)
DETECTOR_SCALE_FACTOR = 1.05
DETECTOR_MIN_NEIGHBORS = 3
DETECTOR_MIN_SIZE = (60, 60)

# Small input images (e.g. old low-resolution datasets, thumbnails) are
# up-scaled so the cascade - which needs enough pixel detail per feature
# window - has a fair chance of finding a face. This trades a little CPU
# time for materially better recall (PERFORMANCE vs RELIABILITY trade-off,
# documented in the report).
DETECTOR_MIN_UPSCALE_WIDTH = 450

# Recognition (LBPH) parameters
FACE_IMG_SIZE = (200, 200)          # every cropped face is normalised to this size
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8

# Confidence threshold: LBPH returns a *distance* (lower = more confident).
# Anything above this threshold is treated as "Unknown" -> protects against
# false-positive attendance marking (SECURITY requirement).
# Empirically tuned (see docs/testing_notes.md): with a small enrolled
# population, LBPH distances for a genuine match were consistently below
# ~40 while an unenrolled ("stranger") face produced ~48+. 45.0 gives a
# safety margin on the "accept" side without needing dozens of registered
# identities to calibrate against.
RECOGNITION_CONFIDENCE_THRESHOLD = 45.0

# Minimum number of valid face samples required before a student can be
# registered - avoids training on too little data (RELIABILITY).
MIN_TRAINING_SAMPLES = 3

# ---------------------------------------------------------------------------
# Attendance rules
# ---------------------------------------------------------------------------
# Once a student is marked present for a given calendar date, further
# recognitions on the same date are ignored (prevents duplicate rows).
ALLOW_MULTIPLE_MARKS_PER_DAY = False

DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = LOGS_DIR / "app.log"
LOG_LEVEL = os.environ.get("FARS_LOG_LEVEL", "INFO")
