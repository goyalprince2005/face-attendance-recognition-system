"""
face_recognizer.py
===================
Wraps OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer.

Why LBPH instead of a deep-embedding model (FaceNet/dlib ResNet)?
  * Ships inside `opencv-contrib-python` - no GPU, no large pretrained
    weight download, so the project stays 100% reproducible offline and
    installs with a single `pip install -r requirements.txt` on any
    machine (portability / SCALABILITY of setup, an explicit goal for
    "no GUI-based setup" and easy evaluator reproduction).
  * Supports `.update()` for incremental training, so adding a new
    student does not require retraining on the entire historical dataset
    (SCALABILITY as the number of registered students grows).
  * Well documented, deterministic, and fast enough for CLI/batch use
    (PERFORMANCE requirement).

Trade-off (documented for the report): LBPH is less accurate than deep
embeddings under pose/illumination variation - mitigated with histogram
equalisation in FaceDetector and augmentation in FaceEncoder.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FaceRecognizer:
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create(
            radius=config.LBPH_RADIUS,
            neighbors=config.LBPH_NEIGHBORS,
            grid_x=config.LBPH_GRID_X,
            grid_y=config.LBPH_GRID_Y,
        )
        self._trained = False
        self._label_names: Dict[int, str] = {}
        self._load_if_exists()

    # ------------------------------------------------------------------
    def _load_if_exists(self):
        if config.MODEL_FILE.exists() and config.LABELS_FILE.exists():
            try:
                self.model.read(str(config.MODEL_FILE))
                with open(config.LABELS_FILE) as f:
                    self._label_names = {int(k): v for k, v in json.load(f).items()}
                self._trained = True
                logger.info(
                    "Loaded existing model with %d known identities",
                    len(self._label_names),
                )
            except cv2.error as exc:
                logger.error("Failed to load existing model: %s", exc)

    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------
    def train(self, faces: List[np.ndarray], labels: List[int],
              label_names: Dict[int, str], incremental: bool = True):
        """Train (or update) the LBPH model.

        Parameters
        ----------
        faces  : list of normalised grayscale face crops
        labels : parallel list of integer label ids
        label_names : mapping label_id -> human readable student_id,
                      merged into the persisted labels file.
        incremental : if True and a model already exists on disk, use
                      cv2's `.update()` so the whole dataset does not
                      need to be retrained from scratch.
        """
        if len(faces) != len(labels):
            raise ValueError("faces and labels must be the same length")
        if not faces:
            raise ValueError("Cannot train on an empty dataset")

        X = np.array(faces)
        y = np.array(labels)

        if incremental and self._trained:
            self.model.update(X, y)
            logger.info("Incrementally updated model with %d new samples", len(faces))
        else:
            self.model.train(X, y)
            self._trained = True
            logger.info("Trained model from scratch on %d samples", len(faces))

        self._label_names.update(label_names)
        self._persist()

    def _persist(self):
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model.write(str(config.MODEL_FILE))
        with open(config.LABELS_FILE, "w") as f:
            json.dump(self._label_names, f, indent=2)
        logger.debug("Persisted model to %s", config.MODEL_FILE)

    # ------------------------------------------------------------------
    def predict(self, face: np.ndarray) -> Tuple[Optional[str], float]:
        """Predict the student_id for a single normalised face crop.

        Returns (student_id_or_None, confidence). A returned student_id
        of None means the face did not match any known identity within
        the configured confidence threshold ("Unknown").
        Note: LBPH confidence is a *distance* metric - lower is better.
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained yet. Run 'train' first.")

        label_id, distance = self.model.predict(face)

        if distance > config.RECOGNITION_CONFIDENCE_THRESHOLD:
            return None, distance

        student_id = self._label_names.get(label_id) or self._label_names.get(str(label_id))
        return student_id, distance

    def known_identities(self) -> Dict[int, str]:
        return dict(self._label_names)
