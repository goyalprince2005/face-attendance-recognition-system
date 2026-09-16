"""
face_encoder.py
================
Responsible for turning a folder of raw registration photos into a
training dataset: normalised, grayscale face crops + integer labels,
ready to be consumed by face_recognizer.FaceRecognizer.train().

Kept separate from face_detector (single detection responsibility) and
from face_recognizer (single recognition/training responsibility) so
each module can be unit-tested and modified independently -
MODULAR / CLEAN IMPLEMENTATION as required by the brief.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src import config
from src.face_detector import FaceDetector
from src.utils.logger import get_logger
from src.utils.validators import list_image_files

logger = get_logger(__name__)


class FaceEncoder:
    def __init__(self, detector: FaceDetector = None):
        self.detector = detector or FaceDetector()

    def encode_directory(self, directory: Path) -> List[np.ndarray]:
        """Read every image in *directory*, detect the (single) face in
        each, normalise it, and return the list of face samples.

        Images where zero or more-than-one face is found are skipped
        with a warning (ERROR HANDLING - bad registration photos should
        not silently corrupt the training set).
        """
        samples: List[np.ndarray] = []
        image_files = list_image_files(directory)

        if not image_files:
            raise ValueError(f"No valid images found in {directory}")

        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning("Could not read image (corrupt/unsupported): %s", img_path)
                continue

            boxes = self.detector.detect(image)
            if len(boxes) == 0:
                logger.warning("No face detected in %s - skipping", img_path.name)
                continue
            if len(boxes) > 1:
                logger.warning(
                    "%s contains %d faces - using the largest face only",
                    img_path.name, len(boxes),
                )
                boxes = [max(boxes, key=lambda b: b[2] * b[3])]

            face = self.detector.crop_and_normalize(image, boxes[0])
            samples.append(face)
            logger.debug("Encoded sample from %s", img_path.name)

        return samples

    @staticmethod
    def augment(face: np.ndarray) -> List[np.ndarray]:
        """Cheap data augmentation (horizontal flip + slight brightness
        jitter) so that students with very few photos still yield a
        usable number of training samples. Improves RELIABILITY of the
        trained model without requiring a webcam capture session."""
        flipped = cv2.flip(face, 1)
        brighter = cv2.convertScaleAbs(face, alpha=1.1, beta=15)
        darker = cv2.convertScaleAbs(face, alpha=0.9, beta=-15)
        return [face, flipped, brighter, darker]

    def build_training_set(self, directory: Path, augment: bool = True) -> List[np.ndarray]:
        samples = self.encode_directory(directory)
        if not augment:
            return samples

        augmented: List[np.ndarray] = []
        for s in samples:
            augmented.extend(self.augment(s))
        return augmented
