"""
face_detector.py
=================
Wraps OpenCV's Haar-cascade based face detector.

Design rationale (documented here so the report's "Design Decisions &
Rationale" section can reference it): Haar cascades were chosen over a
deep-learning detector (e.g. MTCNN/DNN-SSD) because they:
  * ship inside opencv-python with no extra model download,
  * run fast on CPU-only machines (PERFORMANCE requirement),
  * are more than adequate for a controlled attendance-capture scenario
    (frontal, reasonably lit faces).
The trade-off (documented as a limitation) is lower recall on extreme
angles/occlusion compared to modern DNN detectors - acceptable for this
academic project's scope.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

BoundingBox = Tuple[int, int, int, int]  # x, y, w, h


class FaceDetector:
    def __init__(self):
        cascade_path = Path(cv2.data.haarcascades) / config.HAAR_CASCADE_FILE
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from {cascade_path}. "
                "Ensure opencv-python is correctly installed."
            )
        logger.debug("Loaded Haar cascade from %s", cascade_path)

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect faces in a BGR image. Returns a list of (x, y, w, h)
        in the ORIGINAL image's coordinate space."""
        if image is None:
            raise ValueError("detect() received an empty image")

        h, w = image.shape[:2]
        scale = 1.0
        working_image = image
        if w < config.DETECTOR_MIN_UPSCALE_WIDTH:
            scale = config.DETECTOR_MIN_UPSCALE_WIDTH / float(w)
            working_image = cv2.resize(
                image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
            )

        gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improves robustness to lighting variance

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=config.DETECTOR_SCALE_FACTOR,
            minNeighbors=config.DETECTOR_MIN_NEIGHBORS,
            minSize=config.DETECTOR_MIN_SIZE,
        )

        # Map boxes back to the original image's coordinate space.
        results = []
        for (x, y, bw, bh) in faces:
            results.append((
                int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)
            ))
        return results

    def crop_and_normalize(self, image: np.ndarray, box: BoundingBox) -> np.ndarray:
        """Crop the face region, convert to grayscale, and resize to the
        fixed dimensions the LBPH recognizer expects."""
        x, y, w, h = box
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, config.FACE_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        face = cv2.equalizeHist(face)
        return face

    def extract_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Convenience: detect + crop + normalize every face in one call."""
        boxes = self.detect(image)
        return [self.crop_and_normalize(image, box) for box in boxes]
