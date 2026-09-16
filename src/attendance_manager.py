"""
attendance_manager.py
======================
Orchestrates the core "recognition -> attendance" functional module:
given an image (a classroom photo, or a single webcam frame captured
elsewhere), detect every face, recognize each one, and mark attendance
for recognized students.

This is the piece that ties face_detector + face_recognizer + database
together behind one simple API (`process_image`) that the CLI (and any
future GUI/API layer) can call without knowing about OpenCV internals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.database import Database
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.utils.logger import get_logger
from src.utils.validators import validate_image_path

logger = get_logger(__name__)


@dataclass
class RecognitionResult:
    box: tuple
    student_id: Optional[str]
    name: Optional[str]
    confidence: float
    attendance_marked: bool


class AttendanceManager:
    def __init__(self, db: Optional[Database] = None,
                 detector: Optional[FaceDetector] = None,
                 recognizer: Optional[FaceRecognizer] = None):
        self.db = db or Database()
        self.detector = detector or FaceDetector()
        self.recognizer = recognizer or FaceRecognizer()

    def process_image(self, image_path: str, source: str = "image",
                       mark: bool = True) -> List[RecognitionResult]:
        """Run the full pipeline on a single image file and return one
        RecognitionResult per detected face."""
        path = validate_image_path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"OpenCV could not read image: {path}")

        return self.process_frame(image, source=source, mark=mark)

    def process_frame(self, frame: np.ndarray, source: str = "webcam",
                       mark: bool = True) -> List[RecognitionResult]:
        if not self.recognizer.is_trained():
            raise RuntimeError(
                "No trained model found. Register at least one student first "
                "(see: python -m src.cli register --help)."
            )

        results: List[RecognitionResult] = []
        boxes = self.detector.detect(frame)
        logger.info("Detected %d face(s) in frame", len(boxes))

        for box in boxes:
            face = self.detector.crop_and_normalize(frame, box)
            student_id, confidence = self.recognizer.predict(face)

            name = None
            marked = False
            if student_id:
                record = self.db.get_student(student_id)
                name = record["name"] if record else student_id
                if mark:
                    marked = self.db.mark_attendance(student_id, confidence, source=source)
            else:
                logger.debug("Face at %s not recognized (confidence=%.2f)", box, confidence)

            results.append(RecognitionResult(
                box=box, student_id=student_id, name=name,
                confidence=confidence, attendance_marked=marked,
            ))

        return results

    def annotate_image(self, image_path: str, results: List[RecognitionResult],
                        output_path: str) -> str:
        """Draw bounding boxes + labels on the image for visual review and
        save it. Useful evidence for the report's screenshots section
        without requiring a live GUI window during evaluation."""
        image = cv2.imread(image_path)
        for r in results:
            x, y, w, h = r.box
            label = f"{r.name} ({r.confidence:.1f})" if r.student_id else "Unknown"
            color = (0, 200, 0) if r.student_id else (0, 0, 220)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(output_path, image)
        logger.info("Saved annotated image to %s", output_path)
        return output_path
