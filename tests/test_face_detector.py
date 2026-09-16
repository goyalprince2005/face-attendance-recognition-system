import numpy as np

from src.face_detector import FaceDetector


class TestFaceDetector:
    def test_loads_cascade_successfully(self):
        detector = FaceDetector()
        assert detector.cascade is not None

    def test_detect_on_blank_image_finds_nothing(self):
        detector = FaceDetector()
        blank = np.zeros((300, 300, 3), dtype="uint8")
        boxes = detector.detect(blank)
        assert boxes == []

    def test_detect_upscales_small_images(self):
        """A very small image should still be processed without raising,
        exercising the auto-upscale path used for low-resolution inputs."""
        detector = FaceDetector()
        tiny = np.random.randint(0, 255, size=(92, 112, 3), dtype="uint8")
        boxes = detector.detect(tiny)
        assert isinstance(boxes, list)

    def test_crop_and_normalize_returns_configured_size(self):
        from src import config
        detector = FaceDetector()
        image = np.random.randint(0, 255, size=(300, 300, 3), dtype="uint8")
        face = detector.crop_and_normalize(image, (50, 50, 100, 100))
        assert face.shape == config.FACE_IMG_SIZE

    def test_detect_raises_on_none_input(self):
        detector = FaceDetector()
        try:
            detector.detect(None)
            assert False, "expected ValueError"
        except ValueError:
            pass
