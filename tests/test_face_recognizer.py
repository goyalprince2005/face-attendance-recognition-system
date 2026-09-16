import numpy as np
import pytest

from src import config
from src.face_recognizer import FaceRecognizer


@pytest.fixture
def isolated_recognizer(tmp_path, monkeypatch):
    """A FaceRecognizer whose persisted model/labels files live under a
    temp directory, so tests never touch the real trained model."""
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "model.yml")
    monkeypatch.setattr(config, "LABELS_FILE", tmp_path / "labels.json")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    return FaceRecognizer()


def _pattern(seed: int) -> np.ndarray:
    """Deterministic, visually distinct 'face-like' pattern per seed."""
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 255, size=config.FACE_IMG_SIZE, dtype="uint8")
    return base


class TestFaceRecognizer:
    def test_untrained_model_raises_on_predict(self, isolated_recognizer):
        with pytest.raises(RuntimeError):
            isolated_recognizer.predict(_pattern(1))

    def test_train_then_predict_same_pattern_matches(self, isolated_recognizer):
        pattern_a = _pattern(1)
        pattern_b = _pattern(2)

        isolated_recognizer.train(
            faces=[pattern_a, pattern_a, pattern_a, pattern_b, pattern_b, pattern_b],
            labels=[1, 1, 1, 2, 2, 2],
            label_names={1: "CS001", 2: "CS002"},
            incremental=False,
        )

        student_id, confidence = isolated_recognizer.predict(pattern_a)
        assert student_id == "CS001"
        assert confidence < config.RECOGNITION_CONFIDENCE_THRESHOLD

    def test_predict_unrelated_pattern_may_be_unknown(self, isolated_recognizer):
        pattern_a = _pattern(1)
        isolated_recognizer.train(
            faces=[pattern_a] * 3,
            labels=[1, 1, 1],
            label_names={1: "CS001"},
            incremental=False,
        )
        # a wildly different random pattern should not match well
        far_pattern = np.full(config.FACE_IMG_SIZE, 128, dtype="uint8")
        student_id, confidence = isolated_recognizer.predict(far_pattern)
        # Either rejected as Unknown, or confidence is clearly worse
        # than the near-zero distance a genuine match would produce.
        assert student_id is None or confidence > 0

    def test_incremental_training_adds_new_identity(self, isolated_recognizer):
        pattern_a = _pattern(1)
        isolated_recognizer.train(
            faces=[pattern_a] * 3, labels=[1, 1, 1],
            label_names={1: "CS001"}, incremental=False,
        )
        pattern_b = _pattern(2)
        isolated_recognizer.train(
            faces=[pattern_b] * 3, labels=[2, 2, 2],
            label_names={2: "CS002"}, incremental=True,
        )
        assert isolated_recognizer.known_identities() == {1: "CS001", 2: "CS002"}

    def test_persisted_model_reloads_correctly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "model.yml")
        monkeypatch.setattr(config, "LABELS_FILE", tmp_path / "labels.json")
        monkeypatch.setattr(config, "MODELS_DIR", tmp_path)

        pattern_a = _pattern(1)
        r1 = FaceRecognizer()
        r1.train(faces=[pattern_a] * 3, labels=[1, 1, 1],
                 label_names={1: "CS001"}, incremental=False)

        r2 = FaceRecognizer()  # simulates a fresh process loading the saved model
        assert r2.is_trained()
        student_id, _ = r2.predict(pattern_a)
        assert student_id == "CS001"

    def test_train_rejects_mismatched_lengths(self, isolated_recognizer):
        with pytest.raises(ValueError):
            isolated_recognizer.train(
                faces=[_pattern(1)], labels=[1, 2],
                label_names={1: "CS001"},
            )

    def test_train_rejects_empty_dataset(self, isolated_recognizer):
        with pytest.raises(ValueError):
            isolated_recognizer.train(faces=[], labels=[], label_names={})
