# Testing Notes — Recognition Confidence Threshold & Detector Recall

These notes record two empirical investigations carried out during
development (also summarised in Section 11 "Challenges Faced" of the
project report). They are kept here as raw working notes so the
reasoning is traceable, not just the final numbers.

## 1. Recognition confidence threshold

LBPH's `predict()` returns a **distance** (lower = more confident). A
commonly cited default in OpenCV tutorials is a threshold of `70.0` for
"accept as a match".

With the two demo students enrolled (`data/samples/`), predicting on
three held-out test images gave:

| Test image | True identity | Predicted label | Distance |
|---|---|---|---|
| `amit_test.jpg` | Amit Kumar (enrolled) | Amit Kumar | 37.81 |
| `priya_test.jpg` | Priya Singh (enrolled) | Priya Singh | 26.14 |
| `stranger_test.jpg` | Unenrolled subject | **Amit Kumar (wrong!)** | 48.13 |

At the tutorial-default threshold of 70.0, the stranger's face
(distance 48.13) was incorrectly accepted as a match for Amit Kumar,
because 48.13 < 70.0.

**Fix**: since both genuine matches scored well under 40 and the false
match scored just under 50, the threshold was lowered to `45.0`
(`src/config.py -> RECOGNITION_CONFIDENCE_THRESHOLD`). Re-running the
same three images:

| Test image | Predicted | Distance | Correct? |
|---|---|---|---|
| `amit_test.jpg` | Amit Kumar | 37.81 | Yes |
| `priya_test.jpg` | Priya Singh | 26.14 | Yes |
| `stranger_test.jpg` | **Unknown** | 48.13 | Yes |

**Caveat for future users of this project**: this threshold was tuned
against a population of exactly two enrolled identities. As more
students are registered, re-run this same experiment (enroll everyone,
then test against a few deliberately unenrolled photos) and adjust
`RECOGNITION_CONFIDENCE_THRESHOLD` if false accepts/rejects appear -
there is no universal "correct" value, only one that fits the current
enrolled population and photo quality.

## 2. Haar cascade recall on low-resolution images

The bundled demo dataset's source images are only 92x112 pixels. Run
directly through `cv2.CascadeClassifier.detectMultiScale` with
`minSize=(60,60)`, only 10 of 18 sample images (~55%) had a face
detected at all - the minimum-size window was too close to the whole
image size for the cascade's multi-scale search to work reliably.

**Fix**: `FaceDetector.detect()` now upscales any input image narrower
than `config.DETECTOR_MIN_UPSCALE_WIDTH` (450px) using cubic
interpolation before running the cascade, then maps detected boxes back
to the original image's coordinate space. Re-running the same 18 images
after this change: 14/18 (~78%) detected. The remaining misses are
images with an extreme head tilt, a known limitation of the frontal-face
Haar cascade (documented in `README.md` Section 9 and the report's
"Design Decisions & Rationale" section).

## How to reproduce these numbers

```bash
python3 -m src.cli register --id CS2023001 --name "Amit Kumar" --photos data/samples/student_amit_kumar
python3 -m src.cli register --id CS2023002 --name "Priya Singh" --photos data/samples/student_priya_singh
python3 -m src.cli recognize --image data/samples/classroom_test/amit_test.jpg
python3 -m src.cli recognize --image data/samples/classroom_test/priya_test.jpg
python3 -m src.cli recognize --image data/samples/classroom_test/stranger_test.jpg
```
