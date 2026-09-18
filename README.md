# Face Attendance & Recognition System

A command-line computer vision application that automates attendance
by detecting and recognizing human faces in photographs. Built for the
**Computer Vision** course "Build Your Own Project" evaluation.

No GUI is required at any point — every feature is a subcommand of a
single CLI (`python -m src.cli ...`) and can be run over SSH, in CI, or
on a headless server.

---

## 1. Overview

Traditional roll-call attendance is slow and easy to falsify (proxy
attendance). This project automates it: an instructor supplies a photo
(a classroom picture, or a single webcam frame captured earlier), and
the system detects every face in it, recognizes which registered
student each face belongs to, and logs their attendance — automatically
skipping students already marked present that day and flagging
unrecognized faces as "Unknown" instead of guessing.

## 2. Features

| Module | What it does |
|---|---|
| **Student Registration** | Enroll a new student from a folder of 5+ photos. Detects the face in each photo, augments the dataset (flip / brightness jitter), and trains an LBPH face-recognition model incrementally. |
| **Face Recognition & Attendance** | Run recognition on any image. Every detected face is matched against enrolled students (or labeled "Unknown"); a match automatically writes a timestamped attendance record and produces an annotated image for visual proof. |
| **Reporting & Analytics** | Daily attendance, attendance over a custom date range (optionally filtered per student), a per-student summary of total days present, attendance-percentage calculation over a working-day window, and CSV export of any of the above. |

Supporting capabilities: student removal, listing all registered
students, an **optional** webcam-capture helper for registration
photos (falls back gracefully with a clear error if no camera is
present — the guaranteed, camera-free workflow is folder-based
registration).

## 3. Technology Stack

* **Python 3.10+**
* **OpenCV (`opencv-contrib-python`)** — Haar-cascade face detection +
  LBPH (Local Binary Patterns Histograms) face recognition
* **SQLite** (Python standard library `sqlite3`) — zero-config,
  file-based persistence for students & attendance records
* **pytest** — automated unit + integration tests
* **argparse** — CLI

No cloud services, no GPU, and no external model downloads are
required — everything ships inside `opencv-contrib-python` or Python's
standard library, so `pip install -r requirements.txt` is the entire
setup.

## 4. Project Structure

```
face_attendance_system/
├── src/
│   ├── cli.py                 # single command-line entry point
│   ├── config.py              # all tunable settings in one place
│   ├── database.py            # SQLite persistence layer
│   ├── face_detector.py       # Haar-cascade face detection
│   ├── face_encoder.py        # photo folder -> training samples
│   ├── face_recognizer.py     # LBPH training / prediction
│   ├── registration.py        # student enrollment workflow
│   ├── attendance_manager.py  # recognition -> attendance pipeline
│   ├── report_generator.py    # reports, CSV export, statistics
│   └── utils/
│       ├── logger.py          # console + rotating file logging
│       └── validators.py      # input validation helpers
├── tests/                     # 41 pytest unit/integration tests
├── data/
│   ├── known_faces/           # archived enrollment photos (auto-created)
│   └── samples/                # demo dataset used below (see its README.md)
├── docs/diagrams/             # architecture, workflow, UML, ER diagrams
├── models/                    # trained LBPH model + label map (auto-created)
├── exports/                   # generated CSV reports (auto-created)
├── logs/                      # rotating application log (auto-created)
├── requirements.txt
├── statement.md
└── README.md                  # you are here
```

## 5. Environment Setup

### 5.1 Prerequisites
* Python **3.10 or newer** (`python3 --version`)
* `pip` (comes with Python)
* No camera/webcam required — the demo below uses still images.

### 5.2 Clone and install

```bash
git clone https://github.com/<github-username>/<repo-name>.git
cd <repo-name>

# (Recommended) create an isolated virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-contrib-python==4.10.0.84
numpy==1.26.4
pytest==8.3.3
```

### 5.3 Verify the installation

```bash
python3 -m src.cli --help
```

You should see the list of subcommands (`register`, `recognize`,
`report`, `list-students`, `remove-student`, `webcam-capture`).

## 6. Quick Demo (fully reproducible, no camera needed)

The repository ships with a small, ready-to-use demo dataset under
`data/samples/` — see `data/samples/README.md` for exactly what it is
and where it came from (anonymised academic face-recognition benchmark
photos, relabeled with fictional student names — **not** photos of any
real, identifiable person).

```bash
# 1. Register two students from their photo folders
python3 -m src.cli register --id CS2023001 --name "Amit Kumar" \
    --photos data/samples/student_amit_kumar

python3 -m src.cli register --id CS2023002 --name "Priya Singh" \
    --photos data/samples/student_priya_singh

# 2. Confirm they were enrolled
python3 -m src.cli list-students

# 3. Run recognition on images the model has NEVER seen before
python3 -m src.cli recognize --image data/samples/classroom_test/amit_test.jpg \
    --annotate exports/amit_result.jpg

python3 -m src.cli recognize --image data/samples/classroom_test/priya_test.jpg \
    --annotate exports/priya_result.jpg

# An unenrolled person is correctly reported as "Unknown"
python3 -m src.cli recognize --image data/samples/classroom_test/stranger_test.jpg \
    --annotate exports/stranger_result.jpg

# 4. View today's attendance and export it to CSV
python3 -m src.cli report --today --export daily_attendance.csv
cat exports/daily_attendance.csv
```

Expected recognition output (yours will match, since the demo data and
threshold are fixed):

```
Face at (48, 18, 343, 343): Amit Kumar (CS2023001) -> attendance marked | confidence(distance)=37.81
Face at (28, 68, 371, 372): Priya Singh (CS2023002) -> attendance marked | confidence(distance)=26.14
Face at (45, 103, 335, 335): Unknown | confidence(distance)=48.13
```

Open `exports/amit_result.jpg` / `priya_result.jpg` / `stranger_result.jpg`
to see the bounding boxes and labels drawn on the images.

## 7. Using the System With Your Own Photos

```bash
mkdir -p data/known_faces/raw/your_name
# copy 5-10 clear, front-facing JPG/PNG photos of yourself into that folder

python3 -m src.cli register --id <unique-id> --name "Your Name" \
    --photos data/known_faces/raw/your_name

python3 -m src.cli recognize --image path/to/a/new/photo.jpg
```

If a webcam is physically attached to the machine you're running on,
you can optionally capture registration frames directly:

```bash
python3 -m src.cli webcam-capture --out data/known_faces/raw/your_name --count 15
```

This is entirely optional; if no camera is detected the command prints
a clear error and exits — it never blocks the rest of the CLI.

## 8. Full Command Reference

```bash
# Registration
python3 -m src.cli register --id <id> --name "<name>" --photos <dir> [--no-copy]
python3 -m src.cli remove-student --id <id>
python3 -m src.cli list-students

# Recognition / Attendance
python3 -m src.cli recognize --image <path> [--source <label>] [--no-mark] [--annotate <out.jpg>]

# Reporting
python3 -m src.cli report --today [--export <file.csv>]
python3 -m src.cli report --range 2026-09-01 2026-09-15 [--student <id>] [--export <file.csv>]
python3 -m src.cli report --summary [--export <file.csv>]
python3 -m src.cli report --percentage 2026-09-01 2026-09-15 [--export <file.csv>]
```

## 9. Running the Automated Tests

```bash
pip install pytest   # already included in requirements.txt
python3 -m pytest tests/ -v
```

41 tests cover input validation, database CRUD, face detection
(including small-image upscaling), LBPH training/prediction
(training-from-scratch, incremental updates, model
persistence/reload), end-to-end registration against real sample
photos, and report generation (including CSV export and
weekend-excluded attendance-percentage math).

```
============================== 41 passed in ~4s ==============================
```

## 10. Configuration

All tunable parameters (detector sensitivity, recognition confidence
threshold, minimum training samples, attendance duplicate-prevention
rule, log level, file paths) live in **`src/config.py`** — there is no
need to touch any other file to retune the system.

## 11. Design Notes, Diagrams & Known Limitations

* Architecture, workflow, UML (use case / class / sequence) and ER
  diagrams are in `docs/diagrams/`.
* **Why Haar cascade + LBPH instead of a deep-learning pipeline
  (dlib/FaceNet)?** They ship inside `opencv-contrib-python` with zero
  extra model downloads, run fast on CPU, and support incremental
  training — ideal for a fully offline, easily reproducible CLI project.
  The documented trade-off is lower accuracy under extreme pose/lighting
  variation and on very small enrolled populations; see
  `statement.md` and the project report for the full discussion,
  including a real false-positive we found and fixed by tuning
  `RECOGNITION_CONFIDENCE_THRESHOLD`.
* The LBPH model is not selectively "un-trained" when a student is
  removed — the documented workaround is to re-register the remaining
  students from scratch if strict removal from the recognizer itself is
  required (the student's database record and attendance history are
  deleted immediately either way).

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: cv2` | Run `pip install -r requirements.txt` inside the active virtual environment. |
| `No trained model found` on `recognize` | Register at least one student first. |
| `No face detected` during registration | Use clearer, front-facing, well-lit photos; avoid heavy occlusion/extreme angles. |
| Low recognition accuracy | Register more/varied photos per student, or lower `RECOGNITION_CONFIDENCE_THRESHOLD` in `src/config.py` cautiously (this raises false-accept risk). |

# 13. Author

Prince Goyal (24BAI10623)
