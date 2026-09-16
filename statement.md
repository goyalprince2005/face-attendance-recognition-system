# Problem Statement — Face Attendance & Recognition System

## Problem Statement

Manual, roll-call-based attendance in classrooms and workplaces is
slow, disruptive to the start of a session, and vulnerable to proxy
attendance (one person answering for an absent peer). As group sizes
grow, the time cost of manual attendance scales linearly with the
number of people present, and the resulting records are only as
trustworthy as the person taking them.

This project builds a **Face Attendance & Recognition System**: a
computer-vision pipeline that detects faces in an image (a classroom
photo, or a single captured frame) and recognizes each detected face
against a database of previously enrolled identities, automatically
logging a timestamped attendance record for every match. It applies
core Computer Vision course concepts — image preprocessing, Haar
cascade object (face) detection, feature-based face recognition
(LBPH), and dataset augmentation — to a real, everyday problem.

## Scope of the Project

**In scope:**
* Enrolling ("registering") a person from a folder of photographs.
* Detecting all faces present in a supplied image.
* Recognizing each detected face against enrolled identities, with a
  configurable confidence threshold to reject unenrolled ("Unknown")
  faces rather than force a guess.
* Automatically marking one attendance record per enrolled person per
  calendar day, with duplicate-prevention.
* Generating attendance reports: a given day, a custom date range
  (optionally filtered to one person), a per-person summary, and an
  attendance-percentage calculation over a working-day window —
  exportable to CSV.
* A fully command-line interface; no GUI is required to run any
  feature.

**Out of scope (documented limitations, not implemented):**
* Real-time continuous video-stream attendance (the system processes
  single images / frames, not a live video feed, on the reasoning that
  a single classroom photo per session is what the target users
  actually need).
* Liveness / anti-spoofing detection (e.g. detecting a printed photo
  held up to the camera) — a known limitation of LBPH-only pipelines,
  noted for future work.
* Multi-camera / multi-room deployment or a networked/cloud backend —
  the system is intentionally single-machine and file-based (SQLite)
  to keep setup and evaluation friction-free.
* Face recognition across large-scale populations (hundreds+ of
  enrolled identities); LBPH's linear-scan prediction and lack of
  metric-learning based embeddings make it best suited to
  classroom-sized cohorts (tens of people), which matches this
  project's target users below.

## Target Users

* **Course instructors / teaching assistants** who need a quick,
  auditable way to record classroom attendance from a single photo
  taken at the start of a session.
* **Small workplace teams or lab administrators** who want a
  lightweight, self-hosted attendance log without buying into a
  commercial HR/attendance SaaS product.
* **Students of this Computer Vision course**, as the direct
  beneficiary of the codebase for understanding a complete, working
  face-detection-to-recognition pipeline built from first principles
  on top of OpenCV.

## High-Level Features

1. **Student Registration** — enroll a new identity from 5+ sample
   photos; the system detects, normalizes, and augments faces before
   incrementally training the recognition model.
2. **Face Recognition & Attendance Marking** — given any image, detect
   every face, recognize known identities (or flag "Unknown"), and
   write a timestamped, duplicate-safe attendance record for each
   recognized person.
3. **Reporting & Analytics** — daily reports, custom date-range
   reports, per-student summaries, attendance-percentage calculations,
   and CSV export for all of the above.

Supporting capabilities: listing and removing enrolled students, and
an optional (non-essential) webcam-capture helper for collecting
registration photos on machines that have a camera attached.
