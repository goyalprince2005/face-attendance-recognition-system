"""
cli.py
======
Single command-line entry point for the whole system.

The system supports:
    - Student registration
    - Student removal
    - Student listing
    - Image-based face recognition
    - Live webcam face recognition
    - Webcam registration-photo capture
    - Attendance reports

Usage examples
--------------
    python -m src.cli register --id CS001 --name "Aditi Sharma" --photos data/samples/aditi
    python -m src.cli list-students
    python -m src.cli recognize --image data/samples/classroom1.jpg --annotate out.jpg
    python -m src.cli webcam
    python -m src.cli webcam-capture --out data/samples/aditi --count 15
    python -m src.cli report --today
    python -m src.cli report --range 2026-09-01 2026-09-15 --export attendance.csv
    python -m src.cli remove-student --id CS001
"""

import argparse
import sys

from src.attendance_manager import AttendanceManager
from src.database import Database
from src.registration import RegistrationError, RegistrationManager
from src.report_generator import ReportGenerator
from src.utils.logger import get_logger
from src.utils.validators import ValidationError

logger = get_logger(__name__)


def cmd_register(args):
    manager = RegistrationManager()

    try:
        result = manager.register_student(
            args.id,
            args.name,
            args.photos,
            copy_photos=not args.no_copy,
        )
    except RegistrationError as exc:
        print(f"[ERROR] Registration failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Registered '{result['name']}' ({result['student_id']}) "
        f"using {result['samples_used']} training samples."
    )

    return 0


def cmd_remove_student(args):
    manager = RegistrationManager()
    removed = manager.remove_student(args.id)

    if removed:
        print(f"Removed student '{args.id}'.")
        return 0

    print(
        f"[ERROR] Student '{args.id}' was not found.",
        file=sys.stderr,
    )
    return 1


def cmd_list_students(args):
    db = Database()
    students = db.list_students()

    if not students:
        print("No students registered yet.")
        return 0

    print(
        f"{'Student ID':<15}"
        f"{'Name':<25}"
        f"{'Samples':<10}"
        f"{'Registered On'}"
    )
    print("-" * 70)

    for s in students:
        print(
            f"{s['student_id']:<15}"
            f"{s['name']:<25}"
            f"{s['sample_count']:<10}"
            f"{s['created_at'][:19]}"
        )

    return 0


def cmd_recognize(args):
    manager = AttendanceManager()

    try:
        results = manager.process_image(
            args.image,
            source=args.source,
            mark=not args.no_mark,
        )
    except (ValidationError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No faces detected in the supplied image.")
        return 0

    for r in results:
        status = "Unknown"

        if r.student_id:
            status = f"{r.name} ({r.student_id})"

            if r.attendance_marked:
                status += " -> attendance marked"
            else:
                status += " -> already marked today"

        print(
            f"Face at {r.box}: {status} "
            f"| confidence(distance)={r.confidence:.2f}"
        )

    if args.annotate:
        manager.annotate_image(
            args.image,
            results,
            args.annotate,
        )

        print(
            f"Annotated image saved to: {args.annotate}"
        )

    return 0


def cmd_webcam_capture(args):
    """
    Capture registration photos from a webcam.

    This command is intended for collecting training images.
    It does NOT perform face recognition or attendance.
    """
    import cv2
    from pathlib import Path

    out_dir = Path(args.out)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(
            "[ERROR] No webcam available.",
            file=sys.stderr,
        )
        return 1

    saved = 0

    print(
        f"Capturing {args.count} frames."
    )
    print(
        "Press Ctrl+C to stop early."
    )

    try:
        while saved < args.count:
            ok, frame = cap.read()

            if not ok:
                print(
                    "[ERROR] Could not read frame from webcam.",
                    file=sys.stderr,
                )
                break

            path = out_dir / f"frame_{saved:03d}.jpg"

            cv2.imwrite(
                str(path),
                frame,
            )

            saved += 1

    finally:
        cap.release()

    print(
        f"Saved {saved} frame(s) to {out_dir}"
    )

    return 0


def cmd_webcam(args):
    """
    Run live face recognition and attendance using the webcam.

    The webcam continuously captures frames. Each frame is passed
    through the existing AttendanceManager pipeline:

        Webcam
          ↓
        Face Detection
          ↓
        Face Recognition
          ↓
        Attendance Database
          ↓
        OpenCV Display
    """
    import cv2

    manager = AttendanceManager()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(
            "[ERROR] Could not open webcam.",
            file=sys.stderr,
        )
        return 1

    print("=" * 60)
    print("Face Attendance & Recognition System")
    print("=" * 60)
    print("Live webcam recognition started.")
    print("Press 'Q' to quit.")
    print("=" * 60)

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print(
                    "[ERROR] Could not read frame from webcam.",
                    file=sys.stderr,
                )
                break

            try:
                results = manager.process_frame(
                    frame,
                    source="webcam",
                    mark=True,
                )

            except RuntimeError as exc:
                print(
                    f"[ERROR] {exc}",
                    file=sys.stderr,
                )
                break

            # Draw recognition results on the live frame
            for r in results:

                x, y, w, h = r.box

                # -----------------------------
                # Known student
                # -----------------------------
                if r.student_id:

                    label = (
                        f"{r.name} | {r.student_id}"
                    )

                    color = (0, 200, 0)

                    if r.attendance_marked:
                        status = "Attendance marked"
                    else:
                        status = "Already marked today"

                    # Status below bounding box
                    cv2.putText(
                        frame,
                        status,
                        (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                # -----------------------------
                # Unknown person
                # -----------------------------
                else:

                    label = "Unknown"

                    color = (0, 0, 220)

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2,
                )

                # Student name / Unknown label
                cv2.putText(
                    frame,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # Recognition confidence
                confidence_text = (
                    f"Distance: {r.confidence:.2f}"
                )

                cv2.putText(
                    frame,
                    confidence_text,
                    (x, y + h + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

            # Window title
            cv2.imshow(
                "Face Attendance & Recognition System",
                frame,
            )

            # Press Q to quit
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Webcam recognition stopped.")

    return 0


def cmd_report(args):
    gen = ReportGenerator()

    if args.today:
        rows = gen.daily_report()

    elif args.range:
        start, end = args.range

        rows = gen.range_report(
            start,
            end,
            student_id=args.student,
        )

    elif args.summary:
        rows = gen.summary()

    elif args.percentage:
        start, end = args.percentage

        rows = gen.attendance_percentage(
            start,
            end,
        )

    else:
        rows = gen.daily_report()

    if rows:
        columns = list(rows[0].keys())

        gen.print_table(
            rows,
            columns,
        )

    else:
        print(
            "No records found for the given filters."
        )

    if args.export:
        path = gen.export_csv(
            rows,
            args.export,
        )

        print(
            f"\nExported report to: {path}"
        )

    return 0


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="face-attendance",
        description="Face Attendance & Recognition System - CLI",
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ==========================================================
    # REGISTER
    # ==========================================================

    p_register = sub.add_parser(
        "register",
        help="Register a new student from a folder of photos",
    )

    p_register.add_argument(
        "--id",
        required=True,
        help="Unique student ID, e.g. CS2023001",
    )

    p_register.add_argument(
        "--name",
        required=True,
        help="Full name",
    )

    p_register.add_argument(
        "--photos",
        required=True,
        help="Path to a folder containing the student's photos",
    )

    p_register.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not archive photos under data/known_faces",
    )

    p_register.set_defaults(
        func=cmd_register
    )

    # ==========================================================
    # REMOVE STUDENT
    # ==========================================================

    p_remove = sub.add_parser(
        "remove-student",
        help="Delete a registered student and their attendance history",
    )

    p_remove.add_argument(
        "--id",
        required=True,
    )

    p_remove.set_defaults(
        func=cmd_remove_student
    )

    # ==========================================================
    # LIST STUDENTS
    # ==========================================================

    p_list = sub.add_parser(
        "list-students",
        help="List all registered students",
    )

    p_list.set_defaults(
        func=cmd_list_students
    )

    # ==========================================================
    # IMAGE RECOGNITION
    # ==========================================================

    p_recognize = sub.add_parser(
        "recognize",
        help="Run face recognition on an image and mark attendance",
    )

    p_recognize.add_argument(
        "--image",
        required=True,
        help="Path to the image to process",
    )

    p_recognize.add_argument(
        "--source",
        default="image",
        help="Label recorded for the attendance source",
    )

    p_recognize.add_argument(
        "--no-mark",
        action="store_true",
        help="Only recognize, do not write attendance",
    )

    p_recognize.add_argument(
        "--annotate",
        help="Optional path to save an annotated copy of the image",
    )

    p_recognize.set_defaults(
        func=cmd_recognize
    )

    # ==========================================================
    # WEBCAM CAPTURE
    # ==========================================================

    p_webcam_capture = sub.add_parser(
        "webcam-capture",
        help="Capture registration photos from a live webcam",
    )

    p_webcam_capture.add_argument(
        "--out",
        required=True,
        help="Directory to save captured frames",
    )

    p_webcam_capture.add_argument(
        "--count",
        type=int,
        default=15,
        help="Number of frames to capture",
    )

    p_webcam_capture.set_defaults(
        func=cmd_webcam_capture
    )

    # ==========================================================
    # LIVE WEBCAM RECOGNITION
    # ==========================================================

    p_webcam = sub.add_parser(
        "webcam",
        help="Run live face recognition and attendance using webcam",
    )

    p_webcam.set_defaults(
        func=cmd_webcam
    )

    # ==========================================================
    # REPORT
    # ==========================================================

    p_report = sub.add_parser(
        "report",
        help="Generate attendance reports",
    )

    p_report.add_argument(
        "--today",
        action="store_true",
        help="Show today's attendance",
    )

    p_report.add_argument(
        "--range",
        nargs=2,
        metavar=("START", "END"),
        help="YYYY-MM-DD YYYY-MM-DD",
    )

    p_report.add_argument(
        "--student",
        help="Filter by student ID (used with --range)",
    )

    p_report.add_argument(
        "--summary",
        action="store_true",
        help="Total days present per student",
    )

    p_report.add_argument(
        "--percentage",
        nargs=2,
        metavar=("START", "END"),
        help="Attendance %% per student over a range",
    )

    p_report.add_argument(
        "--export",
        help="Export the resulting rows to a CSV file under exports/",
    )

    p_report.set_defaults(
        func=cmd_report
    )

    return parser


def main(argv=None) -> int:

    parser = build_parser()

    args = parser.parse_args(argv)

    try:
        return args.func(args)

    except Exception as exc:
        logger.exception(
            "Unhandled error while running command '%s'",
            args.command,
        )

        print(
            f"[FATAL] {exc}",
            file=sys.stderr,
        )

        return 2


if __name__ == "__main__":
    sys.exit(main())