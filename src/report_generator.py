"""
report_generator.py
====================
The "reporting / analytics" functional module. Turns raw attendance rows
from the database into human-usable outputs: a formatted console table
and an exportable CSV file, plus simple aggregate statistics
(attendance percentage per student over a date range).

Kept separate from database.py: database.py only knows how to fetch
rows; this module knows how to *present* them.
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src import config
from src.database import Database
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    def daily_report(self, date: Optional[str] = None):
        date = date or datetime.now().strftime(config.DATE_FORMAT)
        rows = self.db.get_attendance(start_date=date, end_date=date)
        return rows

    def range_report(self, start_date: str, end_date: str, student_id: Optional[str] = None):
        return self.db.get_attendance(student_id=student_id, start_date=start_date, end_date=end_date)

    def summary(self):
        """Per-student total number of days present."""
        return self.db.attendance_summary()

    def attendance_percentage(self, start_date: str, end_date: str):
        """Compute each student's attendance % over a working-day window
        (weekends excluded) between start_date and end_date inclusive."""
        start = datetime.strptime(start_date, config.DATE_FORMAT)
        end = datetime.strptime(end_date, config.DATE_FORMAT)
        working_days = 0
        d = start
        while d <= end:
            if d.weekday() < 5:  # Mon-Fri
                working_days += 1
            d += timedelta(days=1)
        working_days = max(working_days, 1)

        students = self.db.list_students()
        results = []
        for s in students:
            rows = self.db.get_attendance(student_id=s["student_id"],
                                           start_date=start_date, end_date=end_date)
            present_days = len({r["date"] for r in rows})
            pct = round((present_days / working_days) * 100, 2)
            results.append({
                "student_id": s["student_id"],
                "name": s["name"],
                "present_days": present_days,
                "working_days": working_days,
                "attendance_pct": pct,
            })
        return sorted(results, key=lambda r: r["attendance_pct"], reverse=True)

    def export_csv(self, rows, filename: str) -> Path:
        out_path = config.EXPORTS_DIR / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="") as f:
            if not rows:
                f.write("No attendance records found for the given filters.\n")
                logger.warning("Exported empty report to %s", out_path)
                return out_path

            writer = csv.writer(f)
            writer.writerow(rows[0].keys())
            for row in rows:
                writer.writerow([row[k] for k in row.keys()])

        logger.info("Exported %d rows to %s", len(rows), out_path)
        return out_path

    @staticmethod
    def print_table(rows, columns):
        if not rows:
            print("No records found.")
            return
        widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
        header = " | ".join(c.ljust(widths[c]) for c in columns)
        print(header)
        print("-" * len(header))
        for r in rows:
            print(" | ".join(str(r[c]).ljust(widths[c]) for c in columns))
