from datetime import datetime

from src import config
from src.database import Database
from src.report_generator import ReportGenerator


class TestReportGenerator:
    def test_daily_report_defaults_to_today(self, temp_db: Database):
        temp_db.add_student("CS001", "Amit", 5)
        temp_db.mark_attendance("CS001", confidence=30.0)

        gen = ReportGenerator(db=temp_db)
        rows = gen.daily_report()
        assert len(rows) == 1
        assert rows[0]["student_id"] == "CS001"

    def test_summary_lists_every_student_including_absentees(self, temp_db: Database):
        temp_db.add_student("CS001", "Amit", 5)
        temp_db.add_student("CS002", "Priya", 5)
        temp_db.mark_attendance("CS001", confidence=30.0)

        gen = ReportGenerator(db=temp_db)
        rows = gen.summary()
        by_id = {r["student_id"]: r["days_present"] for r in rows}
        assert by_id == {"CS001": 1, "CS002": 0}

    def test_export_csv_writes_file(self, temp_db: Database, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "EXPORTS_DIR", tmp_path)
        temp_db.add_student("CS001", "Amit", 5)
        temp_db.mark_attendance("CS001", confidence=30.0)

        gen = ReportGenerator(db=temp_db)
        rows = gen.daily_report()
        out_path = gen.export_csv(rows, "test_export.csv")

        assert out_path.exists()
        content = out_path.read_text()
        assert "CS001" in content
        assert "Amit" in content

    def test_export_csv_handles_empty_rows(self, temp_db: Database, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "EXPORTS_DIR", tmp_path)
        gen = ReportGenerator(db=temp_db)
        out_path = gen.export_csv([], "empty.csv")
        assert out_path.exists()
        assert "No attendance records" in out_path.read_text()

    def test_attendance_percentage_excludes_weekends(self, temp_db: Database):
        temp_db.add_student("CS001", "Amit", 5)
        gen = ReportGenerator(db=temp_db)
        # A Mon-Fri work week (2026-09-14 is a Monday)
        results = gen.attendance_percentage("2026-09-14", "2026-09-18")
        assert results[0]["working_days"] == 5
