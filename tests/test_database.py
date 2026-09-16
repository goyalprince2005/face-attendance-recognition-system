from src.database import Database


class TestStudentManagement:
    def test_add_and_retrieve_student(self, temp_db: Database):
        label_id = temp_db.add_student("CS001", "Amit Kumar", sample_count=10)
        assert label_id == 1

        student = temp_db.get_student("CS001")
        assert student["name"] == "Amit Kumar"
        assert student["sample_count"] == 10

    def test_label_ids_increment(self, temp_db: Database):
        l1 = temp_db.add_student("CS001", "Amit", 5)
        l2 = temp_db.add_student("CS002", "Priya", 5)
        assert l2 == l1 + 1

    def test_student_exists(self, temp_db: Database):
        assert not temp_db.student_exists("CS001")
        temp_db.add_student("CS001", "Amit", 5)
        assert temp_db.student_exists("CS001")

    def test_delete_student_removes_record(self, temp_db: Database):
        temp_db.add_student("CS001", "Amit", 5)
        assert temp_db.delete_student("CS001") is True
        assert not temp_db.student_exists("CS001")

    def test_delete_nonexistent_student_returns_false(self, temp_db: Database):
        assert temp_db.delete_student("GHOST") is False

    def test_list_students_sorted_by_name(self, temp_db: Database):
        temp_db.add_student("CS002", "Zara", 5)
        temp_db.add_student("CS001", "Amit", 5)
        names = [s["name"] for s in temp_db.list_students()]
        assert names == ["Amit", "Zara"]


class TestAttendanceManagement:
    def test_mark_attendance_creates_row(self, temp_db: Database):
        temp_db.add_student("CS001", "Amit", 5)
        marked = temp_db.mark_attendance("CS001", confidence=30.5)
        assert marked is True

        rows = temp_db.get_attendance(student_id="CS001")
        assert len(rows) == 1
        assert rows[0]["student_id"] == "CS001"

    def test_duplicate_mark_same_day_is_prevented(self, temp_db: Database):
        temp_db.add_student("CS001", "Amit", 5)
        first = temp_db.mark_attendance("CS001", confidence=30.0)
        second = temp_db.mark_attendance("CS001", confidence=35.0)
        assert first is True
        assert second is False

        rows = temp_db.get_attendance(student_id="CS001")
        assert len(rows) == 1

    def test_attendance_summary_counts_per_student(self, temp_db: Database):
        temp_db.add_student("CS001", "Amit", 5)
        temp_db.add_student("CS002", "Priya", 5)
        temp_db.mark_attendance("CS001", confidence=30.0)

        summary = {row["student_id"]: row["days_present"] for row in temp_db.attendance_summary()}
        assert summary["CS001"] == 1
        assert summary["CS002"] == 0

    def test_deleting_student_removes_their_attendance(self, temp_db: Database):
        temp_db.add_student("CS001", "Amit", 5)
        temp_db.mark_attendance("CS001", confidence=30.0)
        temp_db.delete_student("CS001")

        rows = temp_db.get_attendance()
        assert len(rows) == 0
