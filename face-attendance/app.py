"""
Flask Admin Panel for Face Recognition Attendance System
--------------------------------------------------------
Lets you Add / View / Delete students and view attendance records.
Run with:   python app.py
Then open:  http://localhost:5000
"""

import os
import sqlite3
import csv
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for, flash, send_from_directory
)
from werkzeug.utils import secure_filename

# ---------------- Configuration ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENTS_DIR = os.path.join(BASE_DIR, "students")
DB_PATH = os.path.join(BASE_DIR, "database.db")
ATTENDANCE_CSV = os.path.join(BASE_DIR, "attendance.csv")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(STUDENTS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "change-me-in-production")


# ---------------- Database helpers ----------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the students table if it doesn't exist."""
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                roll_number TEXT    NOT NULL UNIQUE,
                image_path  TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
            )
            """
        )
        conn.commit()

    # Make sure the attendance CSV exists with a header row
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Roll Number", "Date", "Time"])


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- Routes ----------------
@app.route("/")
def index():
    """Dashboard with quick stats."""
    with get_db() as conn:
        total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]

    today = datetime.now().strftime("%Y-%m-%d")
    today_count = 0
    if os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            today_count = sum(1 for row in reader if row.get("Date") == today)

    return render_template(
        "index.html",
        total_students=total_students,
        today_count=today_count,
        today=today,
    )


@app.route("/students")
def list_students():
    with get_db() as conn:
        students = conn.execute(
            "SELECT * FROM students ORDER BY name COLLATE NOCASE"
        ).fetchall()
    return render_template("students.html", students=students)


@app.route("/students/add", methods=["GET", "POST"])
def add_student():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll = request.form.get("roll_number", "").strip()
        file = request.files.get("image")

        if not name or not roll:
            flash("Name and Roll Number are required.", "error")
            return redirect(url_for("add_student"))

        if not file or file.filename == "":
            flash("Please upload a face image.", "error")
            return redirect(url_for("add_student"))

        if not allowed_file(file.filename):
            flash("Only PNG, JPG, and JPEG images are allowed.", "error")
            return redirect(url_for("add_student"))

        ext = file.filename.rsplit(".", 1)[1].lower()
        safe_roll = secure_filename(roll)
        filename = f"{safe_roll}.{ext}"
        save_path = os.path.join(STUDENTS_DIR, filename)

        try:
            with get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO students (name, roll_number, image_path, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, roll, filename, datetime.now().isoformat(timespec="seconds")),
                )
                conn.commit()
        except sqlite3.IntegrityError:
            flash(f"A student with roll number '{roll}' already exists.", "error")
            return redirect(url_for("add_student"))

        file.save(save_path)
        flash(f"Student '{name}' added successfully.", "success")
        return redirect(url_for("list_students"))

    return render_template("add_student.html")


@app.route("/students/delete/<int:student_id>", methods=["POST"])
def delete_student(student_id: int):
    with get_db() as conn:
        student = conn.execute(
            "SELECT * FROM students WHERE id = ?", (student_id,)
        ).fetchone()
        if student is None:
            flash("Student not found.", "error")
            return redirect(url_for("list_students"))

        # Delete image file if it exists
        image_path = os.path.join(STUDENTS_DIR, student["image_path"])
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass

        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()

    flash(f"Student '{student['name']}' deleted.", "success")
    return redirect(url_for("list_students"))


@app.route("/attendance")
def view_attendance():
    rows = []
    if os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    # Show newest first
    rows.reverse()
    return render_template("attendance.html", rows=rows)


@app.route("/students/image/<path:filename>")
def student_image(filename: str):
    return send_from_directory(STUDENTS_DIR, filename)


# ---------------- Entry point ----------------
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
