"""
Flask Admin Panel for Face Recognition Attendance System
--------------------------------------------------------
Lets you Add / View / Delete students and view attendance records.
Run with:   python app.py
Then open:  http://localhost:5000
"""

import os
print(">>> INITIALIZING FACE ATTENDANCE APP...")
import sqlite3
import csv
import threading
import time
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_from_directory, jsonify
)
from werkzeug.utils import secure_filename
import face_recognition
import numpy as np

# ---------------- Configuration ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENTS_DIR = os.path.join(BASE_DIR, "students")
DB_PATH = os.path.join(BASE_DIR, "database.db")
ATTENDANCE_CSV = os.path.join(BASE_DIR, "attendance.csv")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(STUDENTS_DIR, exist_ok=True)
UNKNOWN_FACES_DIR = os.path.join(BASE_DIR, "static", "unknown_faces")
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)


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
                encoding    BLOB,
                created_at  TEXT    NOT NULL
            )
            """
        )
        # Migration: Check if encoding column exists
        try:
            conn.execute("SELECT encoding FROM students LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE students ADD COLUMN encoding BLOB")
        
        # Create unknown_detections table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unknown_detections (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path  TEXT    NOT NULL,
                detected_at TEXT    NOT NULL
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
            # 1. SAVE THE FILE FIRST
            file.save(save_path)

            # 2. RESIZE IMAGE FOR FASTER ENCODING
            print(f">>> OPTIMIZING IMAGE FOR: {name}")
            from PIL import Image
            with Image.open(save_path) as pil_img:
                # Convert to RGB if necessary
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                # Resize to max 800px width/height while maintaining aspect ratio
                pil_img.thumbnail((800, 800), Image.LANCZOS)
                pil_img.save(save_path, "JPEG", quality=85)

            # 3. CALCULATE ENCODING
            print(f">>> ENCODING FACE...")
            img = face_recognition.load_image_file(save_path)
            # Use 'hog' model for speed (default), but ensure it scans the whole image
            face_encs = face_recognition.face_encodings(img, num_jitters=1)
            encoding_blob = face_encs[0].tobytes() if face_encs else None
            print(f">>> ENCODING COMPLETE. FOUND: {len(face_encs)} FACE(S)")

            # 3. SAVE TO DATABASE
            with get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO students (name, roll_number, image_path, encoding, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (name, roll, filename, encoding_blob, datetime.now().isoformat(timespec="seconds")),
                )
                conn.commit()
            
            # 4. UPDATE LIVE SYSTEM
            reload_encodings()
            flash(f"Student '{name}' registered successfully.", "success")
            return redirect(url_for("list_students"))

        except sqlite3.IntegrityError:
            if os.path.exists(save_path): os.remove(save_path) # Cleanup
            flash(f"A student with roll number '{roll}' already exists.", "error")
            return redirect(url_for("add_student"))
        except Exception as e:
            if os.path.exists(save_path): os.remove(save_path) # Cleanup
            flash(f"Error saving student: {str(e)}", "error")
            return redirect(url_for("add_student"))

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


@app.route("/admin")
def admin_dashboard():
    with get_db() as conn:
        total_unknowns = conn.execute("SELECT COUNT(*) FROM unknown_detections").fetchone()[0]
        recent_unknowns = conn.execute("SELECT * FROM unknown_detections ORDER BY detected_at DESC LIMIT 5").fetchall()
    return render_template("admin_dashboard.html", total_unknowns=total_unknowns, recent_unknowns=recent_unknowns)


@app.route("/admin/unknowns")
def list_unknowns():
    with get_db() as conn:
        unknowns = conn.execute("SELECT * FROM unknown_detections ORDER BY detected_at DESC").fetchall()
    return render_template("unknown_detections.html", unknowns=unknowns)


@app.route("/admin/unknowns/delete/<int:detection_id>", methods=["POST"])
def delete_unknown(detection_id: int):
    with get_db() as conn:
        detection = conn.execute("SELECT * FROM unknown_detections WHERE id = ?", (detection_id,)).fetchone()
        if detection:
            img_path = os.path.join(UNKNOWN_FACES_DIR, detection["image_path"])
            if os.path.exists(img_path):
                os.remove(img_path)
            conn.execute("DELETE FROM unknown_detections WHERE id = ?", (detection_id,))
            conn.commit()
            flash("Unknown detection record deleted.", "success")
    return redirect(url_for("list_unknowns"))


@app.route("/unknown/image/<path:filename>")
def unknown_image(filename: str):
    return send_from_directory(UNKNOWN_FACES_DIR, filename)


# ---------------- Camera test (browser-based) ----------------
# ---------------- CCTV / Backend Streaming ----------------
camera_source = "0"  # Default to webcam
known_encodings = []
known_labels = []

def reload_encodings():
    global known_encodings, known_labels
    encs = []
    labs = []
    with get_db() as conn:
        students = conn.execute("SELECT name, roll_number, encoding FROM students WHERE encoding IS NOT NULL").fetchall()
    
    for s in students:
        # Load from BLOB instead of processing image file (MUCH FASTER)
        vec = np.frombuffer(s["encoding"], dtype=np.float64)
        encs.append(vec)
        labs.append(f"{s['name']}|{s['roll_number']}")
        
    known_encodings = encs
    known_labels = labs

@app.route("/api/config-camera", methods=["POST"])
def config_camera():
    global camera_source
    data = request.get_json(silent=True) or {}
    source = data.get("source", "0")
    camera_source = int(source) if source.isdigit() else source
    return jsonify({"ok": True, "source": camera_source})

def generate_frames():
    import cv2
    import face_recognition
    import numpy as np
    global camera_source, known_encodings, known_labels
    if not known_encodings:
        reload_encodings()
        
    cap = cv2.VideoCapture(camera_source)
    marked_today = set() 
    frame_count = 0
    last_unknown_capture = 0 # Cooldown for unknown face captures
    
    last_face_data = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        # Perform face recognition every 3rd frame (faster than 5th, but still saves CPU)
        if frame_count % 3 == 0:
            last_face_data = []
            # Use higher resolution for better CCTV-style detection of distant faces
            # 0.75 is a good balance between speed and accuracy for multiple people
            scale = 0.75
            small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all face locations and encodings in the current frame
            # number_of_times_to_upsample=1 helps find smaller faces in the background
            face_locs = face_recognition.face_locations(rgb_small, number_of_times_to_upsample=1, model="hog")
            face_encs = face_recognition.face_encodings(rgb_small, face_locs)
            
            any_unknown_in_frame = False
            for (top, right, bottom, left), face_enc in zip(face_locs, face_encs):
                name, roll = "Unknown", ""
                if known_encodings:
                    # Compare face with ALL known students at once
                    distances = face_recognition.face_distance(known_encodings, face_enc)
                    best_idx = int(np.argmin(distances))
                    if distances[best_idx] <= 0.55: # Slightly stricter for multi-face accuracy
                        name, roll = known_labels[best_idx].split("|")
                        today = datetime.now().strftime("%Y-%m-%d")
                        if f"{roll}|{today}" not in marked_today:
                            with open(ATTENDANCE_CSV, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([name, roll, today, datetime.now().strftime("%H:%M:%S")])
                            marked_today.add(f"{roll}|{today}")
                    else:
                        any_unknown_in_frame = True

                # Store detection data (scaled back up correctly for CCTV resolution)
                last_face_data.append({
                    "box": (int(top / scale), int(right / scale), int(bottom / scale), int(left / scale)),
                    "label": name.upper()
                })
            
            # CCTV ALERT: Capture Unknown Faces if any unknown was detected in the entire frame
            if any_unknown_in_frame:
                current_time = time.time()
                # Capture at most once every 5 seconds for CCTV-style monitoring
                if current_time - last_unknown_capture > 5:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_name = f"unknown_{timestamp}.jpg"
                    img_path = os.path.join(UNKNOWN_FACES_DIR, img_name)
                    
                    # Save a copy of the frame with RED markers for the admin log
                    # This helps security see exactly who was detected as unknown
                    record_frame = frame.copy()
                    for face in last_face_data:
                        t, r, b, l = face["box"]
                        # Draw red box for records
                        cv2.rectangle(record_frame, (l, t), (r, b), (0, 0, 255), 2)
                        cv2.putText(record_frame, face["label"], (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    cv2.imwrite(img_path, record_frame)
                    
                    # Log to DB
                    with get_db() as conn:
                        conn.execute(
                            "INSERT INTO unknown_detections (image_path, detected_at) VALUES (?, ?)",
                            (img_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        )
                        conn.commit()
                    
                    last_unknown_capture = current_time
                    print(f">>> SECURITY ALERT: Unknown person detected at {timestamp}")

        # Draw all persistent face detections on EVERY frame
        for face in last_face_data:
            top, right, bottom, left = face["box"]
            label = face["label"]
            # Technical corner brackets for better aesthetic
            d = 20
            cv2.line(frame, (left, top), (left + d, top), (255, 255, 255), 2)
            cv2.line(frame, (left, top), (left, top + d), (255, 255, 255), 2)
            cv2.line(frame, (right, top), (right - d, top), (255, 255, 255), 2)
            cv2.line(frame, (right, top), (right, top + d), (255, 255, 255), 2)
            cv2.line(frame, (left, bottom), (left + d, bottom), (255, 255, 255), 2)
            cv2.line(frame, (left, bottom), (left, bottom - d), (255, 255, 255), 2)
            cv2.line(frame, (right, bottom), (right - d, bottom), (255, 255, 255), 2)
            cv2.line(frame, (right, bottom), (right, bottom - d), (255, 255, 255), 2)
            
            cv2.putText(frame, f"> {label}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Flask.response_class(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/camera")
def camera():
    """Live face-recognition test page using the browser's webcam."""
    return render_template("camera.html")


@app.route("/api/students")
def api_students():
    """JSON list of students with image URLs (used by the camera test page)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, roll_number, image_path FROM students"
        ).fetchall()
    return jsonify([
        {
            "id": r["id"],
            "name": r["name"],
            "roll_number": r["roll_number"],
            "image_url": url_for("student_image", filename=r["image_path"]),
        }
        for r in rows
    ])


@app.route("/api/mark-attendance", methods=["POST"])
def api_mark_attendance():
    """Append a row to attendance.csv (one entry per student per day)."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    roll = (data.get("roll_number") or "").strip()
    if not name or not roll:
        return jsonify({"ok": False, "error": "name and roll_number required"}), 400

    today = datetime.now().strftime("%Y-%m-%d")

    # Avoid duplicate entries for the same student on the same day
    if os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "r", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("Roll Number") == roll and row.get("Date") == today:
                    return jsonify({"ok": True, "duplicate": True})

    now = datetime.now()
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, roll, today, now.strftime("%H:%M:%S")])

    return jsonify({"ok": True, "duplicate": False, "time": now.strftime("%H:%M:%S")})


# ---------------- Entry point ----------------
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 3000))
    # Use Waitress (production-grade, multi-threaded WSGI server) so concurrent
    # requests (e.g. parallel model downloads + /api calls) don't block each
    # other and cause proxy 502s.
    try:
        from waitress import serve
        print(f" * Serving Face Attendance admin on http://0.0.0.0:{port} (waitress)")
        serve(app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        # Fallback if waitress isn't installed.
        app.run(host="0.0.0.0", port=port, debug=True,
                use_reloader=False, threaded=True)
