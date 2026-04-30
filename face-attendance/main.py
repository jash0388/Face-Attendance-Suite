"""
Face Recognition Attendance System - main runner
------------------------------------------------
Loads student face images from students/ (and database.db),
opens a camera stream (RTSP / HTTP / Webcam), recognises faces and
appends attendance to attendance.csv.

Usage:
    python main.py                          # Use default webcam (index 0)
    python main.py --source 1               # Use webcam index 1
    python main.py --source rtsp://user:pass@192.168.1.10:554/stream
    python main.py --source http://192.168.1.10:8080/video

Press 'q' in the video window to stop.
"""

import argparse
import csv
import os
import sqlite3
import sys
from datetime import datetime

import cv2
import face_recognition
import numpy as np

# ---------------- Configuration ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENTS_DIR = os.path.join(BASE_DIR, "students")
DB_PATH = os.path.join(BASE_DIR, "database.db")
ATTENDANCE_CSV = os.path.join(BASE_DIR, "attendance.csv")

# Resize factor for the frame sent to face_recognition (smaller = faster).
RESIZE_SCALE = 0.25  # 1/4 size
# How strict the match should be. Lower = stricter. 0.6 is the dlib default.
MATCH_TOLERANCE = 0.5


# ---------------- Helpers ----------------
def load_students_from_db():
    """Return list of (name, roll, image_filename) from the SQLite DB."""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT name, roll_number, image_path FROM students"
    ).fetchall()
    conn.close()
    return [(r["name"], r["roll_number"], r["image_path"]) for r in rows]


def encode_known_students():
    """
    Load every student image and compute its face encoding.
    Returns: (encodings_list, labels_list)  where each label is "Name|Roll".
    """
    encodings = []
    labels = []

    students = load_students_from_db()
    if not students:
        print("[WARN] No students found in the database. "
              "Add students through the Flask admin panel first.")
        return encodings, labels

    print(f"[INFO] Encoding {len(students)} student face(s)...")
    for name, roll, image_filename in students:
        image_path = os.path.join(STUDENTS_DIR, image_filename)
        if not os.path.exists(image_path):
            print(f"[WARN] Missing image for {name} ({roll}): {image_path}")
            continue

        image = face_recognition.load_image_file(image_path)
        face_encs = face_recognition.face_encodings(image)
        if not face_encs:
            print(f"[WARN] No face detected in image for {name} ({roll}).")
            continue

        encodings.append(face_encs[0])
        labels.append(f"{name}|{roll}")
        print(f"[OK]   Encoded {name} ({roll}).")

    return encodings, labels


def ensure_attendance_csv():
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Roll Number", "Date", "Time"])


def mark_attendance(name: str, roll: str, marked_today: set) -> bool:
    """
    Append a row to attendance.csv if (roll, today's date) is not already
    present. Returns True if a new row was written.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"{roll}|{today}"
    if key in marked_today:
        return False

    now = datetime.now()
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, roll, today, now.strftime("%H:%M:%S")])

    marked_today.add(key)
    print(f"[ATTENDANCE] {name} ({roll}) at {now.strftime('%H:%M:%S')}")
    return True


def load_marked_today() -> set:
    """Load any rows already in attendance.csv for today's date so we don't
    double-mark the same student in the same session."""
    marked = set()
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(ATTENDANCE_CSV):
        return marked
    with open(ATTENDANCE_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Date") == today:
                marked.add(f"{row.get('Roll Number')}|{today}")
    return marked


def open_video_source(source):
    """Open a webcam index or an RTSP/HTTP URL."""
    # If `source` is an integer-like string, convert to int (webcam index)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    return cap


# ---------------- Main loop ----------------
def run(source):
    ensure_attendance_csv()
    known_encodings, known_labels = encode_known_students()
    marked_today = load_marked_today()

    print(f"[INFO] Opening video source: {source}")
    cap = open_video_source(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        sys.exit(1)

    print("[INFO] Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Stopping.")
            break

        # Resize for speed, convert BGR (OpenCV) -> RGB (face_recognition)
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
            name, roll = "Unknown", ""

            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, face_enc)
                best_idx = int(np.argmin(distances))
                if distances[best_idx] <= MATCH_TOLERANCE:
                    label = known_labels[best_idx]
                    name, roll = label.split("|", 1)
                    mark_attendance(name, roll, marked_today)

            # Scale coords back up to the original frame size
            scale = int(1 / RESIZE_SCALE)
            top, right, bottom, left = (
                top * scale, right * scale, bottom * scale, left * scale
            )

            color = (0, 200, 0) if name != "Unknown" else (0, 0, 200)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label_text = f"{name} ({roll})" if roll else name
            cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame, label_text, (left + 6, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

        cv2.imshow("Face Recognition Attendance (press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera stopped.")


def parse_args():
    parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. 0, 1) or RTSP/HTTP URL.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.source)
