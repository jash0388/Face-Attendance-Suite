"""
Flask Admin Panel for Face Recognition Attendance System
--------------------------------------------------------
Lets you Add / View / Delete students and view attendance records.
Run with:   python app.py
Then open:  http://localhost:5000
"""

import os
print(">>> INITIALIZING FACE ATTENDANCE APP...")
import time
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_from_directory, jsonify
)
from werkzeug.utils import secure_filename
import face_recognition
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# ---------------- Configuration ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Supabase Config
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "change-me-in-production")


# ---------------- Database helpers ----------------
def init_db():
    """Ensure Supabase buckets exist."""
    try:
        # Check if bucket exists, if not it might throw an error or we can just try to create it
        supabase.storage.create_bucket("face-attendance", options={"public": True})
        print(">>> CREATED SUPABASE BUCKET: face-attendance")
    except Exception as e:
        # Likely already exists
        pass


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- Routes ----------------
@app.route("/")
def index():
    """Dashboard with quick stats."""
    # Get total students
    res = supabase.table("students").select("count", count="exact").execute()
    total_students = res.count if res.count is not None else 0

    # Get today's attendance count
    today = datetime.now().strftime("%Y-%m-%d")
    res_attendance = supabase.table("attendance").select("count", count="exact").eq("date", today).execute()
    today_count = res_attendance.count if res_attendance.count is not None else 0

    return render_template(
        "index.html",
        total_students=total_students,
        today_count=today_count,
        today=today,
    )


@app.route("/students")
def list_students():
    q = request.args.get("q", "").strip()
    query = supabase.table("students").select("*")
    
    if q:
        query = query.or_(f"name.ilike.%{q}%,roll_number.ilike.%{q}%")
        
    res = query.order("name").execute()
    students = res.data
    return render_template("students.html", students=students)


@app.route("/students/add", methods=["GET", "POST"])
def add_student():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll = request.form.get("roll_number", "").strip()
        
        # We now expect 3 images: front, left, right
        files = {
            "front": request.files.get("image_front"),
            "left": request.files.get("image_left"),
            "right": request.files.get("image_right")
        }

        if not name or not roll:
            flash("Name and Roll Number are required.", "error")
            return redirect(url_for("add_student"))

        all_encodings = []
        main_image_url = ""

        try:
            from PIL import Image
            for angle, file in files.items():
                if not file or file.filename == "":
                    continue
                
                ext = file.filename.rsplit(".", 1)[1].lower()
                temp_name = f"temp_{roll}_{angle}.{ext}"
                local_path = os.path.join(BASE_DIR, temp_name)
                
                file.save(local_path)
                
                # Optimize
                with Image.open(local_path) as pil_img:
                    pil_img.thumbnail((640, 640), Image.LANCZOS)
                    pil_img.save(local_path, "JPEG", quality=75)
                
                # Encode
                img = face_recognition.load_image_file(local_path)
                face_encs = face_recognition.face_encodings(img, num_jitters=0)
                if face_encs:
                    all_encodings.append(face_encs[0].tolist())
                
                # Upload to Supabase (Only the front image is used for the UI thumbnail)
                with open(local_path, "rb") as f:
                    supabase.storage.from_("face-attendance").upload(
                        path=f"students/{roll}_{angle}.jpg",
                        file=f,
                        file_options={"cache-control": "3600", "upsert": "true"}
                    )
                
                if angle == "front":
                    main_image_url = supabase.storage.from_("face-attendance").get_public_url(f"students/{roll}_front.jpg")
                
                if os.path.exists(local_path): os.remove(local_path)

            if not all_encodings:
                flash("No faces detected in the uploaded images.", "error")
                return redirect(url_for("add_student"))

            # Save to Database (encoding is now a list of 1-3 vectors)
            student_data = {
                "name": name,
                "roll_number": roll,
                "image_path": main_image_url,
                "encoding": all_encodings, 
                "created_at": datetime.now().isoformat()
            }
            supabase.table("students").insert(student_data).execute()

            reload_encodings()
            flash(f"Face ID Enrollment complete for '{name}'.", "success")
            return redirect(url_for("list_students"))

        except Exception as e:
            flash(f"Enrollment Error: {str(e)}", "error")
            return redirect(url_for("add_student"))

    return render_template("add_student.html")


@app.route("/students/delete/<int:student_id>", methods=["POST"])
def delete_student(student_id: int):
    # Fetch student to get image path
    res = supabase.table("students").select("*").eq("id", student_id).execute()
    if not res.data:
        flash("Student not found.", "error")
        return redirect(url_for("list_students"))
    
    student = res.data[0]
    filename = student["image_path"].split("/")[-1]

    # Delete from Storage
    try:
        supabase.storage.from_("face-attendance").remove([f"students/{filename}"])
    except:
        pass

    # Delete from Database
    supabase.table("students").delete().eq("id", student_id).execute()

    flash(f"Student '{student['name']}' deleted.", "success")
    return redirect(url_for("list_students"))


@app.route("/attendance")
def view_attendance():
    q = request.args.get("q", "").strip()
    query = supabase.table("attendance").select("*")
    
    if q:
        # Filter by name OR roll_number using or condition in Supabase
        query = query.or_(f"name.ilike.%{q}%,roll_number.ilike.%{q}%")
        
    res = query.order("date", desc=True).order("time", desc=True).execute()
    rows = res.data
    return render_template("attendance.html", rows=rows)


@app.route("/admin")
def admin_dashboard():
    # Total unknowns
    res_count = supabase.table("unknown_detections").select("count", count="exact").execute()
    total_unknowns = res_count.count if res_count.count is not None else 0
    
    # Recent unknowns
    res_recent = supabase.table("unknown_detections").select("*").order("detected_at", desc=True).limit(5).execute()
    recent_unknowns = res_recent.data
    
    return render_template("admin_dashboard.html", total_unknowns=total_unknowns, recent_unknowns=recent_unknowns)


@app.route("/admin/unknowns")
def list_unknowns():
    res = supabase.table("unknown_detections").select("*").order("detected_at", desc=True).execute()
    unknowns = res.data
    return render_template("unknown_detections.html", unknowns=unknowns)


@app.route("/admin/unknowns/delete/<int:detection_id>", methods=["POST"])
def delete_unknown(detection_id: int):
    # Fetch detection to get image path
    res = supabase.table("unknown_detections").select("*").eq("id", detection_id).execute()
    if res.data:
        detection = res.data[0]
        filename = detection["image_path"].split("/")[-1]
        
        # Delete from Storage
        try:
            supabase.storage.from_("face-attendance").remove([f"unknowns/{filename}"])
        except:
            pass
            
        # Delete from DB
        supabase.table("unknown_detections").delete().eq("id", detection_id).execute()
        flash("Unknown detection record deleted.", "success")
    
    return redirect(url_for("list_unknowns"))


@app.route("/students/image/<path:filename>")
def student_image(filename: str):
    # Redirect to Supabase Public URL
    url = supabase.storage.from_("face-attendance").get_public_url(f"students/{filename}")
    return redirect(url)

@app.route("/unknown/image/<path:filename>")
def unknown_image(filename: str):
    # Redirect to Supabase Public URL
    url = supabase.storage.from_("face-attendance").get_public_url(f"unknowns/{filename}")
    return redirect(url)


# ---------------- Camera test (browser-based) ----------------
# ---------------- CCTV / Backend Streaming ----------------
camera_source = "0"  # Default to webcam
known_encodings = []
known_labels = []

def reload_encodings():
    global known_encodings, known_labels
    encs = []
    labs = []
    res = supabase.table("students").select("name, roll_number, encoding").execute()
    students = res.data
    
    for s in students:
        raw_encoding = s.get("encoding")
        if raw_encoding:
            # Check if it's a list of multiple encodings (Face ID style)
            if isinstance(raw_encoding[0], list):
                for sub_enc in raw_encoding:
                    encs.append(np.array(sub_enc, dtype=np.float64))
                    labs.append(f"{s['name']}|{s['roll_number']}")
            else:
                # Fallback for old single-encoding records
                encs.append(np.array(raw_encoding, dtype=np.float64))
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
                            # Mark attendance in Supabase
                            supabase.table("attendance").insert({
                                "name": name,
                                "roll_number": roll,
                                "date": today,
                                "time": datetime.now().strftime("%H:%M:%S")
                            }).execute()
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
                    local_temp_path = os.path.join(BASE_DIR, img_name)
                    
                    # Save a copy of the frame with RED markers for the admin log
                    record_frame = frame.copy()
                    for face in last_face_data:
                        t, r, b, l = face["box"]
                        cv2.rectangle(record_frame, (l, t), (r, b), (0, 0, 255), 2)
                        cv2.putText(record_frame, face["label"], (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    cv2.imwrite(local_temp_path, record_frame)
                    
                    # USE A THREAD FOR UPLOAD: Don't block the camera feed!
                    def upload_unknown_task(path, name):
                        try:
                            with open(path, "rb") as f:
                                supabase.storage.from_("face-attendance").upload(
                                    path=f"unknowns/{name}",
                                    file=f,
                                    file_options={"cache-control": "3600", "upsert": "true"}
                                )
                            
                            # Get Public URL and Log to DB
                            image_url = supabase.storage.from_("face-attendance").get_public_url(f"unknowns/{name}")
                            supabase.table("unknown_detections").insert({
                                "image_path": image_url,
                                "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }).execute()
                            
                            if os.path.exists(path): os.remove(path)
                        except Exception as e:
                            print(f">>> ERROR UPLOADING SECURITY SNAPSHOT: {e}")

                    import threading
                    threading.Thread(target=upload_unknown_task, args=(local_temp_path, img_name)).start()
                    
                    last_unknown_capture = current_time
                    print(f">>> SECURITY SNAPSHOT TRIGGERED: {timestamp}")

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
    """JSON list of students with image URLs."""
    res = supabase.table("students").select("id, name, roll_number, image_path").execute()
    return jsonify([
        {
            "id": r["id"],
            "name": r["name"],
            "roll_number": r["roll_number"],
            "image_url": r["image_path"],
        }
        for r in res.data
    ])


@app.route("/api/mark-attendance", methods=["POST"])
def api_mark_attendance():
    """Append a row to Supabase attendance table."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    roll = (data.get("roll_number") or "").strip()
    if not name or not roll:
        return jsonify({"ok": False, "error": "name and roll_number required"}), 400

    today = datetime.now().strftime("%Y-%m-%d")

    # Avoid duplicate entries for the same student on the same day
    res_dup = supabase.table("attendance").select("id").eq("roll_number", roll).eq("date", today).execute()
    if res_dup.data:
        return jsonify({"ok": True, "duplicate": True})

    now = datetime.now()
    supabase.table("attendance").insert({
        "name": name,
        "roll_number": roll,
        "date": today,
        "time": now.strftime("%H:%M:%S")
    }).execute()

    return jsonify({"ok": True, "duplicate": False, "time": now.strftime("%H:%M:%S")})


# ---------------- Entry point ----------------
if __name__ == "__main__":
    init_db()
    reload_encodings()
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
