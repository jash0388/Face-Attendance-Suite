import os
import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
from waitress import serve
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
import time
import base64
from PIL import Image
import io

# Load Environment Variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
app.secret_key = os.getenv("SESSION_SECRET", "face-attendance-secret-2026")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global lists for fast recognition
known_face_encodings = []
known_face_names = []

def reload_encodings():
    global known_face_encodings, known_face_names
    encs, names = [], []
    try:
        res = supabase.table("students").select("name, roll_number, encoding").execute()
        for s in res.data:
            raw_enc = s.get("encoding")
            if raw_enc:
                # Handle both single and multi-vector encodings
                if isinstance(raw_enc, list) and len(raw_enc) > 0:
                    if isinstance(raw_enc[0], list):
                        for sub in raw_enc:
                            encs.append(np.array(sub, dtype=np.float64))
                            names.append(f"{s['name']}|{s['roll_number']}")
                    else:
                        encs.append(np.array(raw_enc, dtype=np.float64))
                        names.append(f"{s['name']}|{s['roll_number']}")
        known_face_encodings, known_face_names = encs, names
        print(f">>> {len(known_face_encodings)} FACE VECTORS LOADED INTO SYSTEM MEMORY.")
    except Exception as e:
        print(f">>> ERROR LOADING ENCODINGS: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/students")
def list_students():
    q = request.args.get("q", "").strip()
    query = supabase.table("students").select("*")
    if q:
        query = query.or_(f"name.ilike.%{q}%,roll_number.ilike.%{q}%")
    res = query.execute()
    return render_template("students.html", students=res.data)

@app.route("/students/add", methods=["GET", "POST"])
def add_student():
    if request.method == "POST":
        name = request.form.get("name")
        roll = request.form.get("roll_number")
        
        angles = ["front", "left", "right"]
        all_encodings = []
        main_image_url = ""

        try:
            for angle in angles:
                file = request.files.get(f"image_{angle}")
                if file:
                    img_bytes = file.read()
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_locs = face_recognition.face_locations(rgb_img)
                    
                    if face_locs:
                        enc = face_recognition.face_encodings(rgb_img, face_locs)[0]
                        all_encodings.append(enc.tolist())
                        
                        if angle == "front":
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{roll}_{timestamp}.jpg"
                            supabase.storage.from_("face-attendance").upload(
                                path=f"students/{filename}",
                                file=img_bytes,
                                file_options={"content-type": "image/jpeg"}
                            )
                            main_image_url = supabase.storage.from_("face-attendance").get_public_url(f"students/{filename}")

            if not all_encodings:
                flash("No faces detected. Ensure good lighting.", "error")
                return redirect(url_for("add_student"))

            supabase.table("students").insert({
                "name": name, "roll_number": roll,
                "image_path": main_image_url, "encoding": all_encodings,
                "created_at": datetime.now().isoformat()
            }).execute()

            reload_encodings()
            flash(f"Face ID Enrolled for {name}.", "success")
            return redirect(url_for("list_students"))

        except Exception as e:
            flash(f"Enrollment Failed: {str(e)}", "error")
            return redirect(url_for("add_student"))

    return render_template("add_student.html")

@app.route("/attendance")
def view_attendance():
    q = request.args.get("q", "").strip()
    query = supabase.table("attendance").select("*")
    if q:
        query = query.or_(f"name.ilike.%{q}%,roll_number.ilike.%{q}%")
    res = query.order("date", desc=True).order("time", desc=True).execute()
    return render_template("attendance.html", rows=res.data)

def generate_frames():
    global known_face_encodings, known_face_names
    if not known_face_encodings: reload_encodings()
    
    cap = cv2.VideoCapture(0)
    marked_today = set()
    frame_count = 0
    last_face_data = []

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        if frame_count % 3 == 0:
            last_face_data = []
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locs = face_recognition.face_locations(rgb_frame)
            face_encs = face_recognition.face_encodings(rgb_frame, face_locs)
            
            for (top, right, bottom, left), face_enc in zip(face_locs, face_encs):
                name, roll = "Unknown", ""
                if known_face_encodings:
                    distances = face_recognition.face_distance(known_face_encodings, face_enc)
                    best_match_idx = np.argmin(distances)
                    if distances[best_match_idx] <= 0.5:
                        label_data = known_face_names[best_match_idx].split("|")
                        name = label_data[0]
                        roll = label_data[1]
                        today = datetime.now().strftime("%Y-%m-%d")
                        if f"{roll}|{today}" not in marked_today:
                            supabase.table("attendance").insert({
                                "name": name, "roll_number": roll, "date": today,
                                "time": datetime.now().strftime("%H:%M:%S")
                            }).execute()
                            marked_today.add(f"{roll}|{today}")

                last_face_data.append({
                    "box": (top, right, bottom, left),
                    "label": name.upper()
                })

        for face in last_face_data:
            t, r, b, l = face["box"]
            color = (16, 185, 129) if face["label"] != "UNKNOWN" else (59, 130, 246)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, face["label"], (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/admin")
def admin_dashboard():
    res_count = supabase.table("unknown_detections").select("count", count="exact").execute()
    total_unknowns = res_count.count if res_count.count is not None else 0
    res_recent = supabase.table("unknown_detections").select("*").order("detected_at", desc=True).limit(5).execute()
    return render_template("admin_dashboard.html", total_unknowns=total_unknowns, recent_unknowns=res_recent.data)

@app.route("/camera")
def camera():
    return render_template("camera.html")

if __name__ == "__main__":
    reload_encodings()
    port = int(os.environ.get("PORT", 3000))
    print(f">>> STARTING PRODUCTION SERVER ON PORT {port}...")
    serve(app, host="0.0.0.0", port=port)
