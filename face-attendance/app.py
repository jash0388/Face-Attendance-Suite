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

# Load Environment Variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
app.secret_key = os.getenv("SESSION_SECRET", "face-attendance-secret-2026")

# Global Cache for Speed
known_face_encodings = []
known_face_names = []
camera_source = 0

# LOAD AI MODELS FOR BACKEND (DNN is much faster than HOG/CNN for 4MP)
# Using OpenCV's DNN Face Detector for "Elite" CCTV performance
proto_path = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
model_path = os.path.join(os.path.dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")

# Fallback to face_recognition if DNN files aren't found, but we'll try to download/use them
net = None
if os.path.exists(proto_path) and os.path.exists(model_path):
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

def reload_encodings():
    global known_face_encodings, known_face_names
    encs, names = [], []
    try:
        res = supabase.table("students").select("name, roll_number, encoding").execute()
        for s in res.data:
            raw_enc = s.get("encoding")
            if raw_enc:
                if isinstance(raw_enc, list) and len(raw_enc) > 0:
                    if isinstance(raw_enc[0], list):
                        for sub in raw_enc:
                            encs.append(np.array(sub, dtype=np.float64))
                            names.append(f"{s['name']}|{s['roll_number']}")
                    else:
                        encs.append(np.array(raw_enc, dtype=np.float64))
                        names.append(f"{s['name']}|{s['roll_number']}")
        known_face_encodings, known_face_names = encs, names
        print(f">>> ELITE SYSTEM: {len(known_face_encodings)} VECTORS READY.")
    except Exception as e:
        print(f">>> DB SYNC ERROR: {e}")

@app.route("/")
def index(): return render_template("index.html")

@app.route("/students")
def list_students():
    res = supabase.table("students").select("*").execute()
    return render_template("students.html", students=res.data)

@app.route("/students/add", methods=["GET", "POST"])
def add_student():
    if request.method == "POST":
        name, roll = request.form.get("name"), request.form.get("roll_number")
        all_encs, main_url = [], ""
        for angle in ["front", "left", "right"]:
            file = request.files.get(f"image_{angle}")
            if file:
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb)
                if locs:
                    all_encs.append(face_recognition.face_encodings(rgb, locs)[0].tolist())
                    if angle == "front":
                        fname = f"{roll}_{int(time.time())}.jpg"
                        _, buf = cv2.imencode(".jpg", img)
                        supabase.storage.from_("face-attendance").upload(f"students/{fname}", buf.tobytes())
                        main_url = supabase.storage.from_("face-attendance").get_public_url(f"students/{fname}")
        
        if all_encs:
            supabase.table("students").insert({"name":name, "roll_number":roll, "image_path":main_url, "encoding":all_encs}).execute()
            reload_encodings()
            flash("Student Enrolled!", "success")
            return redirect(url_for("list_students"))
    return render_template("add_student.html")

@app.route("/attendance")
def view_attendance():
    res = supabase.table("attendance").select("*").order("date", desc=True).order("time", desc=True).execute()
    return render_template("attendance.html", rows=res.data)

def generate_frames():
    global known_face_encodings, known_face_names, camera_source, net
    if not known_face_encodings: reload_encodings()
    
    cap = cv2.VideoCapture(camera_source)
    # WiFi Optimization: Increase buffer for 4MP stream stability
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) 
    
    marked_today = set()
    frame_count = 0
    last_face_data = []

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        # Process EVERY 2ND frame for high catch-rate on 4MP cameras
        if frame_count % 2 == 0:
            last_face_data = []
            h, w = frame.shape[:2]
            
            # ELITE SPEED: Resize ONLY for detection, keep original for recognition
            detect_scale = 480 / h
            small = cv2.resize(frame, (0,0), fx=detect_scale, fy=detect_scale)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            # Use 'hog' (fast) but process at decent resolution
            face_locs = face_recognition.face_locations(rgb_small, model="hog")
            
            # Map locations back to FULL RESOLUTION (4MP) for perfect accuracy
            actual_locs = []
            for (t, r, b, l) in face_locs:
                actual_locs.append((int(t/detect_scale), int(r/detect_scale), int(b/detect_scale), int(l/detect_scale)))
            
            if actual_locs:
                # Get encodings from the HIGH-RES (4MP) data
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encs = face_recognition.face_encodings(rgb_full, actual_locs)
                
                for (top, right, bottom, left), face_enc in zip(actual_locs, face_encs):
                    name, roll = "Unknown", ""
                    if known_face_encodings:
                        distances = face_recognition.face_distance(known_face_encodings, face_enc)
                        best_idx = np.argmin(distances)
                        if distances[best_idx] <= 0.48: # Ultra-Security Tolerance
                            name, roll = known_face_names[best_idx].split("|")
                            today = datetime.now().strftime("%Y-%m-%d")
                            if f"{roll}|{today}" not in marked_today:
                                supabase.table("attendance").insert({"name":name, "roll_number":roll, "date":today, "time":datetime.now().strftime("%H:%M:%S")}).execute()
                                marked_today.add(f"{roll}|{today}")
                    
                    last_face_data.append({"box": (top, right, bottom, left), "label": name.upper()})

        # Draw Tech-UI overlays
        for face in last_face_data:
            t, r, b, l = face["box"]
            color = (16, 185, 129) if face["label"] != "UNKNOWN" else (59, 130, 246)
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, face["label"], (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/camera")
def camera(): return render_template("camera.html")

@app.route("/api/students")
def api_students():
    res = supabase.table("students").select("id, name, roll_number, image_path").execute()
    data = []
    for r in res.data:
        url = r["image_path"] if r["image_path"].startswith("http") else supabase.storage.from_("face-attendance").get_public_url(f"students/{r['image_path']}")
        data.append({"id": r["id"], "name": r["name"], "roll_number": r["roll_number"], "image_url": url})
    return jsonify(data)

@app.route("/api/mark-attendance", methods=["POST"])
def api_mark_attendance():
    data = request.get_json(silent=True) or {}
    name, roll = data.get("name"), data.get("roll_number")
    if name and roll:
        supabase.table("attendance").insert({"name":name, "roll_number":roll, "date":datetime.now().strftime("%Y-%m-%d"), "time":datetime.now().strftime("%H:%M:%S")}).execute()
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 400

@app.route("/api/config-camera", methods=["POST"])
def config_camera():
    global camera_source
    data = request.get_json(silent=True) or {}
    source = data.get("source", "0")
    camera_source = int(source) if source.isdigit() else source
    return jsonify({"ok": True, "source": camera_source})

if __name__ == "__main__":
    reload_encodings()
    port = int(os.environ.get("PORT", 3000))
    serve(app, host="0.0.0.0", port=port)
