# Face Recognition Attendance System

A simple, beginner-friendly attendance system built with Python:

- **Flask** admin panel to add / view / delete students.
- **OpenCV + face_recognition (dlib)** to recognise faces from a webcam, RTSP, or HTTP IP-camera stream.
- **SQLite** for student data and **CSV** for attendance logs.

## Project Structure

```
face-attendance/
├── app.py              # Flask admin panel
├── main.py             # Face recognition + attendance runner
├── requirements.txt
├── students/           # Uploaded student face images
├── static/
│   └── style.css       # Admin panel styling
├── templates/          # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── students.html
│   ├── add_student.html
│   └── attendance.html
├── database.db         # Created automatically (SQLite)
└── attendance.csv      # Created automatically (attendance log)
```

## 1. Install Dependencies

You need **Python 3.9+**. The `face_recognition` library depends on **dlib**, which on a fresh machine usually needs a C/C++ build toolchain and CMake installed.

### Windows
1. Install Python 3.10+ from python.org.
2. Install Visual Studio Build Tools (C++ workload).
3. Install CMake from cmake.org and tick "Add to PATH".
4. In a terminal:
   ```bash
   pip install -r requirements.txt
   ```

### macOS
```bash
brew install cmake
pip install -r requirements.txt
```

### Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
pip install -r requirements.txt
```

> Tip: if `pip install face_recognition` is too slow, install `dlib` first
> with `pip install dlib` to see clearer build errors.

## 2. Run the Admin Panel

```bash
cd face-attendance
python app.py
```

Then open <http://localhost:5000> in your browser:

- **Dashboard** – quick stats.
- **Students → Add Student** – register Name, Roll Number and upload a clear face photo.
- **Students** – view all students; delete with one click.
- **Attendance** – browse all attendance records.

Images are saved into `students/` and student records into `database.db`.

## 3. Run the Face Recognition System

In a **separate terminal**:

```bash
# Default webcam
python main.py

# A different webcam (index 1, 2, ...)
python main.py --source 1

# RTSP IP camera
python main.py --source rtsp://username:password@192.168.1.10:554/stream

# HTTP MJPEG / phone IP camera
python main.py --source http://192.168.1.10:8080/video
```

A window will open showing the live video with bounding boxes:

- **Green box** – recognised student (Name + Roll Number).
- **Red box** – unknown face.

Each recognised student is logged once per day to `attendance.csv` with date and time. Press **`q`** in the video window to quit.

## How It Works (Beginner-Friendly)

1. `app.py` exposes a small Flask web app to manage students. Uploaded images go to `students/` and metadata to a tiny SQLite database `database.db`.
2. `main.py` reads every student from the database, computes a 128-dimension face *encoding* with `face_recognition`, and keeps them in memory.
3. It then opens your camera with OpenCV (`cv2.VideoCapture`).
4. For every frame: it shrinks the frame for speed, finds faces, encodes them, and compares each one to the known encodings using Euclidean distance.
5. The best match below the tolerance (`0.5`) becomes the label; everything else is "Unknown".
6. The first time a known student is seen on a given day, a row is written to `attendance.csv`.

## Tips

- Use a **clear, front-facing** photo with good lighting for each student.
- Only one face per registration photo will be encoded (the first one detected).
- Tweak `MATCH_TOLERANCE` and `RESIZE_SCALE` in `main.py` to trade accuracy for speed.
- The system avoids duplicate attendance per **day** per student. Delete `attendance.csv` to start fresh.
