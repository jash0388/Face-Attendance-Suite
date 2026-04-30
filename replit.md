# Face Recognition Attendance System

A simple, beginner-friendly Python project with two parts:

- **`face-attendance/app.py`** — Flask admin panel (Add / View / Delete students, view attendance).
- **`face-attendance/main.py`** — OpenCV + `face_recognition` runner that opens a webcam / RTSP / HTTP camera, recognises registered students, and writes attendance to `attendance.csv`.

## Project Layout

```
face-attendance/
├── app.py              # Flask admin panel
├── main.py             # Face recognition runner
├── requirements.txt
├── README.md           # Full setup & usage instructions
├── students/           # Stored student face images
├── static/style.css    # Admin panel styling
├── templates/          # Jinja2 HTML templates
├── database.db         # SQLite (auto-created)
└── attendance.csv      # Attendance log (auto-created)
```

## Running

- The Flask admin panel runs in this Repl on port 5000 (workflow: `Start application`).
- The recognition script (`main.py`) needs a real camera, so run it **on your local machine**. See `face-attendance/README.md` for full install + usage instructions.

## Stack

- Python 3.11
- Flask 3
- SQLite (stdlib)
- OpenCV + `face_recognition` (dlib) — for the recognition script
