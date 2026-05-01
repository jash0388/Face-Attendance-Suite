FROM python:3.11

# Install system dependencies for OpenCV and Dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install build-time requirements
RUN pip install --no-cache-dir setuptools wheel

# Copy requirements and install
COPY face-attendance/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
# We copy everything from the face-attendance folder into the current /app dir
COPY face-attendance/ ./

# Ensure storage directories exist
RUN mkdir -p students static/uploads

# Railway uses the PORT environment variable
ENV PORT=3000
EXPOSE 3000

CMD ["python", "app.py"]
