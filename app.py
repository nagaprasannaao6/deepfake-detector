from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import io
import soundfile as sf
import os
import uuid
import cv2
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= AUDIO =================
def detect_audio_logic(audio_bytes):
    data, sr = sf.read(io.BytesIO(audio_bytes))

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)

    data = data.astype(np.float32)
    data = data / (np.max(np.abs(data)) + 1e-6)

    # ✅ Use only first 2 seconds (FASTER)
    data = data[:16000 * 2]

    # FEATURES (optimized)
    mel_spec = librosa.feature.melspectrogram(y=data, sr=16000)
    log_mel = librosa.power_to_db(mel_spec)
    variance = float(np.var(log_mel))

    rms = librosa.feature.rms(y=data)[0]
    temporal_var = float(np.var(rms) * 1000)

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(data)))
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=data, sr=16000)))

    # ✅ Reduced MFCC (faster)
    mfcc = librosa.feature.mfcc(y=data, sr=16000, n_mfcc=5)
    mfcc_var = float(np.var(mfcc))

    # SCORING
    score = 0
    reasons = []

    if variance > 140:
        score += 2
        reasons.append("High spectral variance → natural complexity")
    else:
        reasons.append("Low spectral variance → possible synthetic smoothing")

    if temporal_var > 0.05:
        score += 2
        reasons.append("Dynamic energy variation → human-like speech")
    else:
        reasons.append("Flat energy → AI-like consistency")

    if zcr > 0.05:
        score += 1
        reasons.append("Natural noise patterns detected")

    if mfcc_var > 30:
        score += 1
        reasons.append("Vocal variation detected")

    if spec_centroid > 2000:
        score += 1
        reasons.append("Balanced frequency distribution")

    if score >= 5:
        result = "Real"
    elif score >= 3:
        result = "Suspicious"
    else:
        result = "Fake"

    confidence = round(min(1.0, score / 7), 2)

    return {
        "result": result,
        "confidence": confidence,
        "explanation": reasons,
        "time_series": rms.tolist(),
        "variance": variance,
        "temporal_variance": temporal_var,
        "zcr": zcr,
        "mfcc_variance": mfcc_var,
        "spectral_centroid": spec_centroid
    }

# ================= VIDEO =================
def detect_video_logic(video_file):
    filename = f"{uuid.uuid4()}.mp4"
    path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(path)

    cap = cv2.VideoCapture(path)

    frame_diffs = []
    prev_gray = None

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces_detected = 0
    frames_checked = 0

    # ✅ Optimization settings
    frame_skip = 5
    max_frames = 100
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames
        if frame_count % frame_skip != 0:
            continue

        # Limit frames
        if frame_count > max_frames:
            break

        # Resize for speed
        frame = cv2.resize(frame, (320, 240))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            frame_diffs.append(np.mean(diff))

        prev_gray = gray

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            faces_detected += 1

        frames_checked += 1

    cap.release()
    os.remove(path)

    if len(frame_diffs) == 0:
        return {"result": "Error", "confidence": 0}

    motion_variance = float(np.var(frame_diffs))
    face_ratio = faces_detected / (frames_checked + 1)

    if motion_variance > 2 and face_ratio > 0.5:
        result = "Real Video"
    else:
        result = "Suspicious/Fake Video"

    confidence = round(min(1.0, (motion_variance / 5 + face_ratio) / 2), 2)

    return {
        "result": result,
        "confidence": confidence,
        "motion_variance": motion_variance,
        "face_ratio": face_ratio
    }

# ================= AV SYNC =================
def detect_av_sync(video_file):
    filename = f"{uuid.uuid4()}.mp4"
    path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(path)

    audio_path = path.replace(".mp4", ".wav")

    # ✅ Fixed FFmpeg (portable)
    subprocess.run([
        "ffmpeg",
        "-i", path,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    data, sr = librosa.load(audio_path, sr=16000)

    # Use only first few seconds
    data = data[:16000 * 3]

    rms = librosa.feature.rms(y=data)[0]

    cap = cv2.VideoCapture(path)

    frame_diffs = []
    prev_gray = None

    frame_skip = 5
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (320, 240))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            frame_diffs.append(np.mean(diff))

        prev_gray = gray

    cap.release()

    os.remove(path)
    if os.path.exists(audio_path):
        os.remove(audio_path)

    if len(frame_diffs) == 0:
        return {"result": "Error", "confidence": 0}

    min_len = min(len(rms), len(frame_diffs))
    rms = rms[:min_len]
    frame_diffs = frame_diffs[:min_len]

    correlation = float(np.corrcoef(rms, frame_diffs)[0, 1])
    if np.isnan(correlation):
        correlation = 0

    if correlation > 0.5:
        result = "Synced (Real)"
    elif correlation > 0.2:
        result = "Partially Synced"
    else:
        result = "Not Synced (Fake)"

    return {
        "result": result,
        "confidence": round(abs(correlation), 2),
        "correlation": correlation
    }

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect-audio", methods=["POST"])
def detect_audio():
    return jsonify(detect_audio_logic(request.files["audio"].read()))

@app.route("/detect-video", methods=["POST"])
def detect_video():
    return jsonify(detect_video_logic(request.files["video"]))

@app.route("/detect-sync", methods=["POST"])
def detect_sync():
    return jsonify(detect_av_sync(request.files["video"]))

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
