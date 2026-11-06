import os
import json
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from streamlit_autorefresh import st_autorefresh

load_dotenv()

# ==========================================
# 1. VISION ENGINE CONSTANTS & SETUP
# ==========================================
mp_face_mesh = mp.solutions.face_mesh

# Key points: Nose tip, chin, L eye corner, R eye corner, L mouth corner, R mouth corner
LANDMARK_IDS = [1, 199, 33, 263, 61, 291]

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Left eye corner
    (43.3, 32.7, -26.0),      # Right eye corner
    (-28.9, -28.9, -24.1),    # Left mouth corner
    (28.9, -28.9, -24.1),     # Right mouth corner
], dtype=np.float64)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

YAW_THRESHOLD = 0.5    # radians (~28 degrees)
PITCH_THRESHOLD = 0.6  # radians (~34 degrees)
EAR_THRESHOLD = 0.18   # eye aspect ratio threshold for drowsiness/eye closure


@st.cache_resource
def get_face_mesh():
    """Initializes and caches the MediaPipe FaceMesh solution."""
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# ==========================================
# 2. COMPUTER VISION UTILITIES
# ==========================================
def get_head_pose(face_landmarks, frame_shape):
    """
    Computes Euler angles (pitch, yaw, roll) using solvePnP.
    Decomposition matches OpenCV's coordinate conventions:
      - Pitch: Rotation around X-axis (looking up/down)
      - Yaw:   Rotation around Y-axis (looking left/right)
      - Roll:  Rotation around Z-axis (head tilting)
    """
    h, w = frame_shape[:2]
    image_points = []
    for idx in LANDMARK_IDS:
        lm = face_landmarks.landmark[idx]
        image_points.append((lm.x * w, lm.y * h))
    image_points = np.array(image_points, dtype=np.float64)

    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)

    pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
    yaw = np.arctan2(-rotation_mat[2, 0], sy)
    roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])

    return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}


def eye_aspect_ratio(landmarks, eye_indices, frame_shape):
    """Calculates Eye Aspect Ratio (EAR) for blink and drowsiness detection."""
    h, w = frame_shape[:2]
    points = []
    for idx in eye_indices:
        lm = landmarks.landmark[idx]
        points.append(np.array([lm.x * w, lm.y * h]))

    v1 = np.linalg.norm(points[1] - points[5])
    v2 = np.linalg.norm(points[2] - points[4])
    h_dist = np.linalg.norm(points[0] - points[3])

    if h_dist == 0:
        return 0.0
    return float((v1 + v2) / (2.0 * h_dist))


# ==========================================
# 3. CENTROID TRACKER (WITH PRUNING)
# ==========================================
class FaceCentroidTracker:
    """
    Pairs detected face centroids across frames via minimum Euclidean distance.
    Prunes tracks when a face goes unseen for more than max_disappeared frames.
    """
    def __init__(self, max_distance=150.0, max_disappeared=5):
        self.next_id = 1
        self.tracked_faces = {}        # {sid: (cx, cy)}
        self.disappeared_faces = {}    # {sid: count}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def update(self, current_centroids):
        if not current_centroids:
            for sid in list(self.disappeared_faces.keys()):
                self.disappeared_faces[sid] += 1
                if self.disappeared_faces[sid] > self.max_disappeared:
                    self._deregister(sid)
            return []

        if not self.tracked_faces:
            assigned_ids = []
            for pt in current_centroids:
                assigned_ids.append(self._register(pt))
            return assigned_ids

        existing_ids = list(self.tracked_faces.keys())
        existing_coords = np.array([self.tracked_faces[i] for i in existing_ids])
        curr_coords = np.array(current_centroids)

        # Pairwise distance matrix (existing x current)
        D = np.linalg.norm(existing_coords[:, np.newaxis] - curr_coords, axis=2)

        # Match smallest distances first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_ids = [None] * len(current_centroids)
        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            matched_id = existing_ids[row]
            assigned_ids[col] = matched_id
            self.tracked_faces[matched_id] = current_centroids[col]
            self.disappeared_faces[matched_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # Age out unassigned existing faces
        unused_rows = set(range(len(existing_ids))) - used_rows
        for row in unused_rows:
            sid = existing_ids[row]
            self.disappeared_faces[sid] += 1
            if self.disappeared_faces[sid] > self.max_disappeared:
                self._deregister(sid)

        # Register new centroids
        unused_cols = set(range(len(current_centroids))) - used_cols
        for col in unused_cols:
            assigned_ids[col] = self._register(current_centroids[col])

        return assigned_ids

    def _register(self, pt):
        sid = self.next_id
        self.tracked_faces[sid] = pt
        self.disappeared_faces[sid] = 0
        self.next_id += 1
        return sid

    def _deregister(self, sid):
        del self.tracked_faces[sid]
        del self.disappeared_faces[sid]

    def reset(self):
        self.next_id = 1
        self.tracked_faces.clear()
        self.disappeared_faces.clear()


def process_frame(frame, face_mesh_instance, tracker=None):
    """Processes a single BGR frame in memory and yields engagement metrics."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh_instance.process(rgb)

    if not results.multi_face_landmarks:
        if tracker is not None:
            tracker.update([])
        return []

    raw_data = []
    centroids = []

    for face_landmarks in results.multi_face_landmarks:
        pose = get_head_pose(face_landmarks, frame.shape)
        ear_left = eye_aspect_ratio(face_landmarks, LEFT_EYE, frame.shape)
        ear_right = eye_aspect_ratio(face_landmarks, RIGHT_EYE, frame.shape)
        ear_avg = (ear_left + ear_right) / 2.0

        engaged = (
            pose is not None
            and abs(pose["yaw"]) < YAW_THRESHOLD
            and abs(pose["pitch"]) < PITCH_THRESHOLD
            and ear_avg > EAR_THRESHOLD
        )

        nose = face_landmarks.landmark[1]
        centroids.append((nose.x * w, nose.y * h))

        raw_data.append({
            "yaw": round(pose["yaw"], 3) if pose else None,
            "pitch": round(pose["pitch"], 3) if pose else None,
            "ear": round(ear_avg, 3),
            "engaged": bool(engaged),
        })

    if tracker is not None:
        assigned_ids = tracker.update(centroids)
    else:
        assigned_ids = list(range(1, len(raw_data) + 1))

    students = []
    for i, item in enumerate(raw_data):
        item["student_id"] = assigned_ids[i]
        students.append(item)

    students.sort(key=lambda s: s["student_id"])
    return students


# ==========================================
# 4. GROQ ADVISOR ENGINE
# ==========================================
def _get_groq_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or Streamlit secrets.")
    return Groq(api_key=key)


def generate_advice(history):
    """Summarizes engagement metrics and prompts Groq Llama 3.3 for pedagogical advice."""
    if not history:
        return "No data collected yet."

    total_snapshots = len(history)
    student_ids = sorted({s["student_id"] for snap in history for s in snap})

    summary = {}
    for sid in student_ids:
        engaged = []
        ears = []
        for snap in history:
            for s in snap:
                if s["student_id"] == sid:
                    engaged.append(1 if s["engaged"] else 0)
                    ears.append(s["ear"])
        if engaged:
            summary[sid] = {
                "engagement_rate": round(sum(engaged) / len(engaged), 2),
                "avg_ear": round(sum(ears) / len(ears), 3),
                "appearances": len(engaged),
            }

    prompt = f"""You are an educational consultant analysing anonymised classroom engagement data.

Total snapshots: {total_snapshots}
Per-student summary (engagement_rate is 0.0-1.0; avg_ear below 0.18 suggests drowsiness):
{json.dumps(summary, indent=2)}

Tasks:
1. Identify any student with engagement_rate < 0.5 or avg_ear < 0.18.
2. Suggest ONE concrete pedagogical adjustment the teacher can make in the next 10 minutes.
Keep the response under 150 words. Be specific, encouraging, and actionable.
"""

    client = _get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a concise, practical teaching assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content


# ==========================================
# 5. STREAMLIT APPLICATION & DASHBOARD
# ==========================================
st.set_page_config(page_title="ClassroomPro", layout="wide", page_icon="🎓")
st.title("🎓 ClassroomPro - Live Engagement Dashboard")
st.caption("Privacy by Design: Raw video is analyzed in memory and immediately discarded. Only numerical metrics persist.")

face_mesh = get_face_mesh()

# Initialize per-session state
if "history" not in st.session_state:
    st.session_state.history = []
if "running" not in st.session_state:
    st.session_state.running = False
if "advice" not in st.session_state:
    st.session_state.advice = None
if "tracker" not in st.session_state:
    st.session_state.tracker = FaceCentroidTracker()

# Dashboard controls
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("▶️ Start Session", use_container_width=True):
        st.session_state.running = True
        st.session_state.history = []
        st.session_state.advice = None
        st.session_state.tracker.reset()
with c2:
    if st.button("⏸️ Stop Session", use_container_width=True):
        st.session_state.running = False
with c3:
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.running = False
        st.session_state.history = []
        st.session_state.advice = None
        st.session_state.tracker.reset()

status_area = st.empty()
metrics_area = st.empty()

# Continuous capture loop
if st.session_state.running:
    st_autorefresh(interval=5000, key="classroompro_capture_tick")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_area.error("Cannot access camera (index 0). Please check device permissions.")
        st.session_state.running = False
    else:
        # Flush initial frames to let auto-exposure adjust
        for _ in range(3):
            cap.read()

        # Capture a 3-frame burst to eliminate single-blink anomalies
        burst_snapshots = []
        for _ in range(3):
            ret, frame = cap.read()
            if ret:
                burst_snapshots.append(process_frame(frame, face_mesh, st.session_state.tracker))
        cap.release()

        # Take the maximum EAR per student across the burst
        students = []
        if burst_snapshots:
            valid_bursts = [b for b in burst_snapshots if b]
            if valid_bursts:
                base = {s["student_id"]: s for s in valid_bursts[-1]}
                for frame_result in valid_bursts:
                    for s in frame_result:
                        sid = s["student_id"]
                        if sid in base and s["ear"] > base[sid]["ear"]:
                            base[sid]["ear"] = s["ear"]
                            base[sid]["engaged"] = base[sid]["engaged"] or s["engaged"]
                students = list(base.values())

        if students:
            st.session_state.history.append(students)

        with metrics_area.container():
            st.subheader(f"Snapshot #{len(st.session_state.history)}")
            if not students:
                st.info("No students detected in current frame.")
            else:
                cols = st.columns(min(len(students), 5))
                for i, s in enumerate(students):
                    with cols[i % len(cols)]:
                        label = f"Student {s['student_id']}"
                        value = "✅ Engaged" if s["engaged"] else "⚠️ Disengaged"
                        st.metric(
                            label=label,
                            value=value,
                            delta=f"EAR {s['ear']:.2f}"
                        )

# Post-Session Aggregation & AI Insights
if not st.session_state.running and st.session_state.history:
    st.divider()
    st.subheader("📊 Session Aggregate Summary")
    st.write(f"Total Snapshots Captured: **{len(st.session_state.history)}**")

    student_ids = sorted({s["student_id"] for snap in st.session_state.history for s in snap})
    summary_records = []

    for sid in student_ids:
        eng_flags = []
        ears = []
        for snap in st.session_state.history:
            for s in snap:
                if s["student_id"] == sid:
                    eng_flags.append(1 if s["engaged"] else 0)
                    ears.append(s["ear"])

        if eng_flags:
            rate = sum(eng_flags) / len(eng_flags)
            avg_ear = sum(ears) / len(ears)
            summary_records.append({
                "Student ID": sid,
                "Engagement Rate (%)": round(rate * 100, 1),
                "Average EAR": round(avg_ear, 3),
                "Snapshots Observed": len(eng_flags)
            })
            st.progress(rate, text=f"Student {sid}: {rate * 100:.1f}% Engagement")

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        csv_data = df_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Session Metrics (CSV)",
            data=csv_data,
            file_name="classroompro_session_metrics.csv",
            mime="text/csv",
        )

    if st.button("🤖 Generate AI Teaching Advice", type="primary"):
        with st.spinner("Synthesizing metrics via Groq Llama 3.3..."):
            try:
                st.session_state.advice = generate_advice(st.session_state.history)
            except Exception as e:
                st.session_state.advice = f"Error querying advisor: {e}"

    if st.session_state.advice:
        st.markdown("### 💡 AI Recommendations")
        st.info(st.session_state.advice)