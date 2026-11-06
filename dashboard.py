import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from streamlit_autorefresh import st_autorefresh
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
    st.write(f"Total Snapshots Captured: *{len(st.session_state.history)}*")

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