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
# streamlit run Nov6.py
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