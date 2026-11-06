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
    sy = np.sqrt(rotation_mat[0, 0] * 2 + rotation_mat[1, 0] * 2)

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
