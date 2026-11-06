
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
