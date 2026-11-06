# CityU-GEF-project-Classroom-Pro
ClassroomPro/
├── .env                  # Environment file containing secret keys (GROQ_API_KEY)
├── .gitignore            # Git ignore configuration
├── vision_engine.py      # Core computer vision pipeline (MediaPipe, pose & EAR)
├── advisor.py            # History aggregation and Groq LLM advisory client
├── app.py                # Main Streamlit web dashboard application
├── test_vision.py        # Standalone verification script for webcam metrics
└── test_advisor.py       # Standalone verification script for AI suggestions
```[cite: 1]

---

## Prerequisites & Requirements

* **Python**: Version 3.11 or 3.12[cite: 1].
* **Required Packages**:
  * `opencv-python`[cite: 1]
  * `mediapipe`[cite: 1]
  * `numpy`[cite: 1]
  * `streamlit`[cite: 1]
  * `groq`[cite: 1]
  * `python-dotenv`[cite: 1]
  * `streamlit-autorefresh`[cite: 1]

---

### 1. Create Project Directory & Virtual Environment

* **Windows (Command Prompt / CMD)**[cite: 1]:
  ```cmd
  mkdir ClassroomPro
  cd ClassroomPro
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```[cite: 1]

* **Windows (PowerShell)**[cite: 1]:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```[cite: 1]

* **macOS / Linux**[cite: 1]:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```[cite: 1]

### 2. Install Dependencies

```cmd
pip install --upgrade pip
pip install opencv-python mediapipe numpy streamlit groq python-dotenv streamlit-autorefresh
```[cite: 1]

### 3. Environment Variables Configuration

Create a file named `.env` in the root folder[cite: 1]:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```[cite: 1]

Create a `.gitignore` file to keep secrets out of version control[cite: 1]:
```text
.venv/
.env
__pycache__/
*.pyc
.DS_Store
```[cite: 1]

---

## Running the Application

### 1. Test Component Engines Standalone

Verify vision engine frame extraction[cite: 1]:
```cmd
python test_vision.py
```[cite: 1]

Verify AI teaching advice generation[cite: 1]:
```cmd
python test_advisor.py
```[cite: 1]

### 2. Launch the Streamlit Dashboard

Start the live web UI[cite: 1]:
```cmd
streamlit run app.py
```[cite: 1]

1. Access the web interface at `http://localhost:8501`[cite: 1].
2. Click **▶Start Session** to initiate automatic frame analysis every 5 seconds[cite: 1].
3. View real-time engagement statuses (Engaged vs. Not Engaged) and Eye Aspect Ratio metrics per student[cite: 1].
4. Click **⏸Stop Session** to freeze session history and display aggregate statistics[cite: 1].
5. Click **Generate AI Teaching Advice** to request pedagogical insights based on session metrics[cite: 1].

---

## Calibration Thresholds

Threshold parameters in `vision_engine.py` can be adjusted based on classroom environment and webcam placement[cite: 1]:

| Parameter | Default Value | Description |
|---|---|---|
| `YAW_THRESHOLD` | `0.5` rad (~28°) | Triggers "Not engaged" when student turns head left or right[cite: 1]. |
| `PITCH_THRESHOLD` | `0.6` rad (~34°) | Triggers "Not engaged" when student looks down (e.g., at a phone)[cite: 1]. |
| `EAR_THRESHOLD` | `0.18` | Values below `0.18` indicate closed eyes or drowsiness[cite: 1]. |

---

## 🔧 Troubleshooting Guide

| Issue | Cause | Resolution |
|---|---|---|
| `'source' is not recognized` | Running Linux syntax on Windows CMD | Use `.venv\Scripts\activate.bat`[cite: 1]. |
| `No module named 'mediapipe'` | Virtual environment not active | Ensure `(.venv)` prompt is active before running commands[cite: 1]. |
| `cv2.VideoCapture(0)` error | Camera in use or wrong index | Close other webcam apps or test camera indices `1` or `2`[cite: 1]. |
| `GROQ_API_KEY is not set` | `.env` missing or in wrong path | Place `.env` in the root folder alongside `advisor.py`[cite: 1]. |
| Groq API `403 Forbidden` | Geographic access restriction | Connect via a US/EU VPN server[cite: 1]. |