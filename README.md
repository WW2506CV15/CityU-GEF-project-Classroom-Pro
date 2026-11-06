# CityU-GEF-project-Classroom-Pro
# ClassroomPro

**An AI-powered classroom engagement monitoring and teaching advisory system.**

ClassroomPro is a computer vision-based application designed to help educators gain insights into classroom engagement through real-time facial landmark analysis.

Using **OpenCV** and **MediaPipe**, the system extracts facial landmarks from webcam frames to estimate head orientation and eye activity. These measurements are used to calculate engagement indicators, which are presented through an interactive **Streamlit dashboard**.

Additionally, ClassroomPro integrates **Groq's LLM API** to analyse aggregated session statistics and generate actionable teaching recommendations.

## Key Features

- **Real-time Face Detection:** Detects up to 10 faces in a webcam frame using MediaPipe Face Mesh.
- **Head Pose Estimation:** Estimates yaw, pitch and roll to analyse head orientation.
- **Eye Aspect Ratio (EAR):** Measures eye openness as an indicator of potential drowsiness.
- **Engagement Classification:** Uses configurable thresholds to classify detected faces as engaged or not engaged.
- **Interactive Dashboard:** Displays live engagement metrics and session summaries through Streamlit.
- **AI Teaching Advisory:** Integrates Groq's LLM API to analyse aggregated session data and suggest pedagogical adjustments.
- **Privacy-conscious Processing:** Processes webcam frames in memory without saving raw images to disk.

## System Architecture

The application consists of three main components:

1. **Computer Vision Engine (`compvision.py`)**
   - Captures and processes facial landmarks.
   - Estimates head pose using OpenCV's `solvePnP`.
   - Calculates Eye Aspect Ratio (EAR).
   - Determines engagement indicators using predefined thresholds.

2. **AI Advisory Engine (`advisor.py`)**
   - Aggregates engagement statistics across session snapshots.
   - Constructs a structured prompt containing numerical data.
   - Communicates with Groq's LLM API.
   - Generates concise and actionable teaching recommendations.

3. **Web Dashboard (`dashboard.py`)**
   - Provides an interactive Streamlit interface.
   - Controls webcam sessions.
   - Displays live metrics and engagement summaries.
   - Presents AI-generated teaching recommendations.

## Project Structure

```text
ClassroomPro/
│
├── .env                 # Environment variables (API keys)
├── .gitignore           # Git ignore configuration
│
├── compvision.py        # Computer vision and facial landmark analysis
├── advisor.py           # Session data aggregation and Groq AI integration
├── dashboard.py         # Main Streamlit dashboard
├── tracker.py           # Core tracking logic / execution
└── setup.py             # Project configuration / setup
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Image processing and head pose estimation |
| MediaPipe | Facial landmark detection |
| NumPy | Numerical computations |
| Streamlit | Interactive web dashboard |
| Groq API | LLM-powered teaching recommendations |
| python-dotenv | Environment variable management |
| streamlit-autorefresh | Automatic dashboard refreshing |

## Installation and Setup

### 1. Prerequisites

- Python 3.11 or 3.12
- A working webcam
- A Groq API key (for AI advisory functionality)

### 2. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ClassroomPro
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install opencv-python mediapipe numpy streamlit groq python-dotenv streamlit-autorefresh
```

### 5. Configure environment variables

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Make sure `.env` is included in `.gitignore` to prevent accidentally exposing your API key.

## Running the Application

Start the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The application will be available at:

```text
http://localhost:8501
```

### Basic Workflow

1. Launch the dashboard.
2. Click **Start Session** to begin webcam-based analysis.
3. Allow the application to process classroom snapshots.
4. Monitor the engagement metrics displayed on the dashboard.
5. Click **Stop Session** to view the session summary.
6. Select **Generate AI Teaching Advice** to obtain recommendations based on the collected statistics.

## Testing

ClassroomPro includes two standalone verification scripts.

**1. Test the Computer Vision Engine**

```bash
python tracker.py
```

Verifies:
- Webcam accessibility.
- Facial landmark detection.
- Head pose estimation.
- Eye Aspect Ratio calculations.
- Engagement classification.

**2. Test the AI Advisory Engine**

```bash
python test_advisor.py
```

Uses fabricated session data to verify:
- Engagement statistics aggregation.
- Groq API communication.
- AI-generated teaching recommendations.

A valid `GROQ_API_KEY` is required for the AI advisory test.

## Privacy and Limitations

ClassroomPro is designed with privacy-conscious data processing in mind.

- Raw webcam frames are processed in memory and are not saved to disk.
- Only numerical measurements and session statistics are retained during the application session.
- Aggregated numerical statistics are sent to the Groq API when generating teaching recommendations.

**Important limitations:**

- Engagement classification is based on observable facial indicators, not direct measurements of attention, understanding or cognitive engagement.
- Head orientation and eye openness are imperfect proxies and may produce inaccurate classifications.
- Detection performance can vary with lighting, camera positioning and individual differences.
- The system should be used as an educational support tool rather than a definitive assessment of individual students.

Appropriate consent, transparency and safeguards should be established before using the system in an actual classroom.

## Future Improvements

Potential areas for further development include:

- Individual calibration of facial measurement thresholds.
- More robust temporal analysis of engagement indicators.
- Improved tracking consistency across consecutive snapshots.
- Expanded session analytics and visualisation.
- Further evaluation of classification accuracy under different classroom conditions.

---
