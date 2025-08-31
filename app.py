import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import time
import os

# -----------------------------
# Angle Calculation
# -----------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="FitTrack AI", page_icon="💪", layout="wide")

# Title
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color:#FF4B4B;">💪 FitTrack AI</h1>
        <h3 style="color:gray;">Your AI-powered Exercise Buddy</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚡ Settings")
exercise = st.sidebar.radio("Choose Exercise", ["Right Arm Curl", "Squats", "Push-ups"])
st.sidebar.info("📊 Counter will start once camera is ON")

# Sidebar credit
st.sidebar.markdown(
    """
    ---
    **Made by Pranav Jain (TIET)**
    ---
    """
)

# -----------------------------
# Session State
# -----------------------------
if "run" not in st.session_state:
    st.session_state.run = False
if "counter" not in st.session_state:
    st.session_state.counter = 0
if "stage" not in st.session_state:
    st.session_state.stage = "work"
if "progress" not in st.session_state:
    st.session_state.progress = 0

# Buttons
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("▶ Start Camera"):
        st.session_state.run = True
with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.run = False
with col3:
    if st.button("🔄 Reset Counter"):
        st.session_state.counter = 0
        st.session_state.stage = "work"
        st.session_state.progress = 0

# -----------------------------
# Mediapipe Init
# -----------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
FRAME_WINDOW = st.image([])
pTime = 0

cap = cv2.VideoCapture(0)

# -----------------------------
# Pose Detection Loop
# -----------------------------
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.write("❌ Camera not available")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # ---------------- EXERCISE ANGLES ----------------
            if exercise == "Right Arm Curl":
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)
                work_thresh, relax_thresh = 150, 70  

            elif exercise == "Squats":
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                angle = calculate_angle(hip, knee, ankle)
                work_thresh, relax_thresh = 145, 140  # relaxed squats

            elif exercise == "Push-ups":
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                angle = calculate_angle(shoulder, hip, ankle)
                work_thresh, relax_thresh = 160, 130  # relaxed push-ups

            # ---------------- PROGRESS % ----------------
            progress = np.interp(angle, (relax_thresh, work_thresh), (100, 0))
            progress = int(np.clip(progress, 0, 100))
            st.session_state.progress = progress

            # ---------------- REP COUNTING ----------------
            if angle > work_thresh:
                st.session_state.stage = "work"
            elif angle < relax_thresh and st.session_state.stage == "work":
                st.session_state.stage = "relax"
                st.session_state.counter += 1

            # ---------------- STATUS BOX ON VIDEO ----------------
            cv2.rectangle(image, (0,0), (320,120), (245,117,16), -1)
            cv2.putText(image, f'REPS: {st.session_state.counter}', (10,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(image, f'STAGE: {st.session_state.stage}', (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(image, f'{st.session_state.progress}%', (220,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Progress bar
            bar_x, bar_y = 10, 95
            cv2.rectangle(image, (bar_x, bar_y), (bar_x+200, bar_y+20), (255,255,255), 2)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x+2*progress, bar_y+20), (0,255,0), -1)

        except:
            pass

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        FRAME_WINDOW.image(image, channels="BGR")

cap.release()