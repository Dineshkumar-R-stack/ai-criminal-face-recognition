# ----------------------------
# FIX OpenMP ERROR
# ----------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
import time

from detector import Detector
from identifier import Identifier
from tracker import Sort, score_board

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "D:/Mini_project/focus_face/data"

# ----------------------------
# LOAD MODELS (only once)
# ----------------------------
@st.cache_resource
def load_models():
    det = Detector(
        weight_path=DATA_PATH + "/weights/Resnet50_Final.pth",
        model="re50",
        conf_thresh=0.35
    )

    idt = Identifier(
        embed_db_path=DATA_PATH + "/seed/suspect_db.csv",
        n=0,
        idt_res="VGA",
        box_ratio=1.3,
        model="small"
    )

    trk = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # 🔥 SCORE ACCUMULATION SYSTEM
    spt_entry = score_board.SuspectEntry(0.05)

    return det, idt, trk, spt_entry

det, idt, trk, spt_entry = load_models()

# ----------------------------
# UI
# ----------------------------
st.title("🎥 Real-Time Face Recognition System")

st.sidebar.title("Settings")

source = st.sidebar.selectbox("Camera Source", ["Webcam", "IP Camera"])

ip_url = ""
if source == "IP Camera":
    ip_url = st.sidebar.text_input(
        "IP Camera URL",
        "http://192.168.42.43:8080/video"
    )

run = st.toggle("Start Camera", value=True)

frame_window = st.empty()

# ----------------------------
# CAMERA INIT (only once)
# ----------------------------
if "cap" not in st.session_state:
        st.session_state.cap = cv2.VideoCapture("http://192.168.42.43:8080/video")
'''if source == "Webcam":
        st.session_state.cap = cv2.VideoCapture(0)
    else:'''
cap = st.session_state.cap
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ----------------------------
# MAIN LOOP
# ----------------------------
if run:
    while True:

        # drop old frames (important for IP cam)
        for _ in range(2):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not working")
            break

        img_raw = frame.copy()

        # ----------------------------
        # RESIZE FOR SPEED
        # ----------------------------
        img_det = cv2.resize(img_raw, (640, 480))

        # ----------------------------
        # DETECTION
        # ----------------------------
        boxes = np.clip(det.run(img_det), 0., 1.)

        # ----------------------------
        # SCALE BACK
        # ----------------------------
        if len(boxes) > 0:
            scale_x = img_raw.shape[1] / img_det.shape[1]
            scale_y = img_raw.shape[0] / img_det.shape[0]

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        # ----------------------------
        # TRACKING
        # ----------------------------
        tracked = trk.update(boxes)

        # ----------------------------
        # IDENTIFICATION
        # ----------------------------
        results = idt.run(img_raw, tracked)

        # ----------------------------
        # DRAW RESULTS WITH ACCUMULATION
        # ----------------------------
        for box, score, tid, fid, dist, std_score in results:

            x1, y1, x2, y2 = map(int, box)

            # total score (same as original pipeline)
            total_score = ((score + std_score + (1 - (dist * 1.65))) / 6.0) ** 3

            lets_report, suspect = spt_entry.register(
                score_board.SuspectFace(tid, fid, total_score, box)
            )

            if suspect.is_reported():
                # get name from DB
                name = idt.get_df()[idt.get_df()['ID'] == suspect.get_face_id()]['NAME'].values[0]

                cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_raw, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # not confirmed yet
                cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if lets_report:
                suspect.set_reported()

        # ----------------------------
        # DISPLAY
        # ----------------------------
        frame_window.image(img_raw, channels="BGR")

        time.sleep(0.02)

# ----------------------------
# CLEANUP
# ----------------------------
if not run and "cap" in st.session_state:
    st.session_state.cap.release()