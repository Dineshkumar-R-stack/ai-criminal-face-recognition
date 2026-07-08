# ----------------------------
# FIX OpenMP ERROR
# ----------------------------
import sqlite3
import requests
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, Response
import cv2
import numpy as np
import time

from detector import Detector
from identifier import Identifier
from tracker import Sort, score_board
# ----------------------------
conn = sqlite3.connect("alerts.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT,
    name TEXT,
    time TEXT,
    image_path TEXT
)
""")
conn.commit()

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "D:/Mini_project/focus_face/data"


# ----------------------------
# LOAD MODELS
# ----------------------------
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

# ✅ FIXED (no keyword argument)
spt_entry = score_board.SuspectEntry(0.05)

# ----------------------------
# CAMERA SOURCE
# ----------------------------
# Webcam:
cap = cv2.VideoCapture("rtsp://admin:123456@192.168.0.52:554/stream2")
'''"rtsp://admin:123456@192.168.0.52:554/stream2"'''
'''cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)'''
# 👉 For IP webcam, use this instead:
# cap = cv2.VideoCapture("http://192.168.1.5:8080/video")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ----------------------------
# FLASK APP
# ----------------------------
app = Flask(__name__)

def send_telegram_alert(name, image_path, timestamp):

    BOT_TOKEN = "8777889185:AAH4UB17HrxJIaSv-CRUalboBHw2p1XG4kw"
    CHAT_ID = 6473363120

    url = f"https://api.telegram.org/bot8777889185:AAH4UB17HrxJIaSv-CRUalboBHw2p1XG4kw/sendPhoto"

    with open(image_path, 'rb') as img:
        response = requests.post(
            url,
            data={
                'chat_id': CHAT_ID,
                'caption': f"🚨 ALERT!\n\nPerson: {name}\nTime: {timestamp}"
            },
            files={'photo': img}
        )

    print("Telegram alert sent:", response.status_code)

def generate_frames():
    while True:

        for _ in range(2):
            cap.grab()

        success, frame = cap.read()
        if not success:
            break

        img_raw = frame.copy()

        img_det = cv2.resize(img_raw, (640, 480))

        boxes = np.clip(det.run(img_det), 0., 1.)

        if len(boxes) > 0:
            scale_x = img_raw.shape[1] / img_det.shape[1]
            scale_y = img_raw.shape[0] / img_det.shape[0]

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        tracked = trk.update(boxes)
        results = idt.run(img_raw, tracked)

        for box, score, tid, fid, dist, std_score in results:

            x1, y1, x2, y2 = map(int, box)

            # ----------------------------
            # SAFE NAME ASSIGNMENT
            # ----------------------------
            name = "UNKNOWN"

            if fid != '-':
                df = idt.get_df()
                match = df[df['ID'] == fid]

                if len(match) > 0:
                    name = match['NAME'].values[0]

            # ----------------------------
            # SCORE
            # ----------------------------
            total_score = ((score + std_score + (1 - (dist * 1.65))) / 6.0) ** 3

            lets_report, suspect = spt_entry.register(
                score_board.SuspectFace(tid, fid, total_score, box)
            )

            # ----------------------------
            # DRAW
            # ----------------------------
            if suspect.is_reported():
                cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_raw, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # ----------------------------
            # ALERT + SAVE (ONLY ONCE)
            # ----------------------------
            if lets_report and not suspect.is_reported():

                suspect.set_reported()

                import datetime

                date_folder = datetime.datetime.now().strftime("%Y-%m-%d")
                folder_path = os.path.join("alerts", date_folder)
                os.makedirs(folder_path, exist_ok=True)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                filename = os.path.join(
                    folder_path,
                    f"{suspect.get_face_id()}_{timestamp}.jpg"
                )

                # safe crop
                if y2 > y1 and x2 > x1:
                    cv2.imwrite(filename, img_raw[y1:y2, x1:x2])

                # 🔥 SAFE TELEGRAM CALL
                try:
                    send_telegram_alert(name, filename, timestamp)
                except Exception as e:
                    print("Telegram error:", e)

        ret, buffer = cv2.imencode('.jpg', img_raw)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.02)


# ----------------------------
# ROUTES
# ----------------------------
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Face Recognition</title>
    </head>
    <body>
        <h2>Real-Time Face Recognition</h2>
        <img src="/video">
    </body>
    </html>
    """

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ----------------------------
# RUN SERVER
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)