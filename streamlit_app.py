import streamlit as st
import cv2, numpy as np, mediapipe as mp
from ultralytics import YOLO

st.title("Eye Tracking & Face Benchmark")

mode = st.radio("Mode", ["MediaPipe (Eyes/IRIS/EAR)", "YOLO Face (Benchmark)"])

FRAME_WINDOW = st.image(np.zeros((480,640,3), dtype=np.uint8))
cap = cv2.VideoCapture(0)

if mode.startswith("MediaPipe"):
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
else:
    model = YOLO("yolov8n-face.pt")  # set your local path

run = st.checkbox("Start")
while run:
    ok, frame = cap.read()
    if not ok: break
    if mode.startswith("MediaPipe"):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        if res.multi_face_landmarks:
            h,w = frame.shape[:2]
            lm = res.multi_face_landmarks[0].landmark
            for i in [468,469,470,471,472]:
                x,y = int(lm[i].x*w), int(lm[i].y*h)
                cv2.circle(frame,(x,y),2,(0,255,0),-1)
    else:
        res = model.predict(source=frame, imgsz=640, verbose=False)
        for r in res:
            for b in r.boxes.xyxy.cpu().numpy().astype(int):
                x1,y1,x2,y2 = b
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
