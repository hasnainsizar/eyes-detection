from ultralytics import YOLO
import cv2, time

# NOTE: provide your face weights path (download a YOLO face model, e.g. 'yolov8n-face.pt')
MODEL_PATH = "yolov8n-face.pt"  # replace with your local file

model = YOLO(MODEL_PATH)

def open_cam():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    return cap

cap = open_cam()
if not cap.isOpened():
    raise RuntimeError("Camera not available. Enable Camera permission in System Settings.")

last = time.time(); frames = 0; fps = 0.0
while True:
    ok, frame = cap.read()
    if not ok: break

    res = model.predict(source=frame, imgsz=640, verbose=False)
    # draw boxes
    for r in res:
        for b in r.boxes.xyxy.cpu().numpy().astype(int):
            x1,y1,x2,y2 = b
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # fps calc
    frames += 1
    now = time.time()
    if now - last >= 1.0:
        fps = frames / (now - last)
        frames, last = 0, now

    cv2.putText(frame, f"YOLO Face FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("YOLO Face Benchmark (ESC to exit)", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release(); cv2.destroyAllWindows()
