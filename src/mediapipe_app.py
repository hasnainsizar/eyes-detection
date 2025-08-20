import cv2, numpy as np, mediapipe as mp
mp_face = mp.solutions.face_mesh

# EAR landmark sets (MediaPipe indices)
LEFT =  [33, 160, 158, 133, 153, 144]     # [outer, upper1, upper2, inner, lower2, lower1]
RIGHT = [362, 385, 387, 263, 373, 380]

# For pretty drawing (polylines)
LEFT_OUTLINE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
IRIS = [468, 469, 470, 471, 472]  # shared iris landmarks (center ring)


def EAR(pts):
    # pts: (6,2) array in order above
    A = np.linalg.norm(pts[1]-pts[5])
    B = np.linalg.norm(pts[2]-pts[4])
    C = np.linalg.norm(pts[0]-pts[3])
    return (A + B) / (2.0*C + 1e-6)

def to_xy(landmarks, ids, w, h):
    return np.array([[int(landmarks[i].x*w), int(landmarks[i].y*h)] for i in ids], dtype=float)

def draw_poly(frame, pts, closed=True):
    ptsi = pts.astype(int).reshape(-1,1,2)
    cv2.polylines(frame, [ptsi], closed, (0,255,0), 1)

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

blink_count = 0
ear_thresh = 0.23       # tweak per face/lighting
frames_below = 0
consec_needed = 3       # how many consecutive low-EAR frames = blink

with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                      refine_landmarks=True, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as fm:
    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            # outlines
            l_outline = to_xy(lm, LEFT_OUTLINE, w, h)
            r_outline = to_xy(lm, RIGHT_OUTLINE, w, h)
            draw_poly(frame, l_outline); draw_poly(frame, r_outline)

            # iris ring (draw a few points)
            for i in IRIS:
                x,y = int(lm[i].x*w), int(lm[i].y*h)
                cv2.circle(frame, (x,y), 2, (0,255,0), -1)

            # EAR per eye
            lxy = to_xy(lm, LEFT, w, h)
            rxy = to_xy(lm, RIGHT, w, h)
            l_ear = EAR(lxy); r_ear = EAR(rxy)
            ear = (l_ear + r_ear) / 2.0

            # blink logic
            if ear < ear_thresh:
                frames_below += 1
            else:
                if frames_below >= consec_needed:
                    blink_count += 1
                frames_below = 0

            cv2.putText(frame, f"EAR L:{l_ear:.2f} R:{r_ear:.2f}  Blinks:{blink_count}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("MediaPipe Eyes (ESC to exit)", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release(); cv2.destroyAllWindows()
