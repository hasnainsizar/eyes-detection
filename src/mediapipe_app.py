# --- Quiet logs BEFORE importing mediapipe/tflite ---
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

import cv2, numpy as np, mediapipe as mp
import time, math, statistics, json, glob
from collections import deque
import os.path as osp

mp_face = mp.solutions.face_mesh

# --------- Landmark sets (MediaPipe FaceMesh) ---------
LEFT  = [33,160,158,133,153,144]      # [outer, up1, up2, inner, low2, low1]
RIGHT = [362,385,387,263,373,380]

# --------- Geometry helpers ---------
def EAR(pts):
    # pts: (6,2)
    A = np.linalg.norm(pts[1]-pts[5])
    B = np.linalg.norm(pts[2]-pts[4])
    C = np.linalg.norm(pts[0]-pts[3])
    return (A + B) / (2.0*C + 1e-6)

def to_xy(landmarks, ids, w, h):
    return np.array([[landmarks[i].x*w, landmarks[i].y*h] for i in ids], dtype=float)

def euclid(a, b, keys=("blink_rate","avg_blink_duration_s","perclos")):
    return math.sqrt(sum((a[k]-b[k])**2 for k in keys))

# --------- Auto-pick calibration clips ---------
def _latest_match(patterns):
    cands = []
    for pat in patterns:
        cands.extend(glob.glob(pat, recursive=True))
    if not cands:
        return ""
    return max(cands, key=osp.getmtime)

def autopick_calibration():
    """
    Returns (alert_path, mild_path, drowsy_path)
    Priority A: exact files in data/calib/
    Fallback B: most recent matches under data/raw/** by keywords
    """
    # A) exact names in data/calib/
    a_dir = "data/calib"
    a_alert  = osp.join(a_dir, "alert.mp4")
    a_mild   = osp.join(a_dir, "mildly_drowsy.mp4")
    a_drowsy = osp.join(a_dir, "drowsy.mp4")
    if all(osp.exists(p) for p in (a_alert, a_mild, a_drowsy)):
        return a_alert, a_mild, a_drowsy

    # B) latest matches anywhere under data/raw/**
    b_alert  = _latest_match(["data/raw/**/*alert*.mp4"])
    # handle both 'mildly_drowsy' and 'mild_drowsy'
    b_mild   = _latest_match(["data/raw/**/*mildly*drowsy*.mp4", "data/raw/**/*mild_drowsy*.mp4"])
    # make sure 'drowsy' doesn't accidentally pick 'mildly_drowsy'
    all_drowsy = [p for p in glob.glob("data/raw/**/*drowsy*.mp4", recursive=True)
                  if "mild" not in osp.basename(p)]
    b_drowsy = max(all_drowsy, key=osp.getmtime) if all_drowsy else ""

    if all(p for p in (b_alert, b_mild, b_drowsy)):
        return b_alert, b_mild, b_drowsy

    return "", "", ""

# --------- Features from a (30s) calibration video ---------
def features_from_video(path, ear_thresh=0.25, consec_needed=2, perclos_thresh=0.26):
    """
    Returns:
      blink_rate (per min),
      avg_blink_duration_s,
      perclos (0..1),
      + info fields (duration_s, frames, fps)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = 0
    frames_below = 0
    blink_durations = []
    blink_count = 0
    frames_closed_total = 0

    pcl_thr = perclos_thresh if perclos_thresh is not None else ear_thresh

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            lxy = to_xy(lm, LEFT,  w, h)
            rxy = to_xy(lm, RIGHT, w, h)
            ear = 0.5 * (EAR(lxy) + EAR(rxy))

            # perclos counting
            if ear < pcl_thr:
                frames_closed_total += 1

            # blink detection
            if ear < ear_thresh:
                frames_below += 1
            else:
                if frames_below >= consec_needed:
                    blink_count += 1
                    blink_durations.append(frames_below)
                frames_below = 0

    cap.release()
    duration_s = frames / fps if frames > 0 else 1.0
    blink_rate = (blink_count / (duration_s / 60.0)) if duration_s > 0 else 0.0
    avg_blink_duration_s = (np.mean(blink_durations)/fps) if blink_durations else 0.0
    perclos = frames_closed_total / max(1, frames)

    return {
        "blink_rate": float(round(blink_rate, 2)),
        "avg_blink_duration_s": float(round(avg_blink_duration_s, 3)),
        "perclos": float(round(perclos, 3)),
        "duration_s": float(round(duration_s, 2)),
        "frames": int(frames),
        "fps": float(round(fps, 2)),
    }

# --------- Live demo with rolling window ---------
def open_cam(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap

def live_predict(
    centroids,
    cam_index=0,
    ear_thresh=0.23,
    perclos_thresh=None,
    consec_needed=3,
    window_s=20,
    mirror=True
):
    pcl_thr = perclos_thresh if perclos_thresh is not None else ear_thresh

    cap = open_cam(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Camera not available. Enable Camera permission in System Settings.")

    # Rolling buffers (frame-wise)
    times = deque()
    ear_buf = deque()
    closed_flags = deque()    # 1 if frame considered "closed", else 0

    # Blink state + rolling blink events (time-stamped)
    frames_below = 0
    blink_times = deque()         # timestamps (end time) of each blink
    blink_durs_frames = deque()   # blink durations in frames (aligned with blink_times)

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            now = time.time()

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                lxy = to_xy(lm, LEFT,  w, h)
                rxy = to_xy(lm, RIGHT, w, h)
                ear = 0.5 * (EAR(lxy) + EAR(rxy))

                # Append current frame sample
                times.append(now)
                ear_buf.append(ear)
                closed_flags.append(1 if ear < pcl_thr else 0)

                # Maintain frame-wise window
                while times and (now - times[0]) > window_s:
                    times.popleft(); ear_buf.popleft(); closed_flags.popleft()

                # Blink edge detection (per-frame)
                if ear < ear_thresh:
                    frames_below += 1
                else:
                    if frames_below >= consec_needed:
                        # blink ended; record event with its end time (now)
                        blink_times.append(now)
                        blink_durs_frames.append(frames_below)
                    frames_below = 0

                # Purge blink events outside the window
                while blink_times and (now - blink_times[0]) > window_s:
                    blink_times.popleft()
                    blink_durs_frames.popleft()

                # Compute live features strictly over current window
                win_dur = (times[-1] - times[0]) if len(times) > 1 else 1.0
                perclos_live = (sum(closed_flags) / max(1, len(closed_flags)))

                blinks_in_win = len(blink_times)
                blink_rate_live = (blinks_in_win / (win_dur / 60.0)) if win_dur > 0 else 0.0

                if blinks_in_win > 0:
                    # average blink duration from those blinks that are inside window
                    avg_blink_dur_frames = sum(blink_durs_frames) / blinks_in_win
                else:
                    avg_blink_dur_frames = 0.0
                # Estimate FPS from frame buffer length over window
                est_fps = len(ear_buf) / win_dur if win_dur > 0 else 30.0
                avg_blink_duration_s = avg_blink_dur_frames / max(1e-6, est_fps)

                feat = {
                    "blink_rate": float(round(blink_rate_live, 2)),
                    "avg_blink_duration_s": float(round(avg_blink_duration_s, 3)),
                    "perclos": float(round(perclos_live, 3)),
                }
                # --- Guardrails: override centroid when signal is obvious ---
                mean_closed = feat["perclos"]
                br = feat["blink_rate"]
                avgdur = feat["avg_blink_duration_s"]

                # 1) Strong drowsy: high PERCLOS OR very long average blink
                if mean_closed >= 0.45 or avgdur >= 0.25:
                    pred = "drowsy"
                else:
                    # 2) Strong alert: very low PERCLOS and low blink rate
                    if mean_closed <= 0.08 and br <= 12:
                        pred = "alert"
                    else:
                        # fallback to nearest centroid
                        dists = {k: euclid(feat, v) for k, v in centroids.items()}
                        pred = min(dists, key=dists.get)


                # Nearest centroid
                dists = {k: euclid(feat, v) for k, v in centroids.items()}
                pred = min(dists, key=dists.get)

                # Overlay
                cv2.putText(frame, f"Pred: {pred}", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,255,50), 2)
                cv2.putText(frame, f"blink/min {feat['blink_rate']:.1f}  avg_dur {feat['avg_blink_duration_s']:.2f}s  PERCLOS {feat['perclos']:.2f}",
                            (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"EAR {ear:.3f}", (12, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("Live Drowsiness (q/ESC to quit)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()

# --------- Main (auto-pick enabled) ---------
if __name__ == "__main__":
    # Try to auto-pick calibration clips
    alert_p, mild_p, drowsy_p = autopick_calibration()
    if not (alert_p and mild_p and drowsy_p):
        print("\n[!] No calibration clips found.\n"
              "Put them as either:\n"
              "  A) data/calib/alert.mp4, data/calib/mildly_drowsy.mp4, data/calib/drowsy.mp4\n"
              "or\n"
              "  B) anywhere under data/raw/ with filenames containing:\n"
              "     'alert', 'mildly_drowsy' (or 'mild_drowsy'), and 'drowsy'\n")
        raise SystemExit(1)

    print("[CAL] Using clips:")
    print("  alert :", alert_p)
    print("  mild  :", mild_p)
    print("  drowsy:", drowsy_p)

    # Calibration features
    c_alert  = features_from_video(alert_p,  ear_thresh=0.26, consec_needed=2, perclos_thresh=0.26)
    c_mild   = features_from_video(mild_p,   ear_thresh=0.26, consec_needed=2, perclos_thresh=0.26)
    c_drowsy = features_from_video(drowsy_p, ear_thresh=0.26, consec_needed=2, perclos_thresh=0.26)

    centroids = {
        "alert": c_alert,
        "mildly_drowsy": c_mild,
        "drowsy": c_drowsy,
    }
    for k, v in centroids.items():
        print(f"[CAL] {k:14s} → blink/min {v['blink_rate']:.1f}, avg_dur {v['avg_blink_duration_s']:.2f}s, PERCLOS {v['perclos']:.2f}")

    # Live prediction (no args needed)
    live_predict(
        centroids,
        cam_index=0,
        ear_thresh=0.25,
        perclos_thresh=0.26,
        consec_needed=2,
        window_s=10,
        mirror=True
    )
