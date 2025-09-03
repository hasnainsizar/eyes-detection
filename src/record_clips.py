"""
record_clip.py

Record labeled 30s (configurable) webcam clips and (optionally) append a row
to data/metadata.csv that matches your Step 1 schema.

Usage (single clip):
  python src/record_clip.py --subject_id S001 --fatigue alert --attention on_screen \
    --glasses none --lighting normal --duration 30 --append_metadata data/metadata.csv

Usage (batch from CSV):
  # batch.csv headers: fatigue,attention,glasses,lighting,environment,duration
  python src/record_clip.py --subject_id S001 --batch_file batch.csv --append_metadata data/metadata.csv
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import cv2


def iso_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def open_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index)
    # Set props (best-effort; some cams ignore)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        return None
    return cap


def make_writer(path, frame_w, frame_h, fps):
    # Try mp4v first; if it fails to write frames, fallback to avc1 (mac-friendly)
    for fourcc_str in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, (frame_w, frame_h))
        if writer.isOpened():
            return writer
    return None


def record_one_clip(
    subject_id,
    fatigue,
    attention,
    glasses,
    lighting,
    environment="indoor",
    duration=30,
    cam_index=0,
    width=None,
    height=None,
    fps=30,
    mirror=True,
    countdown=3,
    out_root="data/raw",
    append_metadata_path=None,
    device_name="MacBook",
):
    # Paths & names
    out_dir = os.path.join(out_root, subject_id)
    ensure_dir(out_dir)

    start_iso_planned = iso_now()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%MZ")
    base = f"{subject_id}_{fatigue}_{attention}_{glasses}_{lighting}_{environment}_{stamp}"
    out_path = os.path.join(out_dir, f"{base}.mp4")

    # Camera
    cap = open_camera(cam_index, width, height, fps)
    if cap is None:
        print("[ERROR] Could not open camera. Check permissions in macOS System Settings → Privacy & Security → Camera.", file=sys.stderr)
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps_eff = float(cap.get(cv2.CAP_PROP_FPS)) or float(fps) or 30.0

    writer = make_writer(out_path, frame_w, frame_h, fps_eff)
    if writer is None:
        print("[ERROR] Could not create VideoWriter. Try a different codec or path.", file=sys.stderr)
        cap.release()
        return None

    # Countdown overlay
    if countdown and countdown > 0:
        end_cd = time.time() + countdown
        while time.time() < end_cd:
            ret, frame = cap.read()
            if not ret:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            secs = int(end_cd - time.time()) + 1
            cv2.putText(frame, f"Recording in {secs}…", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                writer.release()
                cv2.destroyAllWindows()
                print("[INFO] Aborted during countdown.")
                return None

    print(f"[INFO] Recording {duration}s → {out_path}")
    start_iso = iso_now()
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera frame grab failed; stopping.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            writer.write(frame)
            # Minimal HUD: elapsed seconds
            elapsed = time.time() - t0
            hud = f"{int(elapsed)}s / {duration}s"
            cv2.putText(frame, hud, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Recording", frame)

            if elapsed >= duration:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Stopped by user (q).")
                break
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    end_iso = iso_now()
    actual_dur = round(max(0.0, time.time() - t0), 3)
    print(f"[INFO] Saved: {out_path} (≈{actual_dur}s)")

    # Append metadata row if requested
    if append_metadata_path:
        ensure_dir(os.path.dirname(append_metadata_path))
        row = {
            "subject_id": subject_id,
            "session_id": base,          # using filename stem as session/clip identity
            "clip_id": base,
            "start_time_iso": start_iso,
            "end_time_iso": end_iso,
            "duration_s": actual_dur,
            "fatigue_label": fatigue,
            "attention_label": attention,
            "glasses": glasses,
            "lighting": lighting,
            "environment": environment,
            "device": device_name,
            "camera_fps": fps_eff,
            "resolution_w": frame_w,
            "resolution_h": frame_h,
            "labeler_id": "",            # fill later if needed
            "labeling_method": "manual/record_clip.py_v1",
            "quality_flag": "ok",
            "notes": "",
            "split": "",
        }
        write_header = not os.path.exists(append_metadata_path)
        with open(append_metadata_path, "a", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer_csv.writeheader()
            writer_csv.writerow(row)
        print(f"[INFO] Appended metadata → {append_metadata_path}")

    return out_path


def run_single(args):
    return record_one_clip(
        subject_id=args.subject_id,
        fatigue=args.fatigue,
        attention=args.attention,
        glasses=args.glasses,
        lighting=args.lighting,
        environment=args.environment,
        duration=args.duration,
        cam_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        mirror=not args.no_mirror,
        countdown=args.countdown,
        out_root=args.out_root,
        append_metadata_path=args.append_metadata,
        device_name=args.device_name,
    )


def run_batch(args):
    # CSV with columns: fatigue,attention,glasses,lighting,environment,duration
    with open(args.batch_file, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    for i, r in enumerate(rows, 1):
        print(f"\n[Batch {i}/{len(rows)}]")
        record_one_clip(
            subject_id=args.subject_id,
            fatigue=r["fatigue"],
            attention=r["attention"],
            glasses=r["glasses"],
            lighting=r["lighting"],
            environment=r.get("environment", "indoor"),
            duration=int(r.get("duration", args.duration)),
            cam_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            mirror=not args.no_mirror,
            countdown=args.countdown,
            out_root=args.out_root,
            append_metadata_path=args.append_metadata,
            device_name=args.device_name,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Record labeled webcam clips for eye dataset.")
    p.add_argument("--subject_id", required=True, help="Subject ID, e.g., S001")
    p.add_argument("--fatigue", choices=["alert", "mildly_drowsy", "drowsy"])
    p.add_argument("--attention", choices=["on_screen", "distracted"])
    p.add_argument("--glasses", choices=["none", "clear", "tinted"])
    p.add_argument("--lighting", choices=["bright", "normal", "dim", "backlit", "mixed"])
    p.add_argument("--environment", default="indoor", choices=["indoor", "outdoor", "in_vehicle"])
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no_mirror", action="store_true", help="Do not mirror preview")
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--out_root", default="data/raw")
    p.add_argument("--append_metadata", default=None, help="Path to data/metadata.csv to append rows")
    p.add_argument("--device_name", default="MacBook")
    p.add_argument("--batch_file", default=None, help="CSV with multiple rows to record sequentially")

    args = p.parse_args()

    if args.batch_file:
        run_batch(args)
    else:
        # Require single-clip args if not batch
        missing = [k for k in ("fatigue", "attention", "glasses", "lighting") if getattr(args, k) is None]
        if missing:
            print(f"[ERROR] Missing arguments for single clip: {missing}. Or provide --batch_file.", file=sys.stderr)
            sys.exit(1)
        run_single(args)
