"""Stage 0 probe: can a public pretrained detector find the ball in the tactical video?

    python possession/cv_probe.py                      # first 5 minutes of play, every 5th frame
    python possession/cv_probe.py --model yolo11s.pt   # a smaller, faster model

No labels are involved. The reference is the provided TRACKING: where the tracker sees the ball,
we project its pitch position into the image with the homography and ask whether the detector
found something there. That tells us how often the detector finds a real ball (recall against
the tracker) and how often it fires somewhere else (likely false hits). On frames where the
tracker has lost the ball, we only count how often the detector finds a candidate.

A few annotated frames are written to data/derived/cv_probe/ for a visual check (that folder is
ignored by git: it contains video frames, which must never be published).
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import sources as D

SPORTS_BALL = 32          # COCO class id for "sports ball"


def project(H9, x, y):
    """Normalised pitch (0..1) -> image pixels, for one 3x3 matrix stored as 9 numbers."""
    m = H9.reshape(3, 3)
    w = m[2, 0] * x + m[2, 1] * y + m[2, 2]
    return (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / w, (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=11185)
    parser.add_argument("--end", type=int, default=18685)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--radius", type=float, default=25.0, help="pixels for a hit")
    parser.add_argument("--video", default=os.path.join(D.ROOT, "data", "tactical", "BAR-ATH_tactical.mp4"))
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: {}   model: {}   image size: {}   confidence >= {}".format(
        device, args.model, args.imgsz, args.conf))

    if not os.path.isfile(args.video):
        raise SystemExit("Video not found: {}".format(args.video))
    cap = cv2.VideoCapture(args.video)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Video: {} x {}, {:.2f} fps, {} frames (submission has {})".format(
        width, height, fps, n_video, D.N_FRAMES))
    if n_video != D.N_FRAMES:
        print("  WARNING: frame count differs from the submission length; check the file")

    ball = np.load(os.path.join(D.DERIVED_DIR, "atd.npz"))["ball"]
    H, _, valid = D.load_homography(D.require(["homography"])[0])
    model = YOLO(args.model)

    out_dir = os.path.join(D.DERIVED_DIR, "cv_probe")
    os.makedirs(out_dir, exist_ok=True)
    saved = {"hit": 0, "miss": 0, "found_when_lost": 0}

    wanted = set(range(args.start, args.end, args.step))
    records = []   # per processed frame: (tracked, [(conf, dist_px or nan, on_pitch)])
    seen_n = hit_n = det_on_seen = far_on_seen = 0
    lost_n = found_on_lost = 0
    dists = []
    began, run_time = time.time(), 0.0

    t = 0
    while t < args.end:
        ok = cap.grab()
        if not ok:
            raise SystemExit("Video ended early at frame {}".format(t))
        if t in wanted:
            ok, frame = cap.retrieve()
            tic = time.time()
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, classes=[SPORTS_BALL],
                                device=device, verbose=False)[0]
            run_time += time.time() - tic
            boxes = res.boxes.xywh.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
            confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
            centres = boxes[:, :2]

            bx, by = ball[t, 0], ball[t, 1]
            tracked = np.isfinite(bx) and np.isfinite(by) and valid[t]
            # every detection's pitch position, via the inverse homography
            Hinv = np.linalg.inv(H[t].reshape(3, 3)) if valid[t] else None
            dets = []
            for (cx, cy, _, _), c in zip(boxes, confs):
                on_pitch = False
                if Hinv is not None:
                    q = Hinv @ np.array([cx, cy, 1.0])
                    px_, py_ = q[0] / q[2] * D.PITCH_LENGTH, q[1] / q[2] * D.PITCH_WIDTH
                    on_pitch = (-3 <= px_ <= D.PITCH_LENGTH + 3) and (-3 <= py_ <= D.PITCH_WIDTH + 3)
                dets.append([float(c), np.nan, on_pitch])
            if tracked:
                uu, vv = project(H[t], float(bx), float(by))
                for d_, (cx, cy, _, _) in zip(dets, boxes):
                    d_[1] = float(np.hypot(cx - uu, cy - vv))
            records.append((tracked, dets))
            label = None
            if tracked:
                u, v = project(H[t], float(bx), float(by))
                seen_n += 1
                if len(centres):
                    det_on_seen += 1
                    d = np.hypot(centres[:, 0] - u, centres[:, 1] - v)
                    dists.append(d.min())
                    if d.min() <= args.radius:
                        hit_n += 1
                        label = "hit"
                    far_on_seen += int(np.sum(d > 3 * args.radius))
                if label is None:
                    label = "miss"
            else:
                lost_n += 1
                if len(centres):
                    found_on_lost += 1
                    label = "found_when_lost"

            if label and saved[label] < 4:
                img = frame.copy()
                for (cx, cy, w, h), c in zip(boxes, confs):
                    cv2.rectangle(img, (int(cx - w / 2) - 4, int(cy - h / 2) - 4),
                                  (int(cx + w / 2) + 4, int(cy + h / 2) + 4), (0, 0, 255), 2)
                    cv2.putText(img, "{:.2f}".format(c), (int(cx) + 8, int(cy) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if tracked:
                    cv2.circle(img, (int(u), int(v)), int(args.radius), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(out_dir, "{}_{}.jpg".format(label, t)), img)
                saved[label] += 1
        t += 1

    print("\nProcessed {} frames in {:.0f} s; detector alone {:.1f} frames/s".format(
        len(wanted), time.time() - began, len(wanted) / max(run_time, 1e-6)))
    print("\nWHERE THE TRACKER SEES THE BALL ({} frames)".format(seen_n))
    if seen_n:
        print("  detector found the same ball (within {:.0f} px)   {:5.1f}%   <- recall vs tracker".format(
            args.radius, 100 * hit_n / seen_n))
        print("  detector found anything at all                {:5.1f}%".format(100 * det_on_seen / seen_n))
        print("  detections far from the tracked ball          {:.2f} per frame   <- likely false hits".format(
            far_on_seen / seen_n))
        if dists:
            print("  median distance to the tracked ball           {:.0f} px".format(np.median(dists)))
    print("\nWHERE THE TRACKER HAS LOST THE BALL ({} frames)".format(lost_n))
    if lost_n:
        print("  detector found a ball candidate               {:5.1f}%".format(100 * found_on_lost / lost_n))
    print("\nExample frames saved to {}".format(os.path.relpath(out_dir, D.ROOT)))
    print("  green circle = tracked ball, red box = detector\n")
    print("\nTHRESHOLD TABLE (detections more than 3 m outside the pitch removed;")
    print("'top' = only the single most confident detection per frame is kept)")
    print("  conf  | seen: top is the ball  any hit  false/frame | lost: has a candidate")
    for thr in (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60):
        s_n = top_hit = any_hit = false = with_det = 0
        l_n = l_det = 0
        for tracked, dets in records:
            kept = [d for d in dets if d[0] >= thr and d[2]]
            if tracked:
                s_n += 1
                if kept:
                    with_det += 1
                    best = max(kept, key=lambda d: d[0])
                    top_hit += int(best[1] <= args.radius)
                    any_hit += int(any(d[1] <= args.radius for d in kept))
                    false += sum(d[1] > 3 * args.radius for d in kept)
            else:
                l_n += 1
                l_det += int(bool(kept))
        print("  {:.2f}  |   {:5.1f}%  (precision {:5.1f}%)  {:5.1f}%   {:5.2f}     |   {:5.1f}%".format(
            thr, 100 * top_hit / max(s_n, 1), 100 * top_hit / max(with_det, 1),
            100 * any_hit / max(s_n, 1), false / max(s_n, 1), 100 * l_det / max(l_n, 1)))
    print("  'top is the ball' = share of tracked frames where the kept top detection is the ball;")
    print("  precision = the same, among frames where something was kept\n")
    full = int(0.41 * 158857)
    if run_time:
        print("Estimated time for all ~{} ball missing scored frames: {:.0f} min".format(
            full, full / (len(wanted) / run_time) / 60))


if __name__ == "__main__":
    main()
