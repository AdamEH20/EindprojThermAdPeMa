#Position tracking
import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import math


@dataclass
class Detection:
    x: float
    y: float
    r: float
    ok: bool


def circularity(area: float, perimeter: float) -> float:
    if perimeter <= 1e-9:
        return 0.0
    return (4.0 * math.pi * area) / (perimeter * perimeter)


def detect_ball_center(frame_bgr: np.ndarray) -> Detection:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(gray, 50, 140)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    h, w = gray.shape[:2]
    frame_area = float(h * w)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.0005 * frame_area:
            continue

        peri = cv2.arcLength(cnt, True)
        circ = circularity(area, peri)

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < 5:
            continue

        score = circ * 2.0 + (area / frame_area)
        if score > best_score:
            best_score = score
            best = (x, y, r)

    if best is None:
        return Detection(0.0, 0.0, 0.0, False)

    x, y, r = best
    return Detection(float(x), float(y), float(r), True)


def parse_roi(s: str) -> Optional[Tuple[int, int, int, int]]:
    s = s.strip()
    if not s:
        return None
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError('ROI must be "x,y,w,h"')
    return tuple(parts)  # type: ignore


def apply_roi(frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, Tuple[int, int]]:
    if roi is None:
        return frame, (0, 0)
    x, y, w, h = roi
    crop = frame[y:y+h, x:x+w]
    return crop, (x, y)


def natural_sort_key(path: str):
    # Sort like frame_1, frame_2, ..., frame_10 (not lexicographic)
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_frames(folder: str, pattern: str) -> List[str]:
    files = glob.glob(os.path.join(folder, pattern))
    files.sort(key=natural_sort_key)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="Folder containing extracted frames")
    ap.add_argument("--pattern", default="*.jpg", help="Glob pattern for frames (e.g. frame_*.jpg)")
    ap.add_argument("--fps", type=float, default=0.0, help="Frame rate used to extract frames (needed for time)")
    ap.add_argument("--ball_diameter_mm", type=float, default=47.0, help="Ball diameter in mm")
    ap.add_argument("--csv_out", default="ball_centers_scaled.csv", help="Output CSV path")
    ap.add_argument("--annotated_dir", default="", help="Optional output folder for annotated frames")
    ap.add_argument("--roi", default="", help='ROI as "x,y,w,h" in pixels (optional)')
    ap.add_argument("--select_roi", action="store_true", help="Interactively select ROI on the first frame")
    ap.add_argument("--use_first_as_origin", action="store_true", help="Use first successful detection as origin")
    ap.add_argument("--origin", default="", help='Origin in px as "x0,y0" (optional, overrides first-as-origin)')
    ap.add_argument("--show", action="store_true", help="Preview while processing (press q to quit)")
    args = ap.parse_args()

    frames = list_frames(args.frames_dir, args.pattern)
    if not frames:
        raise FileNotFoundError(f"No frames found in {args.frames_dir} with pattern {args.pattern}")

    roi = parse_roi(args.roi)

    # ROI selection on first frame
    first = cv2.imread(frames[0])
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")

    if args.select_roi:
        r = cv2.selectROI("Select ROI (press ENTER)", first, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI (press ENTER)")
        x, y, w, h = [int(v) for v in r]
        if w > 0 and h > 0:
            roi = (x, y, w, h)

    # Origin
    origin_px = None
    if args.origin.strip():
        ox, oy = [float(v) for v in args.origin.split(",")]
        origin_px = (ox, oy)

    # Scaling via known diameter
    R_mm = args.ball_diameter_mm / 2.0

    # Annotated output folder
    if args.annotated_dir:
        os.makedirs(args.annotated_dir, exist_ok=True)

    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        csvw = csv.writer(f)
        csvw.writerow([
            "index", "time_s",
            "x_px", "y_px", "radius_px", "found",
            "mm_per_px",
            "x_mm", "y_mm",
            "frame_file"
        ])

        last_x = last_y = last_r = 0.0

        for i, path in enumerate(frames):
            frame = cv2.imread(path)
            if frame is None:
                print(f"WARNING: could not read {path}, skipping")
                continue

            crop, (offx, offy) = apply_roi(frame, roi)
            det = detect_ball_center(crop)

            if det.ok:
                x = det.x + offx
                y = det.y + offy
                r = det.r
                last_x, last_y, last_r = x, y, r
            else:
                x, y, r = last_x, last_y, last_r

            if origin_px is None and args.use_first_as_origin and det.ok and r > 0:
                origin_px = (x, y)

            if r > 0:
                mm_per_px = R_mm / r
            else:
                mm_per_px = ""

            if isinstance(mm_per_px, float) and origin_px is not None:
                x0, y0 = origin_px
                x_mm = (x - x0) * mm_per_px
                y_mm = (y - y0) * mm_per_px
            else:
                x_mm = ""
                y_mm = ""

            if args.fps > 0:
                t = i / args.fps
            else:
                t = ""  # unknown

            csvw.writerow([
                i,
                (f"{t:.6f}" if isinstance(t, float) else ""),
                f"{x:.3f}", f"{y:.3f}", f"{r:.3f}", int(det.ok),
                (f"{mm_per_px:.6f}" if isinstance(mm_per_px, float) else ""),
                (f"{x_mm:.3f}" if isinstance(x_mm, float) else ""),
                (f"{y_mm:.3f}" if isinstance(y_mm, float) else ""),
                os.path.basename(path)
            ])

            if args.annotated_dir:
                out = frame.copy()
                if r > 0:
                    cv2.circle(out, (int(round(x)), int(round(y))), int(round(r)), (0, 255, 0), 2)
                    cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
                if origin_px is not None:
                    x0, y0 = origin_px
                    cv2.circle(out, (int(round(x0)), int(round(y0))), 6, (255, 255, 0), 2)

                cv2.putText(out, f"idx={i} found={det.ok}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

                out_path = os.path.join(args.annotated_dir, os.path.basename(path))
                cv2.imwrite(out_path, out)

            if args.show:
                disp = frame.copy()
                if r > 0:
                    cv2.circle(disp, (int(round(x)), int(round(y))), int(round(r)), (0, 255, 0), 2)
                    cv2.circle(disp, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
                cv2.imshow("tracking (press q to quit)", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if args.show:
        cv2.destroyAllWindows()

    print(f"Done. Wrote CSV: {args.csv_out}")
    if args.annotated_dir:
        print(f"Wrote annotated frames to: {args.annotated_dir}")


if __name__ == "__main__":
    main()