"""
  Footfall counter using YOLO (Ultralytics) for person detection and DeepSORT for tracking.
  Counts entries/exits across a configurable ROI line with robust logic against occlusions.

Features:
  - Person detection (YOLOv8)
  - Tracking with DeepSORT (deep-sort-realtime)
  - ROI line crossing counts (IN / OUT)
  - Trajectory trails (bonus)
  - Heatmap overlay (bonus)
  - FastAPI mini-API to process an uploaded video
"""
import os
import cv2
import sys
import math
import time
import json
import argparse
import tempfile
import numpy as np
from collections import defaultdict, deque
from typing import Tuple, Dict, List, Optional

# Detection: Ultralytics YOLO
from ultralytics import YOLO

# Tracking: Deep SORT (simple API)
from deep_sort_realtime.deepsort_tracker import DeepSort

# Optional: download YouTube videos for convenience
try:
    import yt_dlp
    YT_AVAILABLE = True
except Exception:
    YT_AVAILABLE = False

# Utility Geometry Functions

def segment_intersection(p1, p2, q1, q2) -> bool:
    """Return True if line segments p1-p2 and q1-q2 intersect (inclusive)."""
    def orient(a,b,c):
        return np.sign((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
    def on_seg(a,b,c):
        return min(a[0],b[0]) <= c[0] <= max(a[0],b[0]) and min(a[1],b[1]) <= c[1] <= max(a[1],b[1])
    o1 = orient(p1,p2,q1); o2 = orient(p1,p2,q2)
    o3 = orient(q1,q2,p1); o4 = orient(q1,q2,p2)

    if o1 != o2 and o3 != o4:
        return True
    # collinear special cases
    if o1 == 0 and on_seg(p1,p2,q1): return True
    if o2 == 0 and on_seg(p1,p2,q2): return True
    if o3 == 0 and on_seg(q1,q2,p1): return True
    if o4 == 0 and on_seg(q1,q2,p2): return True
    return False

def point_side_of_line(a: Tuple[int,int], b: Tuple[int,int], p: Tuple[int,int]) -> float:
    """Cross-product sign for point P relative to directed line A->B.
       >0 => left side, <0 => right side, 0 => on the line."""
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

def center_of_bbox(x1,y1,x2,y2):
    return int((x1+x2)/2), int((y1+y2)/2)

def download_youtube(url: str) -> str:
    """Download YouTube video to a temp mp4; return local path."""
    if not YT_AVAILABLE:
        raise RuntimeError("yt-dlp not installed. Install with: pip install yt-dlp")
    tmpdir = tempfile.mkdtemp(prefix="yt_")
    outtmpl = os.path.join(tmpdir, "video.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "mp4/bestaudio/best",
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        outpath = ydl.prepare_filename(info)
        if not outpath.endswith(".mp4"):
            # try to convert extension to mp4 naming
            base, _ = os.path.splitext(outpath)
            mp4_guess = base + ".mp4"
            if os.path.exists(mp4_guess):
                outpath = mp4_guess
        return outpath

# Core Footfall Counter

class FootfallCounter:
    def __init__(
        self,
        roi_line: Tuple[Tuple[int,int], Tuple[int,int]],
        heatmap: bool = False,
        trail_len: int = 30,
        cross_cooldown: int = 25,   # frames to ignore repeat crossings
        deepsort_kwargs: Optional[dict] = None
    ):
        self.roi_a, self.roi_b = roi_line
        self.heatmap_enabled = heatmap
        self.trail_len = trail_len
        self.cross_cooldown = cross_cooldown

        self.in_count = 0
        self.out_count = 0

        # Per-track state
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))
        self.track_last_side: Dict[int, float] = dict()
        self.track_cooldown: Dict[int, int] = defaultdict(int)

        # Heatmap (initialized later after reading first frame's size)
        self.heatmap_accum = None

        # DeepSORT
        if deepsort_kwargs is None:
            deepsort_kwargs = {
                "max_age": 20,
                "n_init": 3,
                "nn_budget": 100,
                "max_iou_distance": 0.7
            }
        self.tracker = DeepSort(**deepsort_kwargs)

    def _update_heatmap(self, h, w, points: List[Tuple[int,int]]):
        if not self.heatmap_enabled:
            return
        if self.heatmap_accum is None:
            self.heatmap_accum = np.zeros((h, w), dtype=np.float32)
        for (cx, cy) in points:
            if 0 <= cy < h and 0 <= cx < w:
                self.heatmap_accum[cy, cx] += 1.0

    def _draw_heatmap_overlay(self, frame):
        if self.heatmap_accum is None:
            return frame
        hm = self.heatmap_accum.copy()
        hm = cv2.GaussianBlur(hm, (0,0), 5)
        if hm.max() > 0:
            hm_norm = (hm / hm.max() * 255).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.7, hm_color, 0.3, 0)
            return overlay
        return frame

    def _check_crossing(self, tid: int) -> Optional[str]:
        """
        Returns 'IN' or 'OUT' if a new crossing is registered for track tid, else None.
        Logic: compare last two centroid positions relative to ROI; if sign flips and segment intersects ROI line,
        classify direction using motion projected on the ROI normal or line direction.
        """
        hist = self.track_history[tid]
        if len(hist) < 2:
            return None
        p_prev = hist[-2]
        p_curr = hist[-1]

        last_side = self.track_last_side.get(tid, None)
        curr_side = np.sign(point_side_of_line(self.roi_a, self.roi_b, p_curr))
        prev_side = np.sign(point_side_of_line(self.roi_a, self.roi_b, p_prev))

        # Save current side for next time
        self.track_last_side[tid] = curr_side

        # Hidden in cooldown? avoid double count
        if self.track_cooldown[tid] > 0:
            self.track_cooldown[tid] -= 1
            return None

        # If side sign flips and the segment intersects ROI
        if curr_side != 0 and prev_side != 0 and curr_side != prev_side:
            if segment_intersection(self.roi_a, self.roi_b, p_prev, p_curr):
                # Determine direction: project motion vector onto line normal
                # A->B normal vector:
                ax, ay = self.roi_a; bx, by = self.roi_b
                line_vec = np.array([bx-ax, by-ay], dtype=np.float32)
                motion = np.array([p_curr[0]-p_prev[0], p_curr[1]-p_prev[1]], dtype=np.float32)

                # Use a perpendicular vector `n = [-ly, lx]` to decide "from which side to which"
                n = np.array([ -line_vec[1], line_vec[0] ], dtype=np.float32)
                dot = float(np.dot(motion, n))

                # Positive dot means motion points toward "left->right" across normal
                # We'll define "IN" as crossing from right side to left side (you can swap to fit your doorway direction).
                direction = "IN" if dot > 0 else "OUT"
                self.track_cooldown[tid] = self.cross_cooldown
                return direction
        return None

    def process_frame(
        self,
        frame: np.ndarray,
        detections_xyxy_conf_cls: List[Tuple[List[float], float, int]],
        draw: bool = True
    ):
        """
        Run DeepSORT on current frame with person detections and update counts.
        detections_xx: list of ( [x1,y1,x2,y2], conf, class_id ) filtered to person.
        """
        h, w = frame.shape[:2]

        # Convert detections to deep-sort format: [ [x1,y1,x2,y2], conf, class_name ]
        ds_inputs = []
        for (xyxy, conf, cls_id) in detections_xyxy_conf_cls:
            cls_name = "person"  # filtered already
            ds_inputs.append([xyxy, conf, cls_name])

        tracks = self.tracker.update_tracks(ds_inputs, frame=frame)
        new_centroids = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom
            x1, y1, x2, y2 = map(int, ltrb)
            cx, cy = center_of_bbox(x1,y1,x2,y2)
            new_centroids.append((cx, cy))
            self.track_history[tid].append((cx, cy))

            # Counting
            crossed = self._check_crossing(tid)
            if crossed == "IN":
                self.in_count += 1
            elif crossed == "OUT":
                self.out_count += 1

            if draw:
                # draw bbox
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                # draw trail
                hist = self.track_history[tid]
                for i in range(1, len(hist)):
                    cv2.line(frame, hist[i-1], hist[i], (255,255,255), 2)

        # Heatmap accumulation
        self._update_heatmap(h, w, new_centroids)

        # Draw ROI line and counts
        if draw:
            cv2.line(frame, self.roi_a, self.roi_b, (0,0,255), 3)
            cv2.putText(frame, f"IN: {self.in_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2)
            cv2.putText(frame, f"OUT: {self.out_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255), 2)

        # Heatmap overlay
        if self.heatmap_enabled:
            frame = self._draw_heatmap_overlay(frame)

        return frame

# Pipeline Runner

def run_pipeline(
    source: str,
    out_path: str,
    roi_line: Optional[Tuple[Tuple[int,int], Tuple[int,int]]] = None,
    conf_thres: float = 0.4,
    iou_thres: float = 0.5,
    show: bool = False,
    heatmap: bool = False,
    max_frames: Optional[int] = None
):
    # Input (handle YouTube)
    input_path = source
    if source.startswith("http"):
        input_path = download_youtube(source)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # Resolve ROI default (horizontal midline)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame.")
    H, W = first_frame.shape[:2]
    if roi_line is None:
        roi_line = ((int(0.1*W), H//2), (int(0.9*W), H//2))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # Models
    model = YOLO("yolov8n.pt")  # auto-downloads
    counter = FootfallCounter(roi_line=roi_line, heatmap=heatmap)

    # Write first frame back to stream? We'll re-process it for consistency
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    from tqdm import tqdm
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    pbar_total = max_frames if max_frames else total_frames
    pbar = tqdm(total=pbar_total, desc="Processing", unit="frame")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        # Detect persons
        # Model returns list of Results; we take first item
        results = model.predict(frame, conf=conf_thres, iou=iou_thres, classes=[0], verbose=False)
        dets = []
        if len(results):
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for b, c, k in zip(boxes, confs, clss):
                    x1, y1, x2, y2 = b.tolist()
                    dets.append(([x1, y1, x2, y2], float(c), int(k)))

        # Process frame through tracker + counting
        vis = counter.process_frame(frame, dets, draw=True)

        writer.write(vis)
        if show:
            cv2.imshow("Footfall Counter", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        pbar.update(1)
    pbar.close()
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    print(json.dumps({
        "in_count": counter.in_count,
        "out_count": counter.out_count,
        "output_video": os.path.abspath(out_path)
    }, indent=2))

# FastAPI Server

def start_api():
    """
    Minimal FastAPI server to accept a video and return processed counts + a video file path.
    Start: python footfall_counter.py --api
    Then POST via:
      curl -F "file=@/path/video.mp4" -F "roi=100,400,1100,400" http://127.0.0.1:8000/process
    """
    from fastapi import FastAPI, UploadFile, File, Form
    import uvicorn

    app = FastAPI(title="Footfall Counter API")

    @app.post("/process")
    async def process(
        file: UploadFile = File(...),
        roi: str = Form(default=""),
        conf: float = Form(default=0.4),
        iou: float = Form(default=0.5),
        heatmap: bool = Form(default=False)
    ):
        # save upload
        tmp = tempfile.mkdtemp(prefix="api_")
        in_path = os.path.join(tmp, file.filename)
        with open(in_path, "wb") as f:
            f.write(await file.read())

        # parse ROI
        roi_line = None
        if roi:
            try:
                x1,y1,x2,y2 = map(int, roi.split(","))
                roi_line = ((x1,y1),(x2,y2))
            except:
                pass

        out_path = os.path.join(tmp, "processed.mp4")
        run_pipeline(
            source=in_path,
            out_path=out_path,
            roi_line=roi_line,
            conf_thres=conf,
            iou_thres=iou,
            show=False,
            heatmap=heatmap
        )
        return {
            "status": "ok",
            "output_video": out_path
        }

    uvicorn.run(app, host="127.0.0.1", port=8000)

# -------- CLI -------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Footfall Counter with YOLO + DeepSORT")
    ap.add_argument("--source", type=str, help="Video path or YouTube URL", default="")
    ap.add_argument("--out", type=str, default="processed.mp4", help="Output video path")
    ap.add_argument("--roi", type=str, default="", help="ROI line as x1,y1,x2,y2")
    ap.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold (NMS)")
    ap.add_argument("--show", action="store_true", help="Show live window")
    ap.add_argument("--heatmap", action="store_true", help="Enable heatmap overlay (bonus)")
    ap.add_argument("--max-frames", type=int, default=0, help="Process at most N frames (0 = all)")
    ap.add_argument("--api", action="store_true", help="Start FastAPI server instead of batch processing")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.api:
        start_api()
        sys.exit(0)

    if not args.source:
        print("Please provide --source <video_or_url>", file=sys.stderr)
        sys.exit(1)

    roi_line = None
    if args.roi:
        try:
            x1,y1,x2,y2 = map(int, args.roi.split(","))
            roi_line = ((x1,y1), (x2,y2))
        except Exception:
            print("Invalid --roi format. Expected x1,y1,x2,y2", file=sys.stderr)

    run_pipeline(
        source=args.source,
        out_path=args.out,
        roi_line=roi_line,
        conf_thres=args.conf,
        iou_thres=args.iou,
        show=args.show,
        heatmap=args.heatmap,
        max_frames=(args.max_frames if args.max_frames > 0 else None)
    )
