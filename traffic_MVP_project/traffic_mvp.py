import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import yaml
import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO


def side_of_line(p, a, b) -> float:
    # Cross product (2D) to know the side of a line segment
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


@dataclass
class TrackState:
    prev_world: Tuple[float, float]  # world coords (meters) OR pseudo-world (pixels/ppm)
    prev_t: float                    # seconds in VIDEO time
    prev_side: float


class TrafficMVP:
    def __init__(self, cfg: dict, model_name: str = "yolov8n.pt"):
        self.cfg = cfg
        self.model = YOLO(model_name)

        # Counting line (in pixels for crossing test)
        self.line_a = tuple(cfg["count_line"]["p1"])
        self.line_b = tuple(cfg["count_line"]["p2"])

        # Fallback calibration (only if homography disabled)
        self.ppm = float(cfg.get("pixels_per_meter", 8.0))

        # Congestion params
        self.speed_ref = float(cfg.get("speed_ref_kmh", 30.0))
        self.density_ref = float(cfg.get("density_ref_count", 20.0))

        # Classes
        self.keep_classes = set(cfg.get("vehicle_classes", ["car", "truck", "bus", "motorcycle"]))

        # Homography (px -> meters)
        hcfg = cfg.get("homography", {})
        self.use_h = bool(hcfg.get("enabled", False))
        self.H = None
        if self.use_h:
            src = np.array(hcfg["src_px"], dtype=np.float32)  # 4 points in px
            dst = np.array(hcfg["dst_m"], dtype=np.float32)   # 4 points in meters
            self.H, _ = cv2.findHomography(src, dst)

        # Tracking state
        self.track_states: Dict[int, TrackState] = {}
        self.count_total = 0
        self.count_by_class = {k: 0 for k in self.keep_classes}
        self.last_speeds: Dict[int, float] = {}  # smoothed km/h

    def to_world_m(self, cx: float, cy: float) -> Tuple[float, float]:
        """
        Convert pixel center -> (x,y) in meters using homography.
        If homography disabled, use fallback pixels_per_meter.
        """
        if not self.use_h or self.H is None:
            return (cx / max(1e-6, self.ppm), cy / max(1e-6, self.ppm))

        pt = np.array([[[cx, cy]]], dtype=np.float32)  # (1,1,2)
        out = cv2.perspectiveTransform(pt, self.H)[0][0]
        return float(out[0]), float(out[1])

    def estimate_speed_kmh(self, tid: int, world_xy: Tuple[float, float], t_video: float) -> float:
        st = self.track_states.get(tid)
        if st is None:
            return 0.0
        dt = max(1e-3, t_video - st.prev_t)
        dx = world_xy[0] - st.prev_world[0]
        dy = world_xy[1] - st.prev_world[1]
        meters = float(np.hypot(dx, dy))
        mps = meters / dt
        return mps * 3.6

    def update_counting(self, tid: int, center_px: Tuple[float, float], world_xy: Tuple[float, float],
                        cls_name: str, t_video: float):
        # Crossing is done in PIXELS (stable), while speed uses WORLD coords (meters)
        side_now = side_of_line(center_px, self.line_a, self.line_b)

        if tid not in self.track_states:
            self.track_states[tid] = TrackState(prev_world=world_xy, prev_t=t_video, prev_side=side_now)
            return

        st = self.track_states[tid]
        # count when sign changes
        if (side_now > 0) != (st.prev_side > 0):
            self.count_total += 1
            if cls_name in self.count_by_class:
                self.count_by_class[cls_name] += 1

        self.track_states[tid] = TrackState(prev_world=world_xy, prev_t=t_video, prev_side=side_now)

    def congestion_score(self, vehicles_in_frame: int, avg_speed: float) -> float:
        density = min(1.0, vehicles_in_frame / max(1e-6, self.density_ref))
        speed_norm = min(1.0, max(0.0, avg_speed / max(1e-6, self.speed_ref)))
        score = 0.6 * density + 0.4 * (1.0 - speed_norm)
        return float(min(1.0, max(0.0, score)))

    def draw_overlay(self, frame, vehicles_in_frame, avg_speed, score, draw_line: bool):
        if draw_line:
            cv2.line(frame, self.line_a, self.line_b, (0, 255, 255), 2)

        y = 30
        cv2.putText(frame, f"Vehicles in frame: {vehicles_in_frame}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28
        cv2.putText(frame, f"Count total: {self.count_total}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28
        cv2.putText(frame, f"Avg speed (km/h): {avg_speed:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28
        cv2.putText(frame, f"Congestion score: {score:.2f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y += 32
        for k, v in self.count_by_class.items():
            cv2.putText(frame, f"{k}: {v}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y += 24

        if score >= 0.75:
            cv2.putText(frame, "ALERT: CONGESTION", (20, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return frame

    def process(self, source, show=False, out_csv="metrics.csv", conf=0.40, draw_line=False):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source: {source}")

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        rows = []
        last_write_video_t = -999.0  # in video seconds

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t_video = frame_idx / fps_video
            frame_idx += 1

            # YOLO tracking (strict conf to reduce false positives)
            results = self.model.track(frame, persist=True, verbose=False, conf=conf)
            r = results[0]

            vehicles_in_frame = 0
            speed_vals = []

            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                names = self.model.names

                for tid, box, cls_id, c in zip(ids, xyxy, cls_ids, confs):
                    cls_name = str(names.get(int(cls_id), cls_id))
                    if cls_name not in self.keep_classes:
                        continue

                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    vehicles_in_frame += 1

                    world_xy = self.to_world_m(cx, cy)
                    kmh = self.estimate_speed_kmh(tid, world_xy, t_video)

                    # smoothing
                    kmh = 0.7 * self.last_speeds.get(tid, kmh) + 0.3 * kmh
                    self.last_speeds[tid] = kmh
                    if kmh > 1.0:
                        speed_vals.append(kmh)

                    self.update_counting(tid, (cx, cy), world_xy, cls_name, t_video)

                    # draw bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls_name}#{tid} {c:.2f}",
                                (int(x1), max(20, int(y1) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            avg_speed = float(np.mean(speed_vals)) if speed_vals else 0.0
            score = self.congestion_score(vehicles_in_frame, avg_speed)

            # write CSV every ~1 second of VIDEO time
            if (t_video - last_write_video_t) >= 1.0:
                rows.append({
                    "t_video_s": round(t_video, 3),
                    "vehicles_in_frame": vehicles_in_frame,
                    "count_total": self.count_total,
                    "avg_speed_kmh": avg_speed,
                    "congestion_score": score,
                    **{f"count_{k}": v for k, v in self.count_by_class.items()}
                })
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                last_write_video_t = t_video

            if show:
                frame = self.draw_overlay(frame, vehicles_in_frame, avg_speed, score, draw_line=draw_line)
                cv2.imshow("Traffic MVP", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

        cap.release()
        cv2.destroyAllWindows()
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[OK] Metrics saved to {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path video.mp4/.mov (converted recommended) or 0 for webcam")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--csv", default="metrics.csv")
    ap.add_argument("--conf", type=float, default=0.40, help="YOLO confidence threshold (e.g. 0.4 / 0.5)")
    ap.add_argument("--draw-line", action="store_true", help="Draw yellow counting line")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    src = int(args.source) if str(args.source).isdigit() else args.source
    app = TrafficMVP(cfg)
    app.process(src, show=args.show, out_csv=args.csv, conf=args.conf, draw_line=args.draw_line)


if __name__ == "__main__":
    main()
