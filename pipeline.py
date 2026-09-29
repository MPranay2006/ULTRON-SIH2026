"""
SAR Drone - Step 4 pipeline (PC version)
Detection (YOLOv8) + ByteTrack IDs + pixel->GPS geotagging (nadir camera)
+ rule-based victim priority + automatic flood/earthquake scene mode.

Run from C:\\sar_drone with the venv active:
    python scripts\\pipeline.py --selftest
    python scripts\\pipeline.py --source videos\\flood_test.mp4
    python scripts\\pipeline.py --source videos\\quake_test.mp4 --save
    python scripts\\pipeline.py --source 0              (webcam)
"""
import argparse
import csv
import math
import os
import time
from collections import deque

import cv2
import numpy as np

# =====================================================================
# 1. SETTINGS
# =====================================================================
CLASS_NAMES = ["person", "debris", "collapsed_building", "flood_water", "fire"]
PERSON, DEBRIS, COLLAPSED, FLOOD, FIRE = 0, 1, 2, 3, 4

EARTH_R = 6378137.0          # Earth radius in metres (WGS-84)
MIN_FRAMES_CONFIRM = 15      # a track must be seen this many frames (~0.5 s) to count as a victim
STILL_WINDOW_S = 5.0         # look back this many seconds to decide "not moving"
STILL_MAX_MOVE_M = 1.0       # moved less than this (metres) in that window => "not moving"
ISOLATION_M = 10.0           # nobody else within this distance => "alone"

# Colours are BGR (OpenCV order), not RGB
HAZARD_COLORS = {DEBRIS: (42, 42, 165), COLLAPSED: (180, 0, 180),
                 FLOOD: (255, 128, 0), FIRE: (0, 0, 255)}
LEVEL_COLORS = {"HIGH": (0, 0, 255), "MEDIUM": (0, 165, 255), "LOW": (0, 200, 0)}
UNCONFIRMED_COLOR = (160, 160, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# =====================================================================
# 2. TELEMETRY (where is the drone?)
#    Step 7 replaces SimTelemetry with a MAVLink version whose get()
#    returns the SAME dictionary, so nothing else in this file changes.
# =====================================================================
class SimTelemetry:
    """Fake GPS + altitude + heading. speed_mps=0 means hovering in place."""

    def __init__(self, lat, lon, alt_m, heading_deg, speed_mps=0.0):
        self.lat0, self.lon0 = lat, lon
        self.alt, self.heading, self.speed = alt_m, heading_deg, speed_mps
        self.t0 = None

    def get(self, t):
        if self.t0 is None:
            self.t0 = t
        dist = self.speed * (t - self.t0)
        h = math.radians(self.heading)
        lat, lon = offset_latlon(self.lat0, self.lon0, dist * math.cos(h), dist * math.sin(h))
        return {"lat": lat, "lon": lon, "alt": self.alt,
                "heading": self.heading, "simulated": True}


# =====================================================================
# 3. GEOTAGGING MATH (camera pointing straight down)
# =====================================================================
def offset_latlon(lat, lon, north_m, east_m):
    """Move a GPS point by some metres north/east (accurate for short distances)."""
    dlat = north_m / EARTH_R
    dlon = east_m / (EARTH_R * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def focal_px(img_w, hfov_deg):
    """Camera focal length in pixels, from image width and horizontal field of view."""
    return (img_w / 2) / math.tan(math.radians(hfov_deg) / 2)


def pixel_to_ground(u, v, img_w, img_h, alt_m, heading_deg, f_px):
    """Pixel (u, v) -> metres north/east of the point directly below the drone."""
    right = (u - img_w / 2) * alt_m / f_px          # + = right side of image
    forward = -(v - img_h / 2) * alt_m / f_px       # + = top of image = drone's nose
    h = math.radians(heading_deg)                   # heading: 0 = north, 90 = east
    north = forward * math.cos(h) - right * math.sin(h)
    east = forward * math.sin(h) + right * math.cos(h)
    return north, east


def pixel_to_gps(u, v, img_w, img_h, tel, f_px):
    n, e = pixel_to_ground(u, v, img_w, img_h, tel["alt"], tel["heading"], f_px)
    return offset_latlon(tel["lat"], tel["lon"], n, e)


def ground_dist_m(lat1, lon1, lat2, lon2):
    """Distance in metres between two nearby GPS points."""
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return EARTH_R * math.hypot(x, y)


def px_dist_point_box(px, py, box):
    """Pixel distance from a point to the nearest edge of a box (0 if inside)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.hypot(dx, dy)


def box_area(box):
    return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)


# =====================================================================
# 4. SCENE CLASSIFIER (flood mode vs earthquake mode)
# =====================================================================
class SceneClassifier:
    """
    Decides NORMAL / FLOOD / EARTHQUAKE.
      flood score = smooth, water-coloured area (muddy brown or blue) + flood_water box area
      quake score = debris/collapsed box area (+ edge density, but ONLY when rubble is detected,
                    because roofs, tents and trees are full of edges too)
    Starts in NORMAL. A hazard mode is only entered after its score has been
    high AND clearly ahead of the other for confirm_s seconds in a row, so a
    few odd frames can't trigger a false alarm. Drops back to NORMAL only if both
    scores stay low for exit_confirm_s seconds (a flooded area stays flooded).
    """

    def __init__(self, alpha=0.05, margin=0.04, enter=0.12, exit_=0.06,
                 confirm_s=3.0, exit_confirm_s=10.0):
        self.alpha, self.margin, self.enter, self.exit = alpha, margin, enter, exit_
        self.confirm_s, self.exit_confirm_s = confirm_s, exit_confirm_s
        self.flood = self.quake = None
        self.mode = "NORMAL"
        self.candidate, self.candidate_since = None, 0.0

    def update(self, frame, hazards, t):
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        muddy = cv2.inRange(hsv, (5, 30, 25), (30, 255, 220))      # brown flood water (incl. shadow)
        bluish = cv2.inRange(hsv, (85, 25, 40), (130, 255, 230))   # blue/grey water
        texture = cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5))
        smooth = (texture < 15).astype(np.uint8) * 255             # water is smooth
        water = cv2.bitwise_and(cv2.bitwise_or(muddy, bluish), smooth)
        water_frac = np.count_nonzero(water) / water.size
        edge_frac = np.count_nonzero(cv2.Canny(gray, 80, 200)) / gray.size

        area = float(frame.shape[0] * frame.shape[1])
        flood_area = min(sum(box_area(b) for c, b in hazards if c == FLOOD) / area, 1.0)
        rubble_area = min(sum(box_area(b) for c, b in hazards
                              if c in (DEBRIS, COLLAPSED)) / area, 1.0)

        f = min(water_frac + 0.4 * flood_area, 1.0)
        q = 0.7 * rubble_area + (0.3 * min(edge_frac * 4, 1.0) if rubble_area > 0.01 else 0.0)
        if self.flood is None:
            self.flood, self.quake = f, q
        else:
            self.flood += self.alpha * (f - self.flood)
            self.quake += self.alpha * (q - self.quake)

        # which mode is the evidence pointing to right now?
        if self.flood >= self.enter and self.flood > self.quake + self.margin:
            want = "FLOOD"
        elif self.quake >= self.enter and self.quake > self.flood + self.margin:
            want = "EARTHQUAKE"
        elif self.flood < self.exit and self.quake < self.exit:
            want = "NORMAL"
        else:
            want = self.mode                                         # unclear: keep current

        # only switch once the new mode has been wanted continuously for confirm_s
        if want == self.mode:
            self.candidate = None
        elif want != self.candidate:
            self.candidate, self.candidate_since = want, t
        elif t - self.candidate_since >= (self.exit_confirm_s if want == "NORMAL" else self.confirm_s):
            self.mode, self.candidate = want, None
        return self.mode


# =====================================================================
# 5. PRIORITY SCORE (0-100, rule-based)
# =====================================================================
def priority_score(cx, cy, hazards, m_per_px, mode, still, isolated):
    def nearest(classes):
        d = [px_dist_point_box(cx, cy, b) for c, b in hazards if c in classes]
        return min(d) * m_per_px if d else math.inf

    score, why = 0, []
    d_fire, d_water, d_rubble = nearest((FIRE,)), nearest((FLOOD,)), nearest((DEBRIS, COLLAPSED))

    if d_fire <= 5:
        score += 40; why.append("fire<5m")
    elif d_fire <= 15:
        score += 20; why.append("fire<15m")

    water_pts = 35 if d_water == 0 else (15 if d_water <= 10 else 0)
    if water_pts:
        why.append("in water" if d_water == 0 else "water<10m")
    rubble_pts = 30 if d_rubble <= 2 else (15 if d_rubble <= 8 else 0)
    if rubble_pts:
        why.append("on rubble" if d_rubble <= 2 else "rubble<8m")
    if mode == "FLOOD" and water_pts:
        water_pts += 10
    if mode == "EARTHQUAKE" and rubble_pts:
        rubble_pts += 10
    score += water_pts + rubble_pts

    if still:
        score += 25; why.append("not moving")
    if isolated:
        score += 10; why.append("alone")
    return min(score, 100), why


def level_of(score):
    return "HIGH" if score >= 60 else ("MEDIUM" if score >= 30 else "LOW")


def is_still(hist, t_now):
    """True if the victim moved less than STILL_MAX_MOVE_M in the last STILL_WINDOW_S seconds."""
    if not hist or t_now - hist[0][0] < STILL_WINDOW_S:
        return False                      # not watched long enough to know yet
    recent = [(la, lo) for (tt, la, lo) in hist if t_now - tt <= STILL_WINDOW_S]
    la0, lo0 = recent[-1]
    return max(ground_dist_m(la0, lo0, la, lo) for la, lo in recent) < STILL_MAX_MOVE_M


# =====================================================================
# 6. DRAWING HELPERS
# =====================================================================
def put_label(img, text, x, y, color, scale=0.5):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    y = max(int(y), th + 4)
    cv2.rectangle(img, (int(x), y - th - 4), (int(x) + tw + 4, y + base - 2), color, -1)
    cv2.putText(img, text, (int(x) + 2, y - 2), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_panel(img, lines, x, y, from_bottom=False, scale=0.55):
    lh = 22
    w = max(cv2.getTextSize(l, FONT, scale, 2)[0][0] for l in lines) + 20
    h = lh * len(lines) + 12
    if from_bottom:
        y = img.shape[0] - h - y
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for k, line in enumerate(lines):
        cv2.putText(img, line, (x + 10, y + 24 + k * lh), FONT, scale,
                    (255, 255, 255), 2, cv2.LINE_AA)


# =====================================================================
# 7. SELF-TEST (checks the maths without needing a model or video)
# =====================================================================
def _clean(x):
    return 0.0 if abs(x) < 0.005 else x


def selftest():
    W, H, alt, hfov = 1280, 720, 30.0, 66.0
    f = focal_px(W, hfov)
    tel = {"lat": 28.613900, "lon": 77.209000, "alt": alt, "heading": 0.0}
    print(f"Focal length {f:.1f} px | ground footprint {2*alt*math.tan(math.radians(hfov/2)):.2f} m wide at {alt:.0f} m")
    cases = [("image centre, heading 0 ", W / 2, H / 2, 0, 0.0, 0.0),
             ("right edge,   heading 0 ", W, H / 2, 0, 0.0, 19.48),
             ("top edge,     heading 0 ", W / 2, 0, 0, 10.96, 0.0),
             ("right edge,   heading 90", W, H / 2, 90, -19.48, 0.0)]
    ok = True
    for name, u, v, hd, exp_n, exp_e in cases:
        tel["heading"] = hd
        n, e = pixel_to_ground(u, v, W, H, alt, hd, f)
        lat, lon = pixel_to_gps(u, v, W, H, tel, f)
        print(f"{name}: north {_clean(n):+7.2f} m, east {_clean(e):+7.2f} m -> {lat:.6f}, {lon:.6f}")
        ok &= abs(n - exp_n) < 0.02 and abs(e - exp_e) < 0.02

    m_per_px = alt / f
    s1, w1 = priority_score(100, 100, [(DEBRIS, (50, 50, 150, 150))], m_per_px, "EARTHQUAKE", True, True)
    s2, w2 = priority_score(100, 100, [], m_per_px, "EARTHQUAKE", False, False)
    print(f"Still, alone, on rubble (earthquake mode): P{s1} {level_of(s1)} {w1}")
    print(f"Walking, in a group, no hazards:           P{s2} {level_of(s2)} {w2}")
    ok &= (s1 == 75 and s2 == 0)
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED - tell Claude the numbers above")


# =====================================================================
# 7b. DETECTION HELPERS
# =====================================================================
def make_tracker_cfg(path="outputs/bytetrack_sar.yaml"):
    """Copy Ultralytics' ByteTrack settings with lower thresholds, so weaker
    (smaller / farther) people can still start and keep a track."""
    try:
        import yaml
        from ultralytics.utils import ROOT
        cfg = yaml.safe_load(open(ROOT / "cfg" / "trackers" / "bytetrack.yaml"))
        cfg.update(track_high_thresh=0.15, track_low_thresh=0.05,
                   new_track_thresh=0.15, track_buffer=60)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
        return path
    except Exception as e:
        print(f"(Could not write custom tracker settings, using defaults: {e})")
        return "bytetrack.yaml"


def boxes_of(res):
    """Result -> list of (class, (x1,y1,x2,y2), conf, track_id or None)."""
    out = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    ids = (res.boxes.id.cpu().numpy().astype(int)
           if res.boxes.id is not None else [None] * len(cls))
    for b, c, cf, tid in zip(xyxy, cls, conf, ids):
        out.append((int(c), tuple(float(x) for x in b), float(cf),
                    None if tid is None else int(tid)))
    return out


# =====================================================================
# 8. MAIN LOOP
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="SAR drone detection pipeline")
    p.add_argument("--source", default="videos/test.mp4", help="video file path, or 0 for webcam")
    p.add_argument("--model", default="models/sar_proto.pt")
    p.add_argument("--conf", type=float, default=0.10, help="min confidence passed to the tracker")
    p.add_argument("--imgsz", type=int, default=640, help="model input size")
    p.add_argument("--mode", default="auto", choices=["auto", "flood", "earthquake"])
    p.add_argument("--person-model", default="",
                   help="optional 2nd model used ONLY for people, e.g. models/yolov8n.pt")
    p.add_argument("--hazard-conf", type=float, default=0.25,
                   help="min confidence for hazard boxes (debris, flood, fire...)")
    p.add_argument("--alt", type=float, default=30.0, help="simulated altitude above ground (m)")
    p.add_argument("--hfov", type=float, default=66.0, help="camera horizontal field of view (deg)")
    p.add_argument("--lat", type=float, default=28.613900, help="simulated drone latitude")
    p.add_argument("--lon", type=float, default=77.209000, help="simulated drone longitude")
    p.add_argument("--heading", type=float, default=0.0, help="simulated heading (0 = north)")
    p.add_argument("--speed", type=float, default=0.0, help="simulated ground speed (m/s), 0 = hover")
    p.add_argument("--max-width", type=int, default=1280, help="shrink frames wider than this")
    p.add_argument("--save", action="store_true", help="save annotated video to outputs/")
    p.add_argument("--no-show", action="store_true", help="no video window (used on the Pi)")
    p.add_argument("--selftest", action="store_true", help="test the maths and exit")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return

    from ultralytics import YOLO   # imported here so --selftest works without a model

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")
    model = YOLO(args.model)
    person_model = None
    if args.person_model:
        if not os.path.exists(args.person_model):
            raise SystemExit(f"Person model not found: {args.person_model}")
        person_model = YOLO(args.person_model)
        print(f"People detected by: {args.person_model} | hazards by: {args.model}")
    tracker_cfg = make_tracker_cfg()

    source = int(args.source) if args.source.isdigit() else args.source
    is_file = not isinstance(source, int)
    if is_file and not os.path.exists(source):
        raise SystemExit(f"Video not found: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps < 1 or src_fps > 240:
        src_fps = 30.0

    telemetry = SimTelemetry(args.lat, args.lon, args.alt, args.heading, args.speed)
    scene = SceneClassifier()
    victims = {}
    os.makedirs("outputs", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    writer, gps_tag = None, "SIMULATED"
    frame_idx, fps, t_last, t_start = 0, 0.0, time.perf_counter(), time.monotonic()

    print(f"Source: {args.source} @ {src_fps:.1f} fps | model: {args.model} | mode: {args.mode}")
    print("Press Q in the video window to quit, SPACE to pause.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > args.max_width:
            s = args.max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        H, W = frame.shape[:2]
        f_px = focal_px(W, args.hfov)
        t = frame_idx / src_fps if is_file else time.monotonic() - t_start
        frame_idx += 1
        tel = telemetry.get(t)
        gps_tag = "SIMULATED" if tel["simulated"] else "LIVE"
        m_per_px = tel["alt"] / f_px

        # --- 8a. detect + track -------------------------------------------
        if person_model is None:
            # one model finds people AND hazards; the tracker gives IDs
            dets = boxes_of(model.track(frame, persist=True, tracker=tracker_cfg,
                                        conf=args.conf, imgsz=args.imgsz, verbose=False)[0])
        else:
            # people: tracked with the person model (COCO 'person' = class 0)
            # hazards: plain detection with our SAR model (no IDs needed)
            dets = [(PERSON, b, cf, tid) for c, b, cf, tid in boxes_of(
                person_model.track(frame, persist=True, tracker=tracker_cfg, classes=[0],
                                   conf=args.conf, imgsz=args.imgsz, verbose=False)[0])]
            dets += boxes_of(model.predict(frame, conf=args.hazard_conf, imgsz=args.imgsz,
                                           classes=[DEBRIS, COLLAPSED, FLOOD, FIRE],
                                           verbose=False)[0])
        persons = [(tid, b, cf) for c, b, cf, tid in dets if c == PERSON and tid is not None]
        hazards = [(c, b) for c, b, cf, tid in dets if c != PERSON and cf >= args.hazard_conf]

        # --- 8b. scene mode -----------------------------------------------
        mode = scene.update(frame, hazards, t) if args.mode == "auto" else args.mode.upper()

        # --- 8c. geotag every tracked person ------------------------------
        current = {}
        for tid, box, cf in persons:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            lat, lon = pixel_to_gps(cx, cy, W, H, tel, f_px)
            v = victims.setdefault(tid, {"first_t": t, "frames": 0, "best_conf": 0.0,
                                         "hist": deque(maxlen=600), "max_priority": 0})
            v["frames"] += 1
            v["last_t"], v["lat"], v["lon"] = t, lat, lon
            v["best_conf"] = max(v["best_conf"], cf)
            v["hist"].append((t, lat, lon))
            current[tid] = (box, cx, cy, lat, lon)

        # --- 8d. draw hazards ---------------------------------------------
        for c, box in hazards:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), HAZARD_COLORS[c], 2)
            put_label(frame, CLASS_NAMES[c], x1, y1, HAZARD_COLORS[c])

        # --- 8e. priority score + draw each person -------------------------
        for tid, (box, cx, cy, lat, lon) in current.items():
            v = victims[tid]
            still = is_still(v["hist"], t)
            isolated = all(ground_dist_m(lat, lon, o[3], o[4]) > ISOLATION_M
                           for oid, o in current.items() if oid != tid)
            score, why = priority_score(cx, cy, hazards, m_per_px, mode, still, isolated)
            lvl = level_of(score)
            v.update(priority=score, level=lvl, reasons=why)
            v["max_priority"] = max(v["max_priority"], score)
            x1, y1, x2, y2 = map(int, box)
            if v["frames"] >= MIN_FRAMES_CONFIRM:
                cv2.rectangle(frame, (x1, y1), (x2, y2), LEVEL_COLORS[lvl], 2)
                put_label(frame, f"ID{tid} P{score} {lvl}", x1, y1, LEVEL_COLORS[lvl])
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), UNCONFIRMED_COLOR, 1)

        # --- 8f. on-screen panels -----------------------------------------
        now = time.perf_counter()
        inst = 1.0 / max(now - t_last, 1e-6)
        t_last = now
        fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
        n_conf = sum(1 for v in victims.values() if v["frames"] >= MIN_FRAMES_CONFIRM)
        n_view = sum(1 for k in current if victims[k]["frames"] >= MIN_FRAMES_CONFIRM)
        mode_txt = "NORMAL - no hazard detected" if mode == "NORMAL" else mode
        hud = [f"MODE: {mode_txt} ({'auto' if args.mode == 'auto' else 'manual'})",
               f"GPS: {gps_tag}  {tel['lat']:.6f}, {tel['lon']:.6f}",
               f"ALT {tel['alt']:.0f} m   HDG {tel['heading']:.0f} deg",
               f"FPS {fps:.1f}   Victims in view: {n_view}"]
        if args.mode == "auto" and scene.flood is not None:
            hud.append(f"scene: flood={scene.flood:.2f}  quake={scene.quake:.2f}")
        draw_panel(frame, hud, 10, 10)

        in_view = sorted((tid for tid in current if victims[tid]["frames"] >= MIN_FRAMES_CONFIRM),
                         key=lambda k: -victims[k]["priority"])[:3]
        if in_view:
            lines = ["TOP PRIORITY IN VIEW"] + [
                f"ID{k} P{victims[k]['priority']} {victims[k]['level']}  "
                f"{victims[k]['lat']:.6f}, {victims[k]['lon']:.6f}" for k in in_view]
            draw_panel(frame, lines, 10, 10, from_bottom=True)

        # --- 8g. save / show ----------------------------------------------
        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(f"outputs/demo_{stamp}.mp4",
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps if is_file else 10, (W, H))
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("SAR pipeline (Q = quit, SPACE = pause)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)
        if frame_idx % 100 == 0:
            print(f"frame {frame_idx}: {fps:.1f} FPS | mode {mode} | confirmed victims {n_conf}")

    # --- 9. clean up + write the victim report ----------------------------
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if frame_idx == 0:
        raise SystemExit("No frames were read - the video file may be corrupt or an unsupported codec.")

    rows = sorted(((tid, v) for tid, v in victims.items() if v["frames"] >= MIN_FRAMES_CONFIRM),
                  key=lambda kv: -kv[1]["max_priority"])
    csv_path = f"outputs/victims_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "latitude", "longitude", "max_priority", "last_priority",
                    "last_level", "last_reasons", "frames_seen", "first_seen_s",
                    "last_seen_s", "best_conf", "gps_source"])
        for tid, v in rows:
            w.writerow([tid, f"{v['lat']:.7f}", f"{v['lon']:.7f}", v["max_priority"],
                        v.get("priority", 0), v.get("level", "LOW"),
                        "; ".join(v.get("reasons", [])), v["frames"],
                        f"{v['first_t']:.1f}", f"{v['last_t']:.1f}",
                        f"{v['best_conf']:.2f}", gps_tag.lower()])

    print(f"\nDone: {frame_idx} frames processed, ~{fps:.1f} FPS at the end")
    print(f"Confirmed victims: {len(rows)} -> report saved to {csv_path}")
    if args.save:
        print(f"Annotated video saved to outputs/demo_{stamp}.mp4")


if __name__ == "__main__":
    main()
