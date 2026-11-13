"""Video -> station-level metrics.

Two piece definitions (select with `piece_mode` in the zones YAML or `--piece-mode`):

  transit    (default) A "piece" is a unit that ENTERS a station zone and later LEAVES it.
             Presence = dark-engine pixel fraction in the zone (hysteresis + debounce). A transition only counts
             if there is motion inside the zone around it, which rejects slow lighting/focus drift.
             Units already in a zone at t=0 (or still there at the end) are censored: reported as events,
             but they do not produce a completed cycle time. No detector is needed in this mode.

  occupancy  (legacy heuristic) YOLO+ByteTrack confirms a worker is at the zone; a motion-hysteresis state
             machine turns bursts of activity into work cycles. One burst = one piece.

Privacy: both modes measure STATIONS. No face detection, no person IDs; people are never boxed, drawn or stored.
All video/image outputs have every detected person Gaussian-blurred (config: blur_people, default on).
(Occupancy mode uses tracker IDs only inside the frame loop to debounce, then discards them.)
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import events as E

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "out"

DEFAULT_PRESENCE = {"dark_v": 70, "enter": 0.22, "leave": 0.18, "min_hold_s": 0.4,
                    "motion_confirm": 0.04, "confirm_window_s": 1.5, "motion_px_thresh": 14}


@dataclass
class Station:
    id: str
    name: str
    polygon: np.ndarray
    mask: np.ndarray | None = None
    area: int = 1
    # ---- shared
    state: str = "idle"
    state_since: float = 0.0
    cycles: list[tuple[float, float]] = field(default_factory=list)      # completed pieces
    intervals: list[tuple[float, float, str]] = field(default_factory=list)
    # ---- occupancy mode
    energy: float = 0.0
    in_cycle: bool = False
    cycle_start: float = 0.0
    last_hot: float = 0.0
    # ---- transit mode
    occupied: bool = False
    cand: bool = False
    cand_since: float = 0.0
    motion_hist: deque = field(default_factory=lambda: deque(maxlen=400))
    pending: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)                     # confirmed enter/leave
    open_enter: float | None = None
    present_at_start: bool = False
    last_dwell: float = 0.0

    def avg_cycle(self) -> float:
        return statistics.fmean(e - s for s, e in self.cycles) if self.cycles else 0.0

    confirm_thr: float = 0.04

    def _count(self, kind: str) -> int:
        """Confirmed events plus pending ones that already have enough motion evidence (keeps the live overlay in sync)."""
        return (sum(e["type"] == kind for e in self.events)
                + sum(e["type"] == kind and e["max_motion"] >= self.confirm_thr for e in self.pending))

    @property
    def n_in(self) -> int:
        return self._count("enter")

    @property
    def n_out(self) -> int:
        return self._count("leave")


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_stations(cfg: dict, w: int, h: int) -> list[Station]:
    out = []
    pad = int(cfg.get("zone_pad_px", 0))   # grow each zone outward so slow camera creep can't clip a unit at the edge
    for s in cfg["stations"]:
        poly = np.array(s["polygon"], dtype=np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        if pad:
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pad + 1, 2 * pad + 1)))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 2, True).reshape(-1, 2).astype(np.int32)
        out.append(Station(s["id"], s["name"], poly, mask, max(1, int((mask > 0).sum()))))
    return out


def _set_state(st: Station, state: str, t: float) -> None:
    if state != st.state:
        if t > st.state_since:
            st.intervals.append((st.state_since, t, st.state))
        st.state, st.state_since = state, t


# ----------------------------------------------------------------------------- occupancy (legacy)
def update_occupancy(st: Station, occupied: bool, energy: float, t: float, m: dict) -> None:
    st.energy = 0.6 * st.energy + 0.4 * energy
    hot = occupied and st.energy >= m["high"]
    warm = occupied and st.energy >= m["low"]
    if hot:
        st.last_hot = t
        if not st.in_cycle:
            st.in_cycle, st.cycle_start = True, t
            _set_state(st, "active", t)
    elif st.in_cycle and (not warm) and t - st.last_hot >= m["lull_s"]:
        end = st.last_hot
        if end - st.cycle_start >= m["min_cycle_s"]:
            st.cycles.append((st.cycle_start, end))
        st.in_cycle = False
        _set_state(st, "idle", end)


# ----------------------------------------------------------------------------- transit (default)
def update_transit(st: Station, presence: float, motion: float, t: float, p: dict) -> None:
    st.motion_hist.append((t, motion))
    st.confirm_thr = p["motion_confirm"]
    if t == 0.0 or (not st.motion_hist or len(st.motion_hist) == 1):
        st.present_at_start = presence >= p["enter"]
        st.occupied = st.cand = st.present_at_start
        st.cand_since = t
        if st.occupied:
            _set_state(st, "active", t)
        return
    # hysteresis on the raw presence signal
    want = presence >= p["enter"] if not st.cand else presence > p["leave"]
    if want != st.cand:
        st.cand, st.cand_since = want, t
    # debounce: the candidate must persist min_hold_s before the state flips
    if st.cand != st.occupied and t - st.cand_since >= p["min_hold_s"]:
        st.occupied = st.cand
        ev_t = st.cand_since
        lo = ev_t - p["confirm_window_s"]
        st.pending.append({"type": "enter" if st.occupied else "leave", "t": ev_t,
                           "max_motion": max((e for tt, e in st.motion_hist if tt >= lo), default=0.0)})
        _set_state(st, "active" if st.occupied else "idle", ev_t)
    # extend motion evidence for pending events, finalise those past their confirm window
    for ev in list(st.pending):
        if t <= ev["t"] + p["confirm_window_s"]:
            ev["max_motion"] = max(ev["max_motion"], motion)
        else:
            _finalise(st, ev, p)


def _finalise(st: Station, ev: dict, p: dict) -> None:
    if ev in st.pending:
        st.pending.remove(ev)
    ev["confirmed"] = ev["max_motion"] >= p["motion_confirm"]
    if not ev["confirmed"]:
        return
    st.events.append({"type": ev["type"], "t": round(ev["t"], 2), "motion": round(ev["max_motion"], 3)})
    if ev["type"] == "enter":
        st.open_enter = ev["t"]
    elif st.open_enter is not None:                       # enter AND leave both observed -> a completed piece
        st.cycles.append((st.open_enter, ev["t"]))
        st.last_dwell = ev["t"] - st.open_enter
        st.open_enter = None


# ----------------------------------------------------------------------------- drawing / output
def draw(frame: np.ndarray, stations: list[Station], t: float, mode: str) -> np.ndarray:
    sc = max(1.0, frame.shape[1] / 1280)
    worst = max((s for s in stations if s.cycles), key=lambda s: s.avg_cycle(), default=None)
    overlay = frame.copy()

    def col_of(s: Station):
        if s is worst:
            return (0, 0, 255)
        return (0, 200, 0) if s.state == "active" else (200, 200, 200)

    for s in stations:
        cv2.fillPoly(overlay, [s.polygon], col_of(s))
    frame = cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)
    for s in stations:
        c = col_of(s)
        cv2.polylines(frame, [s.polygon], True, c, int((4 if s is worst else 3) * sc))
        tag = (int(s.polygon[:, 0].min()) + 6, int(s.polygon[:, 1].max()) - int(8 * sc))   # short id tag on the zone
        cv2.putText(frame, s.id, tag, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sc, (0, 0, 0), int(5 * sc), cv2.LINE_AA)
        cv2.putText(frame, s.id, tag, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sc, c, int(2 * sc), cv2.LINE_AA)
    # legend panel (top-left): one row per station, so labels never collide when zones are close together
    rows = []
    for s in stations:
        if mode == "transit":
            rows.append((f"{s.id} {s.name}: in {s.n_in} out {s.n_out} done {len(s.cycles)}  "
                         + ("UNIT" if s.occupied else "empty"), col_of(s)))
        else:
            rows.append((f"{s.id} {s.name}: pieces {len(s.cycles)} avg {s.avg_cycle():.1f}s  {s.state.upper()}"
                         + ("  << BOTTLENECK" if s is worst else ""), col_of(s)))
    pw, ph, lh = int(760 * sc), int(24 * sc) + int(34 * sc) * (len(rows) + 1), int(34 * sc)
    box = frame.copy()
    cv2.rectangle(box, (0, 0), (pw, ph), (0, 0, 0), -1)
    frame = cv2.addWeighted(box, 0.55, frame, 0.45, 0)
    for i, (txt, c) in enumerate(rows):
        cv2.putText(frame, txt, (12, int(32 * sc) + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.62 * sc,
                    (255, 255, 255), int(2 * sc), cv2.LINE_AA)
        cv2.circle(frame, (pw - int(24 * sc), int(25 * sc) + lh * (i + 1)), int(8 * sc), c, -1)
    cv2.putText(frame, f"t={t:5.1f}s", (10, int(32 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sc, (0, 0, 0), int(5 * sc), cv2.LINE_AA)
    cv2.putText(frame, f"t={t:5.1f}s", (10, int(32 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sc, (255, 255, 255), int(2 * sc), cv2.LINE_AA)
    return frame


def blur_people(frame: np.ndarray, boxes) -> np.ndarray:
    """Strong Gaussian blur over every detected person box. No boxes are drawn, nothing is stored or identified."""
    h, w = frame.shape[:2]
    k = (max(31, (w // 12) | 1), max(31, (w // 12) | 1))
    for x1, y1, x2, y2 in boxes:
        pw, ph = (x2 - x1) * 0.12, (y2 - y1) * 0.08          # slight margin so edges of the head are covered
        x1, y1, x2, y2 = int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(w, x2 + pw)), int(min(h, y2 + ph))
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], k, 0)
    return frame


def person_boxes(model, frame: np.ndarray) -> list:
    """Person detections used ONLY to blur (low threshold, high res so small background workers are caught)."""
    r = model.predict(frame, classes=[0], conf=0.12, imgsz=1280, verbose=False)[0]
    return [] if r.boxes is None else r.boxes.xyxy.cpu().numpy().astype(int).tolist()


def _to_h264(path: Path) -> None:
    """OpenCV writes mp4v, which browsers can't play; re-encode to H.264 so Streamlit can show it."""
    try:
        import imageio_ffmpeg
        tmp = path.with_suffix(".h264.mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart", str(tmp)],
                       check=True)
        tmp.replace(path)
    except Exception as e:
        print("H.264 transcode skipped:", e)


def snapshot(stations: list[Station], t: float) -> dict:
    return {"t": round(t, 1), "stations": {s.id: {
        "name": s.name, "pieces": len(s.cycles), "avg_cycle_s": round(s.avg_cycle(), 1), "state": s.state,
        "entered": s.n_in, "left": s.n_out, "occupied": s.occupied} for s in stations}}


# ----------------------------------------------------------------------------- shared-schema output
def write_events(stations: list[Station], mode: str, db_path: Path | None, clip_s: float, video_name: str,
                 line: str = "CAM1", shift: str = "CLIP") -> int:
    """Write confirmed station events into the shared `events` table with source='vision' (seconds from clip start).

    Unit identity is NOT tracked between stations, so unit_id is NULL. A `leave` with no earlier `enter` means the unit
    was already inside when the clip began; an `enter` with no `leave` means it was still inside at the end. Those
    unmatched events are written as-is and never turned into a cycle time.
    """
    rows = []
    for s in stations:
        k = 0
        if mode == "transit":
            open_vid = None
            for ev in sorted(s.events, key=lambda e: e["t"]):
                if ev["type"] == "enter":
                    k += 1
                    open_vid = f"vision:{s.id}:{k}"
                    rows.append(("vision", None, open_vid, line, s.id, "enter", ev["t"], "s", shift, None))
                else:
                    if open_vid is None:
                        k += 1
                        open_vid = f"vision:{s.id}:{k}"
                    rows.append(("vision", None, open_vid, line, s.id, "leave", ev["t"], "s", shift, None))
                    open_vid = None
        else:                                              # legacy occupancy mode: each work cycle = one complete visit
            for a, b in s.cycles:
                k += 1
                vid = f"vision:{s.id}:{k}"
                rows.append(("vision", None, vid, line, s.id, "enter", round(a, 2), "s", shift, None))
                rows.append(("vision", None, vid, line, s.id, "leave", round(b, 2), "s", shift, None))
    con = E.connect(db_path)
    E.clear_source(con, "vision")
    E.insert_events(con, rows)
    for s in stations:
        E.set_label(con, "vision", line, s.id, s.name)
    E.set_meta(con, "vision", ts_unit="s", ingest="vision.pipeline", piece_mode=mode, clip=video_name,
               clip_seconds=round(clip_s, 1), note="seconds from clip start; unit identity not tracked")
    con.commit()
    E.refresh_catalog(con)
    con.close()
    return len(rows)


# ----------------------------------------------------------------------------- main loop
def run(video: Path, zones: Path, db_path: Path | None = None, stride: int = 1, limit_s: float | None = None,
        preview: bool = False, piece_mode: str | None = None) -> dict:
    cfg = load_config(zones)
    mode = piece_mode or cfg.get("piece_mode", "transit")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stations = build_stations(cfg, w, h)
    OUT.mkdir(parents=True, exist_ok=True)
    do_blur = bool(cfg.get("blur_people", True))   # people are blurred in ALL video/image outputs
    model = None
    if do_blur or mode == "occupancy":
        from ultralytics import YOLO
        model = YOLO(cfg.get("model", "yolov8n.pt"))
    if preview:
        ok, f0 = cap.read()
        if do_blur:
            f0 = blur_people(f0, person_boxes(model, f0))
        cv2.imwrite(str(OUT / "zones_preview.jpg"), draw(f0, stations, 0.0, mode))
        print("wrote", OUT / "zones_preview.jpg")
        return {}

    pres = {**DEFAULT_PRESENCE, **cfg.get("presence", {})}
    m = cfg.get("motion", {})
    writer = cv2.VideoWriter(str(OUT / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
    timeline, prev_gray, i, next_snap = [], None, 0, 0.0
    first = True
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % stride:
            continue
        t = (i - 1) / fps
        if limit_s and t > limit_s:
            break
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        if mode == "transit":
            val = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
            dark = val < pres["dark_v"]
            for s in stations:
                inside = s.mask > 0
                presence = float((dark & inside).sum()) / s.area
                en = 0.0 if prev_gray is None else \
                    float(((cv2.absdiff(gray, prev_gray) > pres["motion_px_thresh"]) & inside).sum()) / s.area
                if first:
                    s.motion_hist.clear()
                update_transit(s, presence, en, t, pres)
        else:
            res = model.track(frame, persist=True, tracker=cfg.get("tracker", "bytetrack.yaml"),
                              classes=[0], verbose=False)[0]
            boxes = []
            if res.boxes is not None and res.boxes.id is not None:  # confirmed tracks only; IDs dropped here
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            for s in stations:
                occ = any(s.mask[max(y1, 0):y2, max(x1, 0):x2].sum() / 255 > 0.05 * s.area for x1, y1, x2, y2 in boxes)
                en = 0.0 if prev_gray is None else float(((cv2.absdiff(gray, prev_gray) > 18) & (s.mask > 0)).sum()) / s.area
                update_occupancy(s, occ, en, t, m)
        first = False
        prev_gray = gray
        out_frame = frame.copy()      # analysis above used the raw frame; only the OUTPUT is blurred
        if do_blur:
            out_frame = blur_people(out_frame, person_boxes(model, frame))
        writer.write(draw(out_frame, stations, t, mode))
        if t >= next_snap:
            timeline.append(snapshot(stations, t))
            next_snap += 1.0
    cap.release()
    writer.release()
    _to_h264(OUT / "annotated.mp4")
    end_t = (i - 1) / fps
    for s in stations:  # close out open state
        if mode == "transit":
            for ev in list(s.pending):
                _finalise(s, ev, pres)
        elif s.in_cycle and s.last_hot - s.cycle_start >= m["min_cycle_s"]:
            s.cycles.append((s.cycle_start, s.last_hot))
        _set_state(s, "idle", end_t)
        s.intervals.append((s.state_since, end_t, s.state))

    n_ev = write_events(stations, mode, db_path, end_t, Path(video).name, cfg.get("line", "CAM1"), cfg.get("clip_shift", "CLIP"))

    summary = {"mode": mode, "video_s": round(end_t, 1), "events_written": n_ev, "stations": {}}
    for s in stations:
        d = {"entered": s.n_in, "left": s.n_out, "completed_pieces": len(s.cycles),
             "avg_cycle_s": round(s.avg_cycle(), 1), "present_at_start": s.present_at_start,
             "still_present_at_end": s.occupied, "events": s.events}
        if mode == "transit":
            if s.present_at_start and s.events and s.events[0]["type"] == "leave":
                d["censored_dwell_lower_bound_s"] = s.events[0]["t"]          # entered before the clip began
            if s.open_enter is not None:
                d["open_dwell_so_far_s"] = round(end_t - s.open_enter, 1)      # still inside at clip end
        summary["stations"][s.id] = d
    (OUT / "timeline.json").write_text(json.dumps(timeline))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def print_summary(summary: dict) -> None:
    print(f"\nmode={summary['mode']}  video={summary['video_s']}s")
    print(f"{'station':<8}{'entered':>8}{'left':>6}{'completed':>11}{'avg cycle':>11}   notes")
    for sid, d in summary["stations"].items():
        notes = []
        if d["present_at_start"]:
            notes.append("unit already present at t=0")
        if "censored_dwell_lower_bound_s" in d:
            notes.append(f"left at t={d['censored_dwell_lower_bound_s']}s (dwell >= that, censored)")
        if "open_dwell_so_far_s" in d:
            notes.append(f"unit still inside at end ({d['open_dwell_so_far_s']}s so far)")
        ev = ", ".join(f"{e['type']}@{e['t']}s" for e in d["events"])
        print(f"{sid:<8}{d['entered']:>8}{d['left']:>6}{d['completed_pieces']:>11}"
              f"{(str(d['avg_cycle_s']) + 's') if d['completed_pieces'] else '-':>11}   {'; '.join(notes)}  [{ev}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "data" / "input_stable.mp4"))
    ap.add_argument("--zones", default=str(ROOT / "data" / "zones.yaml"))
    ap.add_argument("--db", default=None, help="events DB (default data/events.db or $LINE_DB)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (speed)")
    ap.add_argument("--limit-s", type=float)
    ap.add_argument("--piece-mode", choices=["transit", "occupancy"], help="override piece_mode from the zones file")
    ap.add_argument("--preview", action="store_true", help="only render the zone overlay on frame 0")
    a = ap.parse_args()
    s = run(Path(a.video), Path(a.zones), Path(a.db) if a.db else None, a.stride, a.limit_s, a.preview, a.piece_mode)
    if s:
        print_summary(s)

