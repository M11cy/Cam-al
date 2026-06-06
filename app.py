import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


CONFIG_PATH = Path("config.json")
MODEL_NAME = "yolov8n.pt"
PERSON_CLASS = "person"
PHONE_CLASS = "cell phone"


Box = Tuple[int, int, int, int]


@dataclass
class AppConfig:
    roi: Optional[Box] = None
    absence_seconds: float = 4.0
    phone_alert_seconds: float = 60.0
    confidence: float = 0.35
    phone_confidence: float = 0.55
    phone_min_area_ratio: float = 0.0005
    phone_max_area_ratio: float = 0.04
    phone_min_aspect_ratio: float = 0.35
    phone_max_aspect_ratio: float = 2.9
    phone_person_zone_scale: float = 1.55


@dataclass
class Detection:
    label: str
    confidence: float
    box: Box


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        return AppConfig()

    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    roi = tuple(data["roi"]) if data.get("roi") else None
    phone_confidence = float(data.get("phone_confidence", 0.55))
    if phone_confidence == 0.25:
        phone_confidence = 0.55

    return AppConfig(
        roi=roi,
        absence_seconds=float(data.get("absence_seconds", 4.0)),
        phone_alert_seconds=float(data.get("phone_alert_seconds", 60.0)),
        confidence=float(data.get("confidence", 0.35)),
        phone_confidence=phone_confidence,
        phone_min_area_ratio=float(data.get("phone_min_area_ratio", 0.0005)),
        phone_max_area_ratio=float(data.get("phone_max_area_ratio", 0.04)),
        phone_min_aspect_ratio=float(data.get("phone_min_aspect_ratio", 0.35)),
        phone_max_aspect_ratio=float(data.get("phone_max_aspect_ratio", 2.9)),
        phone_person_zone_scale=float(data.get("phone_person_zone_scale", 1.55)),
    )


def save_config(config: AppConfig) -> None:
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "roi": list(config.roi) if config.roi else None,
                "absence_seconds": config.absence_seconds,
                "phone_alert_seconds": config.phone_alert_seconds,
                "confidence": config.confidence,
                "phone_confidence": config.phone_confidence,
                "phone_min_area_ratio": config.phone_min_area_ratio,
                "phone_max_area_ratio": config.phone_max_area_ratio,
                "phone_min_aspect_ratio": config.phone_min_aspect_ratio,
                "phone_max_aspect_ratio": config.phone_max_aspect_ratio,
                "phone_person_zone_scale": config.phone_person_zone_scale,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def box_center(box: Box) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def point_in_box(point: Tuple[int, int], box: Box) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def expand_box(box: Box, width: int, height: int, scale: float = 1.25) -> Box:
    x1, y1, x2, y2 = box
    cx, cy = box_center(box)
    w = int((x2 - x1) * scale)
    h = int((y2 - y1) * scale)
    return (
        max(0, cx - w // 2),
        max(0, cy - h // 2),
        min(width, cx + w // 2),
        min(height, cy + h // 2),
    )


def is_phone_like_shape(phone: Detection, frame_width: int, frame_height: int, config: AppConfig) -> bool:
    x1, y1, x2, y2 = phone.box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    area_ratio = box_area(phone.box) / max(1, frame_width * frame_height)
    aspect_ratio = box_width / box_height

    return (
        phone.confidence >= config.phone_confidence
        and config.phone_min_area_ratio <= area_ratio <= config.phone_max_area_ratio
        and config.phone_min_aspect_ratio <= aspect_ratio <= config.phone_max_aspect_ratio
    )


def filter_confirmed_phones(
    phones: List[Detection],
    people_in_roi: List[Detection],
    workplace: Box,
    frame_width: int,
    frame_height: int,
    config: AppConfig,
) -> List[Detection]:
    confirmed: List[Detection] = []
    for phone in phones:
        phone_center = box_center(phone.box)
        if not point_in_box(phone_center, workplace):
            continue
        if not is_phone_like_shape(phone, frame_width, frame_height, config):
            continue

        for person in people_in_roi:
            person_zone = expand_box(person.box, frame_width, frame_height, scale=config.phone_person_zone_scale)
            if point_in_box(phone_center, person_zone):
                confirmed.append(phone)
                break

    return confirmed


def detect(model: YOLO, frame: np.ndarray, config: AppConfig) -> List[Detection]:
    result = model.predict(frame, conf=config.phone_confidence, verbose=False)[0]
    names: Dict[int, str] = result.names
    detections: List[Detection] = []

    for item in result.boxes:
        cls_id = int(item.cls[0])
        label = names.get(cls_id, str(cls_id))
        confidence = float(item.conf[0])
        if label == PERSON_CLASS and confidence < config.confidence:
            continue
        if label != PERSON_CLASS and label != PHONE_CLASS:
            continue

        x1, y1, x2, y2 = item.xyxy[0].tolist()
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                box=(int(x1), int(y1), int(x2), int(y2)),
            )
        )

    return detections


def draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    x, y = origin
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def draw_status_panel(
    frame: np.ndarray,
    present: bool,
    phone_detected: bool,
    phone_alert: bool,
    absence_elapsed: float,
    phone_elapsed: float,
    config: AppConfig,
) -> None:
    panel_h = 140
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_h), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    presence_text = "PRESENT" if present else "ABSENT"
    presence_color = (80, 220, 120) if present else (60, 80, 255)
    phone_text = "PHONE ALERT" if phone_alert else "PHONE DETECTED" if phone_detected else "NO PHONE"
    phone_color = (60, 80, 255) if phone_alert else (255, 210, 80) if phone_detected else (190, 190, 190)

    draw_label(frame, f"Employee: {presence_text}", (18, 35), presence_color)
    draw_label(frame, f"Phone: {phone_text}", (18, 68), phone_color)

    if not present:
        alert_after = max(0.0, config.absence_seconds - absence_elapsed)
        text = f"Absent for {absence_elapsed:.1f}s, alert in {alert_after:.1f}s"
        if absence_elapsed >= config.absence_seconds:
            text = f"EMPLOYEE ALERT: absent for {absence_elapsed:.1f}s"
        draw_label(frame, text, (18, 101), (60, 80, 255))

    if phone_detected:
        alert_after = max(0.0, config.phone_alert_seconds - phone_elapsed)
        text = f"Phone visible for {phone_elapsed:.1f}s, alert in {alert_after:.1f}s"
        if phone_alert:
            text = f"PHONE ALERT: visible for {phone_elapsed:.1f}s"
        draw_label(frame, text, (18, 130), phone_color)


def select_roi(frame: np.ndarray) -> Optional[Box]:
    selected = cv2.selectROI("Workplace monitor", frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = selected
    if w <= 0 or h <= 0:
        return None
    return int(x), int(y), int(x + w), int(y + h)


def run(camera_index: int) -> None:
    config = load_config()
    model = YOLO(MODEL_NAME)
    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera with index {camera_index}")

    last_seen_at = time.monotonic()
    phone_first_seen_at: Optional[float] = None
    phone_seen_until = 0.0

    cv2.namedWindow("Workplace monitor", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        detections = detect(model, frame, config)

        people = [d for d in detections if d.label == PERSON_CLASS]
        phone_candidates = [d for d in detections if d.label == PHONE_CLASS]
        workplace = config.roi or (0, 0, width, height)

        people_in_roi = [person for person in people if point_in_box(box_center(person.box), workplace)]
        present = bool(people_in_roi)
        confirmed_phones = filter_confirmed_phones(
            phone_candidates,
            people_in_roi,
            workplace,
            width,
            height,
            config,
        )

        now = time.monotonic()
        if present:
            last_seen_at = now

        if confirmed_phones:
            phone_seen_until = now + 1.5

        phone_detected = now < phone_seen_until
        if phone_detected:
            if phone_first_seen_at is None:
                phone_first_seen_at = now
        else:
            phone_first_seen_at = None

        phone_elapsed = now - phone_first_seen_at if phone_first_seen_at is not None else 0.0
        phone_alert = phone_elapsed >= config.phone_alert_seconds
        absence_elapsed = now - last_seen_at

        cv2.rectangle(frame, workplace[:2], workplace[2:], (255, 210, 80), 2)
        draw_label(frame, "work zone", (workplace[0] + 8, max(24, workplace[1] + 24)), (255, 210, 80))

        for person in people:
            color = (80, 220, 120) if person in people_in_roi else (170, 170, 170)
            cv2.rectangle(frame, person.box[:2], person.box[2:], color, 2)
            draw_label(frame, f"person {person.confidence:.2f}", (person.box[0], max(24, person.box[1] - 8)), color)

        for phone in phone_candidates:
            confirmed = phone in confirmed_phones
            color = (60, 80, 255) if confirmed else (150, 150, 150)
            label = "phone" if confirmed else "ignored phone candidate"
            cv2.rectangle(frame, phone.box[:2], phone.box[2:], color, 2)
            draw_label(frame, f"{label} {phone.confidence:.2f}", (phone.box[0], max(24, phone.box[1] - 8)), color)

        draw_status_panel(frame, present, phone_detected, phone_alert, absence_elapsed, phone_elapsed, config)
        draw_label(frame, "r: set zone  s: save  q/Esc: quit", (18, height - 18), (235, 235, 235))

        cv2.imshow("Workplace monitor", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            selected = select_roi(frame)
            if selected:
                config.roi = selected
        if key == ord("s"):
            save_config(config)
            print(f"Saved: {CONFIG_PATH.resolve()}")

    capture.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local workplace camera monitor")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, default: 0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.camera)
