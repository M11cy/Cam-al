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
MODEL_NAME = "yolov8s.pt"
POSE_MODEL_NAME = "yolov8n-pose.pt"
PERSON_CLASS = "person"
PHONE_CLASS = "cell phone"
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10


Box = Tuple[int, int, int, int]


@dataclass
class AppConfig:
    roi: Optional[Box] = None
    absence_seconds: float = 4.0
    phone_alert_seconds: float = 60.0
    inference_confidence: float = 0.15
    object_img_size: int = 960
    pose_img_size: int = 640
    confidence: float = 0.35
    phone_confidence: float = 0.35
    phone_confirm_score: float = 0.58
    phone_min_area_ratio: float = 0.0001
    phone_max_area_ratio: float = 0.06
    phone_min_aspect_ratio: float = 0.35
    phone_max_aspect_ratio: float = 2.9
    phone_person_zone_scale: float = 1.55
    keypoint_confidence: float = 0.25
    head_down_ratio: float = 0.045
    phone_hand_distance_ratio: float = 0.32
    require_head_down_for_phone: bool = True
    require_phone_near_hand: bool = True


@dataclass
class Detection:
    label: str
    confidence: float
    box: Box


@dataclass
class PoseDetection:
    confidence: float
    box: Box
    keypoints: np.ndarray
    keypoint_confidences: np.ndarray


@dataclass
class PhoneAssessment:
    phone: Detection
    score: float
    confirmed: bool
    reason: str


def load_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        return AppConfig()

    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    roi = tuple(data["roi"]) if data.get("roi") else None
    phone_confidence = float(data.get("phone_confidence", 0.35))
    if phone_confidence in (0.25, 0.55):
        phone_confidence = 0.35

    return AppConfig(
        roi=roi,
        absence_seconds=float(data.get("absence_seconds", 4.0)),
        phone_alert_seconds=float(data.get("phone_alert_seconds", 60.0)),
        inference_confidence=float(data.get("inference_confidence", 0.15)),
        object_img_size=int(data.get("object_img_size", 960)),
        pose_img_size=int(data.get("pose_img_size", 640)),
        confidence=float(data.get("confidence", 0.35)),
        phone_confidence=phone_confidence,
        phone_confirm_score=float(data.get("phone_confirm_score", 0.58)),
        phone_min_area_ratio=float(data.get("phone_min_area_ratio", 0.0001)),
        phone_max_area_ratio=float(data.get("phone_max_area_ratio", 0.06)),
        phone_min_aspect_ratio=float(data.get("phone_min_aspect_ratio", 0.35)),
        phone_max_aspect_ratio=float(data.get("phone_max_aspect_ratio", 2.9)),
        phone_person_zone_scale=float(data.get("phone_person_zone_scale", 1.55)),
        keypoint_confidence=float(data.get("keypoint_confidence", 0.25)),
        head_down_ratio=float(data.get("head_down_ratio", 0.045)),
        phone_hand_distance_ratio=float(data.get("phone_hand_distance_ratio", 0.24)),
        require_head_down_for_phone=bool(data.get("require_head_down_for_phone", True)),
        require_phone_near_hand=bool(data.get("require_phone_near_hand", True)),
    )


def save_config(config: AppConfig) -> None:
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "roi": list(config.roi) if config.roi else None,
                "absence_seconds": config.absence_seconds,
                "phone_alert_seconds": config.phone_alert_seconds,
                "inference_confidence": config.inference_confidence,
                "object_img_size": config.object_img_size,
                "pose_img_size": config.pose_img_size,
                "confidence": config.confidence,
                "phone_confidence": config.phone_confidence,
                "phone_confirm_score": config.phone_confirm_score,
                "phone_min_area_ratio": config.phone_min_area_ratio,
                "phone_max_area_ratio": config.phone_max_area_ratio,
                "phone_min_aspect_ratio": config.phone_min_aspect_ratio,
                "phone_max_aspect_ratio": config.phone_max_aspect_ratio,
                "phone_person_zone_scale": config.phone_person_zone_scale,
                "keypoint_confidence": config.keypoint_confidence,
                "head_down_ratio": config.head_down_ratio,
                "phone_hand_distance_ratio": config.phone_hand_distance_ratio,
                "require_head_down_for_phone": config.require_head_down_for_phone,
                "require_phone_near_hand": config.require_phone_near_hand,
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


def box_iou(first: Box, second: Box) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = box_area((ix1, iy1, ix2, iy2))
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def distance(first: Tuple[int, int], second: Tuple[int, int]) -> float:
    return float(np.hypot(first[0] - second[0], first[1] - second[1]))


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


def phone_shape_score(phone: Detection, frame_width: int, frame_height: int, config: AppConfig) -> Tuple[float, str]:
    x1, y1, x2, y2 = phone.box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    area_ratio = box_area(phone.box) / max(1, frame_width * frame_height)
    aspect_ratio = box_width / box_height

    if area_ratio < config.phone_min_area_ratio:
        return 0.0, "too small"
    if area_ratio > config.phone_max_area_ratio:
        return 0.0, "too large"
    if aspect_ratio < config.phone_min_aspect_ratio or aspect_ratio > config.phone_max_aspect_ratio:
        return 0.1, "bad shape"
    return 0.2, "shape ok"


def keypoint(pose: PoseDetection, index: int, config: AppConfig) -> Optional[Tuple[int, int]]:
    if pose.keypoint_confidences[index] < config.keypoint_confidence:
        return None
    x, y = pose.keypoints[index]
    return int(x), int(y)


def keypoints_for(pose: PoseDetection, indexes: List[int], config: AppConfig) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for index in indexes:
        point = keypoint(pose, index, config)
        if point:
            points.append(point)
    return points


def is_head_down(pose: PoseDetection, config: AppConfig) -> bool:
    nose = keypoint(pose, NOSE, config)
    eyes = keypoints_for(pose, [LEFT_EYE, RIGHT_EYE], config)
    shoulders = keypoints_for(pose, [LEFT_SHOULDER, RIGHT_SHOULDER], config)
    person_height = max(1, pose.box[3] - pose.box[1])

    if nose and eyes:
        eye_y = sum(point[1] for point in eyes) / len(eyes)
        return (nose[1] - eye_y) / person_height >= config.head_down_ratio

    if nose and shoulders:
        shoulder_y = sum(point[1] for point in shoulders) / len(shoulders)
        return (shoulder_y - nose[1]) / person_height < 0.33

    return False


def is_phone_near_hand(phone: Detection, pose: PoseDetection, config: AppConfig) -> bool:
    hand_points = keypoints_for(pose, [LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW], config)
    if not hand_points:
        return False

    phone_center = box_center(phone.box)
    person_height = max(1, pose.box[3] - pose.box[1])
    max_distance = person_height * config.phone_hand_distance_ratio
    return any(distance(phone_center, point) <= max_distance for point in hand_points)


def find_matching_pose(person: Detection, poses: List[PoseDetection]) -> Optional[PoseDetection]:
    best_pose: Optional[PoseDetection] = None
    best_iou = 0.0
    for pose in poses:
        iou = box_iou(person.box, pose.box)
        if iou > best_iou:
            best_iou = iou
            best_pose = pose
    return best_pose if best_iou >= 0.2 else None


def assess_phones(
    phones: List[Detection],
    people_in_roi: List[Detection],
    poses: List[PoseDetection],
    workplace: Box,
    frame_width: int,
    frame_height: int,
    config: AppConfig,
) -> List[PhoneAssessment]:
    assessments: List[PhoneAssessment] = []
    for phone in phones:
        score = 0.0
        reasons: List[str] = []
        phone_center = box_center(phone.box)

        if not point_in_box(phone_center, workplace):
            assessments.append(PhoneAssessment(phone, 0.0, False, "outside zone"))
            continue

        shape_score, shape_reason = phone_shape_score(phone, frame_width, frame_height, config)
        if shape_score == 0.0:
            assessments.append(PhoneAssessment(phone, 0.0, False, shape_reason))
            continue
        score += shape_score
        reasons.append(shape_reason)

        if phone.confidence >= config.phone_confidence:
            score += 0.25
            reasons.append("conf ok")
        else:
            score += 0.1
            reasons.append("low conf")

        best_person_score = 0.0
        best_person_reasons: List[str] = []
        for person in people_in_roi:
            pose = find_matching_pose(person, poses)
            person_zone = expand_box(person.box, frame_width, frame_height, scale=config.phone_person_zone_scale)
            if not point_in_box(phone_center, person_zone):
                continue

            person_score = 0.15
            person_reasons = ["near person"]
            if pose:
                phone_near_hand = is_phone_near_hand(phone, pose, config)
                head_down = is_head_down(pose, config)
                if phone_near_hand:
                    person_score += 0.25
                    person_reasons.append("near hand")
                elif config.require_phone_near_hand:
                    person_score -= 0.15
                    person_reasons.append("not near hand")

                if head_down:
                    person_score += 0.2
                    person_reasons.append("head down")
                elif config.require_head_down_for_phone:
                    person_score -= 0.1
                    person_reasons.append("head up")
            else:
                person_reasons.append("no pose")

            if person_score > best_person_score:
                best_person_score = person_score
                best_person_reasons = person_reasons

        score += best_person_score
        reasons.extend(best_person_reasons)
        confirmed = score >= config.phone_confirm_score
        assessments.append(PhoneAssessment(phone, score, confirmed, ", ".join(reasons) or "not near person"))

    return assessments


def detect(model: YOLO, frame: np.ndarray, config: AppConfig) -> List[Detection]:
    result = model.predict(frame, conf=config.inference_confidence, imgsz=config.object_img_size, verbose=False)[0]
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


def detect_poses(model: YOLO, frame: np.ndarray, config: AppConfig) -> List[PoseDetection]:
    result = model.predict(frame, conf=config.confidence, imgsz=config.pose_img_size, verbose=False)[0]
    if result.keypoints is None:
        return []

    keypoints = result.keypoints.xy.cpu().numpy()
    keypoint_confidences = result.keypoints.conf.cpu().numpy()
    poses: List[PoseDetection] = []

    for index, item in enumerate(result.boxes):
        confidence = float(item.conf[0])
        x1, y1, x2, y2 = item.xyxy[0].tolist()
        poses.append(
            PoseDetection(
                confidence=confidence,
                box=(int(x1), int(y1), int(x2), int(y2)),
                keypoints=keypoints[index],
                keypoint_confidences=keypoint_confidences[index],
            )
        )

    return poses


def draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    x, y = origin
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def draw_pose_points(frame: np.ndarray, pose: PoseDetection, config: AppConfig) -> None:
    for index in [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]:
        point = keypoint(pose, index, config)
        if point:
            cv2.circle(frame, point, 4, (255, 210, 80), -1)


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
    pose_model = YOLO(POSE_MODEL_NAME)
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
        poses = detect_poses(pose_model, frame, config)

        people = [d for d in detections if d.label == PERSON_CLASS]
        phone_candidates = [d for d in detections if d.label == PHONE_CLASS]
        workplace = config.roi or (0, 0, width, height)
        people_in_roi = [person for person in people if point_in_box(box_center(person.box), workplace)]
        present = bool(people_in_roi)

        phone_assessments = assess_phones(
            phone_candidates,
            people_in_roi,
            poses,
            workplace,
            width,
            height,
            config,
        )
        confirmed_phones = [assessment.phone for assessment in phone_assessments if assessment.confirmed]

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
            pose = find_matching_pose(person, poses)
            cv2.rectangle(frame, person.box[:2], person.box[2:], color, 2)
            draw_label(frame, f"person {person.confidence:.2f}", (person.box[0], max(24, person.box[1] - 8)), color)
            if pose:
                draw_pose_points(frame, pose, config)
                posture = "head down" if is_head_down(pose, config) else "head up"
                draw_label(frame, posture, (person.box[0], min(height - 24, person.box[3] + 24)), (255, 210, 80))

        for assessment in phone_assessments:
            phone = assessment.phone
            color = (60, 80, 255) if assessment.confirmed else (150, 150, 150)
            label = "phone in hand" if assessment.confirmed else "phone candidate"
            cv2.rectangle(frame, phone.box[:2], phone.box[2:], color, 2)
            draw_label(
                frame,
                f"{label} conf {phone.confidence:.2f} score {assessment.score:.2f}",
                (phone.box[0], max(24, phone.box[1] - 8)),
                color,
            )
            if not assessment.confirmed:
                draw_label(frame, assessment.reason[:52], (phone.box[0], min(height - 24, phone.box[3] + 24)), color)

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
