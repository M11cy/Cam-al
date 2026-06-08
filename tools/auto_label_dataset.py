import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
from ultralytics import YOLO


CLASS_IDS = {
    "person": 0,
    "cell phone": 1,
    "phone": 1,
    "other_object": 2,
    "other object": 2,
    "other": 2,
    "distractor": 2,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
Box = Tuple[int, int, int, int, int]


def split_for(path: Path, val_ratio: float) -> str:
    digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_ratio else "train"


def yolo_line(box: Box, width: int, height: int) -> str:
    class_id, x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def iter_images(raw_dir: Path) -> Iterable[Path]:
    return sorted(path for path in raw_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def should_keep(label: str, confidence: float, person_conf: float, phone_conf: float, other_conf: float) -> Optional[int]:
    class_id = CLASS_IDS.get(label)
    if class_id is None:
        return None
    if class_id == 0 and confidence < person_conf:
        return None
    if class_id == 1 and confidence < phone_conf:
        return None
    if class_id == 2 and confidence < other_conf:
        return None
    return class_id


def predict_boxes(model: YOLO, image_path: Path, imgsz: int, conf: float, person_conf: float, phone_conf: float, other_conf: float) -> Tuple[List[Box], int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")

    height, width = image.shape[:2]
    result = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]
    names: Dict[int, str] = result.names
    boxes: List[Box] = []

    for item in result.boxes:
        label = names.get(int(item.cls[0]), str(int(item.cls[0])))
        confidence = float(item.conf[0])
        class_id = should_keep(label, confidence, person_conf, phone_conf, other_conf)
        if class_id is None:
            continue

        x1, y1, x2, y2 = item.xyxy[0].tolist()
        boxes.append((class_id, int(x1), int(y1), int(x2), int(y2)))

    return boxes, width, height


def save_draft(image_path: Path, boxes: List[Box], width: int, height: int, raw_dir: Path, dataset_dir: Path, val_ratio: float, copy_images: bool) -> None:
    split = split_for(image_path, val_ratio)
    label_out = dataset_dir / "labels_draft" / split / f"{image_path.stem}.txt"
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\n".join(yolo_line(box, width, height) for box in boxes), encoding="utf-8")

    if copy_images:
        image_out = dataset_dir / "images_draft" / split / image_path.name
        image_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_out)

    print(f"{image_path.relative_to(raw_dir)}: {len(boxes)} draft boxes -> {label_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create draft YOLO labels from a pretrained or custom model")
    parser.add_argument("--raw", default="dataset/raw", help="Directory with captured frames")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--model", default="yolov8s.pt", help="Model for pre-labeling, for example yolov8s.pt or runs/detect/shop-phone/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15, help="Low global confidence so doubtful boxes can be corrected manually")
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--phone-conf", type=float, default=0.25)
    parser.add_argument("--other-conf", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--copy-images", action="store_true", help="Also copy images into dataset/images_draft")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    dataset_dir = Path(args.dataset)
    image_paths = list(iter_images(raw_dir))
    if not image_paths:
        raise RuntimeError(f"No images found in {raw_dir}")

    model = YOLO(args.model)
    for image_path in image_paths:
        boxes, width, height = predict_boxes(
            model,
            image_path,
            imgsz=args.imgsz,
            conf=args.conf,
            person_conf=args.person_conf,
            phone_conf=args.phone_conf,
            other_conf=args.other_conf,
        )
        save_draft(image_path, boxes, width, height, raw_dir, dataset_dir, args.val_ratio, args.copy_images)


if __name__ == "__main__":
    main()
