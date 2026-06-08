import argparse
import hashlib
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2


CLASSES = {
    "p": (0, "person"),
    "t": (1, "phone"),
    "o": (2, "other_object"),
}

Box = Tuple[int, int, int, int, int]


def yolo_line(box: Box, width: int, height: int) -> str:
    class_id, x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def split_for(path: Path, val_ratio: float) -> str:
    digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_ratio else "train"


def draw_boxes(image, boxes: List[Box]):
    preview = image.copy()
    colors = {
        0: (80, 220, 120),
        1: (60, 80, 255),
        2: (150, 150, 150),
    }
    names = {value[0]: value[1] for value in CLASSES.values()}
    for class_id, x1, y1, x2, y2 in boxes:
        color = colors[class_id]
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(preview, names[class_id], (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return preview


def save_annotation(image_path: Path, image, boxes: List[Box], dataset_dir: Path, val_ratio: float) -> None:
    split = split_for(image_path, val_ratio)
    image_out = dataset_dir / "images" / split / image_path.name
    label_out = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, image_out)
    height, width = image.shape[:2]
    label_out.write_text("\n".join(yolo_line(box, width, height) for box in boxes), encoding="utf-8")
    print(f"Saved {image_out} with {len(boxes)} boxes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate frames into YOLO labels")
    parser.add_argument("--raw", default="dataset/raw", help="Directory with captured frames")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    dataset_dir = Path(args.dataset)
    image_paths = sorted(
        path for path in raw_dir.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {raw_dir}")

    window = "Annotate dataset"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Keys: p=person, t=phone, o=other_object, u=undo, n=save next, s=save empty/skip, q=quit")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        boxes: List[Box] = []
        while True:
            preview = draw_boxes(image, boxes)
            cv2.putText(
                preview,
                f"{image_path.name} | p person | t phone | o other | u undo | n save | s empty | q quit",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if key == ord("u") and boxes:
                boxes.pop()
                continue
            if key == ord("n"):
                save_annotation(image_path, image, boxes, dataset_dir, args.val_ratio)
                break
            if key == ord("s"):
                save_annotation(image_path, image, [], dataset_dir, args.val_ratio)
                break
            char = chr(key).lower()
            if char in CLASSES:
                class_id, name = CLASSES[char]
                selected = cv2.selectROI(window, preview, fromCenter=False, showCrosshair=True)
                x, y, w, h = selected
                if w > 0 and h > 0:
                    boxes.append((class_id, int(x), int(y), int(x + w), int(y + h)))
                    print(f"Added {name}: {(int(x), int(y), int(x + w), int(y + h))}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
