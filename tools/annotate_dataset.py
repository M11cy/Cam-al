import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


CLASSES = {
    "p": (0, "person"),
    "t": (1, "phone"),
    "o": (2, "other_object"),
}

COLORS = {
    0: (80, 220, 120),
    1: (60, 80, 255),
    2: (150, 150, 150),
}

Box = Tuple[int, int, int, int, int]


@dataclass
class AnnotationState:
    image = None
    boxes: List[Box] = None
    selected_class_id: int = 1
    selected_class_name: str = "phone"
    drawing: bool = False
    start: Optional[Tuple[int, int]] = None
    current: Optional[Tuple[int, int]] = None


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


def normalize_box(class_id: int, start: Tuple[int, int], end: Tuple[int, int], width: int, height: int) -> Optional[Box]:
    x1, y1 = start
    x2, y2 = end
    left = max(0, min(x1, x2))
    top = max(0, min(y1, y2))
    right = min(width - 1, max(x1, x2))
    bottom = min(height - 1, max(y1, y2))
    if right - left < 8 or bottom - top < 8:
        return None
    return class_id, left, top, right, bottom


def draw_boxes(image, boxes: List[Box]):
    preview = image.copy()
    names = {value[0]: value[1] for value in CLASSES.values()}
    for class_id, x1, y1, x2, y2 in boxes:
        color = COLORS[class_id]
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(preview, names[class_id], (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return preview


def draw_state(state: AnnotationState, image_name: str):
    preview = draw_boxes(state.image, state.boxes)
    if state.drawing and state.start and state.current:
        height, width = state.image.shape[:2]
        draft = normalize_box(state.selected_class_id, state.start, state.current, width, height)
        if draft:
            _, x1, y1, x2, y2 = draft
            cv2.rectangle(preview, (x1, y1), (x2, y2), COLORS[state.selected_class_id], 2)

    help_text = (
        f"{image_name} | selected: {state.selected_class_name} | "
        "drag mouse to draw | p person | t phone | o other | u undo | n save | s empty | q quit"
    )
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 48), (24, 24, 24), -1)
    cv2.putText(preview, help_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return preview


def on_mouse(event, x, y, _flags, state: AnnotationState) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.start = (x, y)
        state.current = (x, y)
        return

    if event == cv2.EVENT_MOUSEMOVE and state.drawing:
        state.current = (x, y)
        return

    if event == cv2.EVENT_LBUTTONUP and state.drawing:
        state.drawing = False
        state.current = (x, y)
        height, width = state.image.shape[:2]
        box = normalize_box(state.selected_class_id, state.start, state.current, width, height)
        if box:
            state.boxes.append(box)
            print(f"Added {state.selected_class_name}: {box[1:]}")
        state.start = None
        state.current = None


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
    state = AnnotationState()
    cv2.setMouseCallback(window, on_mouse, state)
    print("Draw boxes by dragging the mouse.")
    print("Keys: p=person, t=phone, o=other_object, u=undo, n=save next, s=save empty/skip, q=quit")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        state.image = image
        state.boxes = []
        state.drawing = False
        state.start = None
        state.current = None

        while True:
            cv2.imshow(window, draw_state(state, image_path.name))
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if key == 255:
                continue
            if key == ord("u") and state.boxes:
                state.boxes.pop()
                continue
            if key == ord("n"):
                save_annotation(image_path, image, state.boxes, dataset_dir, args.val_ratio)
                break
            if key == ord("s"):
                save_annotation(image_path, image, [], dataset_dir, args.val_ratio)
                break

            char = chr(key).lower()
            if char in CLASSES:
                state.selected_class_id, state.selected_class_name = CLASSES[char]
                print(f"Selected {state.selected_class_name}. Drag the mouse to draw a box.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
