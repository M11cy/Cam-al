import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom phone detector")
    parser.add_argument("--data", default="dataset/data.yaml", help="YOLO data.yaml")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--name", default="shop-phone")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise RuntimeError(f"Missing {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )


if __name__ == "__main__":
    main()
