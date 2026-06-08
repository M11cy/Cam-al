import argparse
import time
from pathlib import Path

import cv2


def next_frame_name(output_dir: Path) -> Path:
    index = 1
    while True:
        path = output_dir / f"frame_{index:05d}.jpg"
        if not path.exists():
            return path
        index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture training frames from a local camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, default: 0")
    parser.add_argument("--out", default="dataset/raw", help="Output directory")
    parser.add_argument("--count", type=int, default=0, help="Auto-capture this many frames, 0 = manual")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between auto-captured frames")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera with index {args.camera}")

    window = "Capture training frames"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    saved = 0
    last_capture_at = 0.0

    print("Controls: Space = save frame, q/Esc = quit")
    if args.count:
        print(f"Auto-capturing {args.count} frames every {args.interval:.1f}s")

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        preview = frame.copy()
        cv2.putText(
            preview,
            f"saved: {saved} | Space: save | q/Esc: quit",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, preview)

        now = time.monotonic()
        if args.count and saved < args.count and now - last_capture_at >= args.interval:
            path = next_frame_name(output_dir)
            cv2.imwrite(str(path), frame)
            saved += 1
            last_capture_at = now
            print(f"Saved {path}")
            if saved >= args.count:
                break

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            path = next_frame_name(output_dir)
            cv2.imwrite(str(path), frame)
            saved += 1
            print(f"Saved {path}")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
