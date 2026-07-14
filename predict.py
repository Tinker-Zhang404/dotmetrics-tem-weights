"""Run DotMetrics YOLOv8n detector inference on a TEM image or directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_WEIGHTS = Path(__file__).with_name("yolov8n-DotMetrics.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DotMetrics YOLOv8n TEM nanoparticle detector."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image, a directory of images, or a supported video source.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to a compatible YOLOv8 weight file.",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.30, help="NMS IoU threshold.")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, such as 0 for the first GPU or cpu.",
    )
    parser.add_argument("--project", default="runs/detect", help="Output directory.")
    parser.add_argument("--name", default="predict", help="Output subdirectory name.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format labels.")
    parser.add_argument("--save-conf", action="store_true", help="Include confidences in labels.")
    parser.add_argument(
        "--exist-ok", action="store_true", help="Reuse the output directory if it exists."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Weight file not found: {weights}")

    predict_args = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "save": True,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    results = YOLO(weights).predict(**predict_args)
    if results:
        print(f"Results saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
