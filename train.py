"""Train a YOLOv8n detector with the recorded DotMetrics run settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


DEFAULT_CONFIG = Path(__file__).parent / "training" / "train8-args.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8n detector using the recorded DotMetrics settings."
    )
    parser.add_argument(
        "--data", required=True, help="Path to a local Ultralytics dataset YAML file."
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Path to a training settings YAML file."
    )
    parser.add_argument("--project", default="runs/detect", help="Training output directory.")
    parser.add_argument("--name", default="dotmetrics-yolov8n", help="Run name.")
    parser.add_argument("--device", default=None, help="Training device, such as 0 or cpu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)

    base_model = settings.pop("model")
    for key in ("task", "mode", "data", "project", "name", "save_dir"):
        settings.pop(key, None)

    settings.update(data=args.data, project=args.project, name=args.name)
    if args.device is not None:
        settings["device"] = args.device

    YOLO(base_model).train(**settings)


if __name__ == "__main__":
    main()
