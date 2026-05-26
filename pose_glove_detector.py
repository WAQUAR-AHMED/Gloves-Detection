from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

try:
    from save_json_log import save_json_log_to_database
except ImportError:  # pragma: no cover
    save_json_log_to_database = None

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: opencv-python. Install with "
        "`venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
    ) from exc

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: ultralytics. Install with "
        "`venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
    ) from exc


LOGGER = logging.getLogger("pose_glove_detector")
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "Models"
DEFAULT_GLOVE_MODEL = MODELS_DIR / "best.pt"
DEFAULT_POSE_MODEL = "Models/yolo11m-pose.pt"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
DEFAULT_CONF_CONFIG_PATH = PROJECT_DIR / "class_confidence_config.yaml"

GLOVES_ALIASES = {"gloves", "glove", "gloved"}
NO_GLOVES_ALIASES = {"no_gloves", "no_glove", "not_gloved", "not_gloves"}


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class PersonDetection:
    person_id: int
    bbox: tuple[int, int, int, int]
    detections: List[Detection]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect people with YOLO pose, then detect gloves/no_gloves per person."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image file or directory path.",
    )
    parser.add_argument(
        "--pose-model",
        default=DEFAULT_POSE_MODEL,
        help=f"Pose model path/name. Default: {DEFAULT_POSE_MODEL}",
    )
    parser.add_argument(
        "--glove-model",
        default=str(DEFAULT_GLOVE_MODEL),
        help=f"Glove model path. Default: {DEFAULT_GLOVE_MODEL}",
    )
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=None,
        help="Confidence threshold for person detection. Overrides config when provided.",
    )
    parser.add_argument(
        "--glove-conf",
        type=float,
        default=None,
        help="Base confidence threshold for glove model. Overrides config when provided.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for annotated images and JSON logs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--conf-config",
        default=str(DEFAULT_CONF_CONFIG_PATH),
        help=(
            "Path to YAML confidence config. "
            f"Default: {DEFAULT_CONF_CONFIG_PATH}"
        ),
    )
    return parser.parse_args()


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def canonical_glove_label(label: str) -> str:
    normalized = normalize_label(label)
    if normalized in GLOVES_ALIASES:
        return "gloves"
    if normalized in NO_GLOVES_ALIASES:
        return "no_gloves"
    return normalized


def detection_log_label(label: str) -> str:
    if label == "gloves":
        return "gloved_hand"
    if label == "no_gloves":
        return "bare_hand"
    return label


def collect_images(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        images: List[Path] = []
        for pattern in patterns:
            images.extend(sorted(source.glob(pattern)))
        return images
    return []


def parse_simple_yaml(config_text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in config_text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if not value:
            new_map: dict[str, Any] = {}
            parent[key] = new_map
            stack.append((indent, new_map))
            continue

        lowered = value.lower()
        if lowered in {"true", "false"}:
            parsed_value: Any = lowered == "true"
        else:
            try:
                parsed_value = float(value) if "." in value else int(value)
            except ValueError:
                parsed_value = value.strip("'").strip('"')
        parent[key] = parsed_value

    return root


def load_conf_config(config_path: Path) -> tuple[float, float, dict[str, float]]:
    if not config_path.exists():
        LOGGER.warning("Confidence config not found: %s", config_path)
        return 0.25, 0.25, {}

    try:
        payload = parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        LOGGER.warning("Failed to read confidence config %s: %s", config_path, exc)
        return 0.25, 0.25, {}

    pose_conf = payload.get("pose_conf", 0.25)
    glove_conf = payload.get("glove_conf", {})
    if not isinstance(pose_conf, (int, float)):
        pose_conf = 0.25
    if not isinstance(glove_conf, dict):
        glove_conf = {}

    default_conf = glove_conf.get("default", 0.25)
    class_conf = glove_conf.get("class_conf", {})
    if not isinstance(class_conf, dict):
        class_conf = {}
    if not isinstance(default_conf, (int, float)):
        default_conf = 0.25

    normalized: dict[str, float] = {}
    for label, conf in class_conf.items():
        if not isinstance(label, str) or not isinstance(conf, (int, float)):
            continue
        normalized[canonical_glove_label(label)] = float(conf)

    return float(pose_conf), float(default_conf), normalized


def filter_detections_by_class_conf(
    detections: List[Detection],
    default_conf: float,
    class_conf: dict[str, float],
) -> List[Detection]:
    filtered: List[Detection] = []
    for detection in detections:
        threshold = class_conf.get(detection.label, default_conf)
        if detection.confidence >= threshold:
            filtered.append(detection)
    return filtered


def run_pose_person_detection(pose_model: YOLO, image, pose_conf: float) -> List[tuple[int, int, int, int]]:
    results = pose_model.predict(source=image, conf=pose_conf, verbose=False)
    result = results[0]
    persons: List[tuple[int, int, int, int]] = []
    if result.boxes is None:
        return persons

    names = result.names
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label = normalize_label(str(names[class_id]))
        if label != "person":
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        persons.append((x1, y1, x2, y2))
    return persons


def run_glove_detection_on_crop(
    glove_model: YOLO,
    crop,
    offset_x: int,
    offset_y: int,
    glove_conf: float,
) -> List[Detection]:
    results = glove_model.predict(source=crop, conf=glove_conf, verbose=False)
    result = results[0]
    detections: List[Detection] = []
    if result.boxes is None:
        return detections

    names = result.names
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label = canonical_glove_label(str(names[class_id]))
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        detections.append(
            Detection(
                label=label,
                confidence=confidence,
                bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
            )
        )
    return detections


def summarize_person_status(detections: List[Detection]) -> str:
    has_gloves = any(det.label == "gloves" for det in detections)
    has_bare_hands = any(det.label == "no_gloves" for det in detections)
    if has_gloves and has_bare_hands:
        return "mixed"
    if has_gloves:
        return "gloved_hand"
    if has_bare_hands:
        return "bare_hand"
    return "no_hand_detection"


def annotate(image, persons: List[PersonDetection]):
    annotated = image.copy()
    gloves_count = 0
    no_gloves_count = 0

    for person in persons:
        x1, y1, x2, y2 = person.bbox
        person_status = summarize_person_status(person.detections)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            annotated,
            f"person_{person.person_id} {person_status}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 180, 0),
            2,
        )

    for person in persons:
        for det in person.detections:
            x1, y1, x2, y2 = det.bbox
            display_label = detection_log_label(det.label)
            if det.label == "gloves":
                gloves_count += 1
                color = (0, 200, 0)
            elif det.label == "no_gloves":
                no_gloves_count += 1
                color = (0, 0, 255)
            else:
                color = (255, 165, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{display_label} {det.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    summary = f"persons: {len(persons)} | gloves: {gloves_count} | no_gloves: {no_gloves_count}"
    cv2.putText(annotated, summary, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(annotated, summary, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 1)
    return annotated, gloves_count, no_gloves_count


def to_json_record(
    image_path: Path,
    persons: List[PersonDetection],
) -> dict[str, Any]:
    flat_detections = [
        {
            "person_id": person.person_id,
            "label": detection_log_label(det.label),
            "confidence": round(det.confidence, 4),
            "bbox": list(det.bbox),
        }
        for person in persons
        for det in person.detections
    ]
    return {
        "filename": image_path.name,
        "detections": flat_detections,
        "persons": [
            {
                "person_id": person.person_id,
                "person_bbox": list(person.bbox),
                "status": summarize_person_status(person.detections),
                "detections": [
                    {
                        "label": detection_log_label(det.label),
                        "confidence": round(det.confidence, 4),
                        "bbox": list(det.bbox),
                    }
                    for det in person.detections
                ],
            }
            for person in persons
        ],
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    source = Path(args.source)
    glove_model_path = Path(args.glove_model)
    output_dir = Path(args.output_dir)
    conf_config_path = Path(args.conf_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not glove_model_path.exists():
        LOGGER.error("Glove model not found: %s", glove_model_path)
        return 1

    images = collect_images(source)
    if not images:
        LOGGER.error("No images found at source: %s", source)
        return 1

    pose_model = YOLO(args.pose_model)
    glove_model = YOLO(str(glove_model_path))
    config_pose_conf, default_conf, class_conf = load_conf_config(conf_config_path)
    pose_conf = args.pose_conf if args.pose_conf is not None else config_pose_conf
    glove_conf = args.glove_conf if args.glove_conf is not None else default_conf
    all_logs: List[dict[str, Any]] = []

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Skipping unreadable image: %s", image_path)
            continue

        person_boxes = run_pose_person_detection(pose_model, image, pose_conf)
        persons: List[PersonDetection] = []
        img_h, img_w = image.shape[:2]
        for person_id, (x1, y1, x2, y2) in enumerate(person_boxes, start=1):
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            detections = run_glove_detection_on_crop(glove_model, crop, x1, y1, glove_conf)
            detections = filter_detections_by_class_conf(detections, default_conf, class_conf)
            persons.append(
                PersonDetection(
                    person_id=person_id,
                    bbox=(x1, y1, x2, y2),
                    detections=detections,
                )
            )

        annotated, gloves_count, no_gloves_count = annotate(image, persons)
        out_image = output_dir / f"{image_path.stem}_annotated{image_path.suffix or '.jpg'}"
        cv2.imwrite(str(out_image), annotated)

        image_log = to_json_record(image_path, persons)
        image_log["output_image"] = str(out_image)
        image_log["gloves_count"] = gloves_count
        image_log["no_gloves_count"] = no_gloves_count

        out_json = output_dir / f"{image_path.stem}.json"
        out_json.write_text(json.dumps(image_log, indent=2), encoding="utf-8")
        if save_json_log_to_database is not None:
            save_json_log_to_database(image_log, LOGGER)
        all_logs.append(image_log)

        LOGGER.info(
            "Processed %s | persons=%s gloves=%s no_gloves=%s",
            image_path.name,
            len(persons),
            gloves_count,
            no_gloves_count,
        )

    batch_json = output_dir / "detections_all.json"
    batch_json.write_text(json.dumps(all_logs, indent=2), encoding="utf-8")
    LOGGER.info("Saved batch log to %s", batch_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
