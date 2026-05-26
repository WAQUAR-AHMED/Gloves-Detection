# Biz-Tech Glove Compliance Detection

This repository detects people, identifies whether their hands are gloved or bare, writes annotated outputs, generates structured JSON logs, and can optionally save those logs into a database.

## Workflow

The main production flow is implemented in `pose_glove_detector.py`.

```text
Input image or folder
        |
        v
YOLO pose model detects persons
        |
        v
Each person region is cropped
        |
        v
YOLO glove model runs on each crop
        |
        v
Per-class thresholds from class_confidence_config.yaml are applied
        |
        v
Each person gets a person_id and status
        |
        v
Annotated image + per-image JSON + detections_all.json are saved
        |
        v
Optional database insert if .env enables DB logging
```

## Why YOLO Pose Helps

Using a pose model before glove detection has a practical advantage beyond person localization.

Pose estimation gives body structure awareness, especially wrist and arm keypoints. That creates a strong path for future improvement:

- glove detections can be validated against likely hand regions instead of anywhere inside the person box
- detections far from the wrist or hand area can be rejected
- this can reduce false positives caused by PPE-like objects, clothing textures, tools, or background regions
- it also creates a cleaner upgrade path for right-hand and left-hand reasoning

Planned improvement direction:

1. detect the person with YOLO pose
2. read wrist and arm keypoints
3. estimate expected hand regions from those keypoints
4. accept glove detections only when they overlap or stay close to those regions

This would make the system more spatially aware and better at confirming that the glove is actually on the hand, not just somewhere inside the person crop.

## What This Project Produces

- Person detection using a pose model
- Glove or bare-hand detection inside each person crop
- Per-person statuses:
  - `gloved_hand`
  - `bare_hand`
  - `mixed`
  - `no_hand_detection`
- Annotated images in `outputs/`
- One JSON log per image
- Batch JSON output in `outputs/detections_all.json`
- Optional database persistence of JSON logs

## Example Output

Annotated sample from this repository:

![Demo Annotation](assets\Trial_annotated.jpg)

The JSON output contains:

- `filename`
- `detections`
- `persons`
- `output_image`
- class count summaries

Each person record includes:

- `person_id`
- `person_bbox`
- `status`
- person-specific detections with `label`, `confidence`, and `bbox`

## Dataset Information

Primary project dataset reference:

- Roboflow app workspace: <https://app.roboflow.com/glovesdetection/glovesdetection/dataset>

Publicly verifiable related Roboflow source:

- Roboflow Universe dataset: <https://universe.roboflow.com/glovesdetection/ppe-cpxsz-l7w5v>

Verified public details from Roboflow Universe:

- Workspace: `glovesdetection`
- Dataset name: `PPE Dataset`
- Dataset type: Object Detection
- Public page reports 16k images
- Public page lists 11 classes including `person`, `gloves`, and `no gloves`

Important note:

- The `app.roboflow.com` URL may point to a private or workspace-specific project view.
- The public Universe dataset above is the closest publicly verifiable source I found for the same workspace name.
- If your training dataset is a different private version inside Roboflow, update this section later with the exact version details.

Kaggle training notebook:

- `[Kaggle Notebook link]`<https://www.kaggle.com/code/waquarahmed1/gloves-and-no-gloves-training-notebook?scriptVersionId=322281248>

## Model Performance

Training metrics you provided:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 280 | 363 | 0.981 | 0.954 | 0.983 | 0.823 |
| gloves | 69 | 127 | 0.967 | 0.920 | 0.971 | 0.818 |
| no-gloves | 211 | 236 | 0.994 | 0.987 | 0.995 | 0.827 |

Interpretation:

- Overall detection quality is strong.
- `no-gloves` performs slightly better than `gloves`.
- The `mAP50-95` values suggest the model is accurate but still sensitive to tighter localization quality, which is normal.

## Repository Structure

- `pose_glove_detector.py`: main two-stage detection pipeline
- `glove_pose_detector.py`: glove-only detection script
- `class_confidence_config.yaml`: pose and class-wise confidence thresholds
- `save_json_log.py`: separate database logging script
- `.env.example`: optional database configuration template
- `Models/`: trained weights
- `outputs/`: annotated images and JSON logs

## Setup

Install dependencies using the existing virtual environment:

```powershell
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Current Python package requirements:

- `ultralytics`
- `opencv-python`

## Configuration

### Class Confidence Thresholds

Edit `class_confidence_config.yaml`:

```yaml
pose_conf: 0.25
glove_conf:
  default: 0.3
  class_conf:
    gloves: 0.45
    no_gloves: 0.25
```

How thresholds work:

- `pose_conf` controls person detection confidence for the YOLO pose model
- `glove_conf.default` is the base YOLO threshold for glove detections
- `glove_conf.class_conf` applies an additional class-specific filter after inference
- `--pose-conf` and `--glove-conf` still override the YAML values when passed on the command line

### Database Setup

Copy `.env.example` to `.env` and configure it:

```env
DB_ENABLED=true
DB_BACKEND=sqlite
DB_PATH=outputs/detections.db
DB_TABLE=json_logs
```

Current database support:

- `sqlite`

Database behavior:

- If `.env` is missing, JSON logs are still saved locally
- If `DB_ENABLED=false`, database writes are skipped
- If `DB_ENABLED=true`, each JSON log is inserted into the configured SQLite database

## How Database Logging Works

Database logging is intentionally separated from the detector logic.

`save_json_log.py` is responsible for:

- loading `.env`
- checking whether database logging is enabled
- creating the SQLite table if it does not exist
- inserting the full JSON payload into the configured table

The detector still works even when the database is not configured.

## Running the Project

Run the main pipeline on a folder:

```powershell
venv\Scripts\python.exe .\pose_glove_detector.py --source .\image
```

Run the main pipeline on a single image:

```powershell
venv\Scripts\python.exe .\pose_glove_detector.py --source .\image\trial2.jpg
```

Run with explicit model thresholds:

```powershell
venv\Scripts\python.exe .\pose_glove_detector.py --source .\image --pose-conf 0.25 --glove-conf 0.20
```

Run the glove-only detector:

```powershell
venv\Scripts\python.exe .\glove_pose_detector.py --image .\image\trial2.jpg
```

## Saving JSON Logs to the Database

### Automatic save during inference

When `pose_glove_detector.py` finishes an image, it:

1. saves the JSON file to `outputs/`
2. checks database configuration
3. inserts the same JSON record into the database if enabled

### Manual save with the separate script

You can save an existing JSON log manually:

```powershell
venv\Scripts\python.exe .\save_json_log.py --json-file .\outputs\trail6.json
```

## JSON Log Format

Example structure:

```json
{
  "filename": "image1.jpg",
  "detections": [
    {
      "person_id": 1,
      "label": "gloved_hand",
      "confidence": 0.92,
      "bbox": [268, 99, 350, 256]
    }
  ],
  "persons": [
    {
      "person_id": 1,
      "person_bbox": [200, 40, 420, 420],
      "status": "gloved_hand",
      "detections": [
        {
          "label": "gloved_hand",
          "confidence": 0.92,
          "bbox": [268, 99, 350, 256]
        }
      ]
    }
  ]
}
```

## Notes

- `pose_glove_detector.py` is the main script for the end-to-end workflow.
- `save_json_log.py` keeps database persistence separate from computer vision inference.
- SQLite was chosen to keep setup simple and avoid extra dependencies.
- If you later add PostgreSQL or MySQL support, that should be implemented inside `save_json_log.py` without changing the detector pipeline.
