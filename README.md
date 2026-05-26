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
- One JSON log per image in `outputs/json_logs/`
- Batch JSON output in `outputs/json_logs/detections_all.json`
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
- `outputs/`: annotated images
- `outputs/json_logs/`: per-image JSON logs and batch JSON output

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

1. saves the JSON file to `outputs/json_logs/`
2. checks database configuration
3. inserts the same JSON record into the database if enabled

### Manual save with the separate script

You can save an existing JSON log manually:

```powershell
venv\Scripts\python.exe .\save_json_log.py --json-file .\outputs\json_logs\trail6.json
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


## Part 2 — Assessment Questions

### Q1: Choosing the Right Approach

For this problem, I would start with object detection to locate each product on the assembly line, then pass the cropped region to a binary classification model that answers one simple question: label present or not. This two-stage approach keeps each model focused on a narrow task, which tends to perform better than asking one model to do everything at once. Detection handles the "where is the product" problem, and classification handles the "what is its state" problem — clean separation of concerns. If this pipeline underperforms, my fallback would be to try an anomaly detection approach: train only on labeled products as the "normal" baseline, so anything that looks different (i.e., missing a label) gets flagged as an anomaly. This works well when defective samples are rare and hard to collect in large numbers.

---

### Q2: Debugging a Poorly Performing Model

The first thing I would check is whether the training images and the new factory images actually look the same — differences in lighting, camera angle, or image resolution are the most common culprits and are easy to miss. I would plot a sample of training images alongside the failing test images side by side to spot any obvious visual shift. Next, I would look at the confusion matrix to understand how the model is failing — is it always predicting one class, or is it randomly wrong? I would also check if the 1000 training images were diverse enough, or if they all came from the same shift, same camera, or same lighting condition, which would mean the model never learned to generalize. Finally, I would run the model on a small hand-labeled set of the new factory images and measure precision and recall separately to get a clearer picture than overall accuracy alone.

---

### Q3: Accuracy vs Real Risk

Accuracy is the wrong metric here, and the numbers prove it — missing 1 in 10 defective products sounds small until you realize that in a manufacturing line running thousands of units, that translates to a serious volume of defects reaching customers. The metric I would focus on instead is recall, which measures how many actual defects the model correctly catches out of all real defects. In quality control, a false negative — letting a bad product through — is far more costly than a false positive — stopping a good product for re-inspection. I would also look at the F-beta score with beta greater than 1, which explicitly weights recall more heavily than precision to reflect this asymmetry. The business cost of a recall or a customer complaint almost always outweighs the cost of a brief line stoppage, so the metric needs to reflect that priority.

---

### Q4: Annotation Edge Cases

I would remove heavily blurry images from the training set but keep partially visible ones with careful handling. Blurry images are genuinely ambiguous — even a skilled human annotator cannot reliably say whether a label is present or not, so feeding that noise to a model only teaches it to be uncertain. Partially visible objects are a different story: they actually reflect real factory conditions where a product might be at the edge of the frame or mid-transition on the belt, so training on them improves robustness. The trade-off is annotation cost and consistency — partial objects take more time to label and different annotators may label them differently, which introduces noise of its own. My approach would be to set a clear rule, such as "include if at least 60% of the product is visible and the label region is discernible," so the dataset stays both realistic and reliably annotated.
