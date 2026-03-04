# Week 2: Object Detection & Multi-Object Tracking

**Master in Computer Vision (MCV) · C6 Module**

This project covers two main tasks on the **AICity dataset (S03/C010)**:

- **Task 1** : Object detection: off-the-shelf evaluation, fine-tuning, and k-fold cross-validation
- **Task 2** : Multi-object tracking: IoU-based tracker, Kalman filter (SORT), and tracking evaluation

---

## Dataset

The code expects the **AICity S03/C010** sequence:

```
data/
└── AICity_data/
    └── train/
        └── S03/
            └── c010/
                ├── vdo.avi           # Raw video
                ├── annotations.xml   # CVAT-format ground truth
                ├── gt/gt.txt         # MOT-format ground truth (fallback)
                └── det/              # Pre-computed detections
                    ├── det_mask_rcnn.txt
                    ├── det_ssd512.txt
                    └── det_yolo3.txt
```

Set the dataset path via the `DATA_ROOT` environment variable.

---

## Install dependencies:

```bash
pip install torch torchvision ultralytics opencv-python numpy pandas scipy matplotlib tqdm
pip install trackeval   # optional but recommended
```

---

## Running the Code

### Environment variable

```bash
export DATA_ROOT=/path/to/AICity_data/train/S03/c010
```

### Run everything

```bash
DATA_ROOT=/path/to/S03/c010 bash run_all.sh
```

### Run individual tasks

```bash
# Task 1.1 — off-the-shelf (YOLOv8m + YOLOv8x + pre-computed)
python task1/task1_1_off_the_shelf.py

# Task 1.1 — all YOLO sizes + YOLOv9 + Faster R-CNN
python task1/task1_1_off_the_shelf.py --yolo_sizes n s m l x --yolov9 --faster_rcnn

# Task 1.1 — pre-computed detectors only (no GPU required)
python task1/task1_1_off_the_shelf.py --precomputed_only

# Task 1.2 — fine-tune
python task1/task1_2_finetune.py

# Task 1.3 — k-fold
python task1/task1_3_kfold.py

# Task 2.1 — IoU tracker
python task2/task2_1_overlap_tracker.py

# Task 2.2 — Kalman / SORT tracker
python task2/task2_2_kalman_tracker.py

# Task 2.3 — evaluation
python task2/task2_3_evaluate_tracking.py
```

---

## Outputs

| Directory | Contents |
|-----------|----------|
| `results/task1_1/` | Detection files, mAP metrics, `best_model.json` |
| `results/task1_2/` | Fine-tuned detection files, ablation CSV, `best_config.json` |
| `results/task1_3/` | K-fold metrics CSV |
| `results/task2_1/` | IoU tracker output files and best configs |
| `results/task2_2/` | SORT tracker output files and best configs |
| `results/task2_3/` | IDF1/HOTA evaluation summaries |
| `plots/` | All metric comparison and ablation plots |
| `qualitative/` | Example visualisation images |

---