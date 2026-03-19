# MTMC Vehicle Tracking — AI City Challenge Track 1

Multi-Target Multi-Camera (MTMC) vehicle tracking system built on a **Doc2GraphFormer**-style graph neural network. The system links vehicle tracklets across multiple cameras by constructing a graph where nodes represent tracklets and edges represent potential cross-camera matches, then predicts same-vehicle links to produce global vehicle IDs.

## Presentation

- **Slides:** [Click here for the Presentation](https://www.canva.com/design/DAHELjos8YY/vCHw7ATOZ_-x1HKMHFChHA/edit?utm_content=DAHELjos8YY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
- **PDF:** [`Week4_Team6_Final_Presentation.pdf`](Week4_Team6_Final_Presentation.pdf)

## Pipeline

1. **Data Loading** — Load CityFlow sequences (S01, S03, S04) with ground-truth or YOLO tracklets.
2. **YOLO Tracking** — Run per-camera detection and tracking, outputting MOTChallenge-format results.
3. **Tracklet Filtering** — Apply ROI masks, NMS, minimum confidence/area thresholds, and track length filtering.
4. **Feature Extraction** — Extract visual (CLIP/ViT), geometric, spatio-temporal, ROI, and world-coordinate features per tracklet.
5. **Graph Construction** — Build DGL graphs with configurable edge types (`cross_cam`, `fully`, `temporal`).
6. **Training** — Train the MTMCGraphformer with focal loss, camera auxiliary task, and early stopping.
7. **Evaluation** — Evaluate using AI City Challenge metrics (IDF1, etc.).

## Model Architecture

**MTMCGraphformer** uses a GraphFormer backbone with edge-conditioned attention, a camera classifier head, a vehicle linker for same-vehicle edge prediction, and a temporal grouper. Post-processing applies Hungarian matching per camera pair for one-to-one assignment, followed by connected-component extraction for global IDs.

### Feature Modalities

| Modality | Description |
|---|---|
| Visual | CLIP or ViT embeddings (512/768-d), optionally with segmentation masks |
| Geometric | Bounding box size, aspect ratio |
| Spatio-temporal | Timestamp, velocity |
| ROI | Boundary proximity (entry/exit zones) |
| World Coordinates | Homography-based projection from calibration files |

## Project Structure

```
├── camera_sync_offsets.py       # Per-camera timing offsets for temporal alignment
├── cityflow_preprocessing.py    # CityFlow / AIC22 data loading
├── dataset_extras.py            # ROI masks, calibration, MTSC tracklets
├── eval.py                      # AI City Challenge evaluation
├── filter_tracklets.py          # YOLO tracklet filtering (ROI, NMS, etc.)
├── generate_gt_mtmc.py          # Generate gt_mtmc.txt from per-camera GT
├── graphformer.py               # GraphFormer backbone + MTMC heads
├── paths.py                     # Path configuration
├── pretrain_reid.py             # Re-ID contrastive pre-training (CLIP backbone)
├── requirements.txt             # Python dependencies
├── run_yolo_tracking.py         # YOLO tracking on CityFlow cameras
├── train_mtmc.py                # Main training / evaluation entry point
├── training_utils.py            # Loss functions, metrics, early stopping, submission format
├── utils.py                     # NMS, data loaders, misc utilities
├── vehicle_feature_builder.py   # Tracklet feature extraction
└── vehicle_graph_builder.py     # DGL graph construction
```

## Setup

```bash
pip install -r requirements.txt
```

### Data

The project expects the CityFlow / AIC22 Track 1 dataset laid out as:

```
<SEQS_ROOT>/
├── S01/   # Training
├── S03/   # Testing
└── S04/   # Training
```

Update `paths.py` to point `SEQS_ROOT` to your data directory.

## Usage

**Run YOLO tracking on all cameras:**

```bash
python run_yolo_tracking.py
```

**Filter tracklets:**

```bash
python filter_tracklets.py
```

**Pre-train the Re-ID backbone (optional):**

```bash
python pretrain_reid.py
```

**Train and evaluate the MTMC model:**

```bash
python train_mtmc.py
```

**Evaluate:**

```bash
python eval.py
```

## Tech Stack

PyTorch, DGL, Hugging Face Transformers (CLIP / ViT), Ultralytics YOLO, OpenCV, NetworkX, scikit-learn, NumPy, Matplotlib.