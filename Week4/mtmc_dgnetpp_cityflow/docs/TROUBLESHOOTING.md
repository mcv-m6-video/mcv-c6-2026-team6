# Troubleshooting

## Common Setup Issues

### `ModuleNotFoundError: No module named 'mtmc'`

The repository entry scripts under `tools/` add `src/` to `sys.path` automatically. If this still appears, run commands from the repository root:

```bash
cd mtmc_dgnetpp_cityflow
python tools/prepare_reid_crops.py --help
```

### Dataset path typo or missing sequence

Use the dataset root:

```bash
../AI_CITY_CHALLENGE_2022_TRAIN
```

The loader now reports clearer errors for missing roots and unknown sequences.

### `pytrec_eval` or `motmetrics` missing

Install all dependencies from the repository root:

```bash
pip install -r requirements.txt
```

### Evaluator tries to download ROI files from Drive

The local evaluator in `../AI_CITY_CHALLENGE_2022_TRAIN/eval/eval.py` was patched to work offline. It now discovers `roi.jpg` files directly from the dataset folders.

### NumPy 2.0 compatibility issue with `np.asfarray`

The evaluator was patched with a compatibility shim for `motmetrics` on NumPy 2.x.

## Split Reminder

This repository assumes the following experimental protocol:

- Train sequences: `S01`, `S04`
- Test sequence: `S03`

Do not evaluate S03 predictions against the full `ground_truth_train.txt`. First create `ground_truth_s03.txt`:

```bash
cd ../AI_CITY_CHALLENGE_2022_TRAIN/eval
awk '$1>=10 && $1<=15' ground_truth_train.txt > ground_truth_s03.txt
```

## Precision Is Very Low

Symptoms:
- Too many predicted rows in `track1.txt`
- High IDR and low IDP

Typical fixes:
1. Use a cleaner MTSC input such as `mtsc_tc_ssd512.txt`.
2. Increase `mtmc.min_sim`.
3. Increase `mtmc.min_global_cameras`.
4. Increase `mtmc.min_global_rows`.
5. Lower `mtmc.max_global_ids`.
6. Use `tools/postfilter_track1.py` for fast iteration without rerunning feature extraction.
