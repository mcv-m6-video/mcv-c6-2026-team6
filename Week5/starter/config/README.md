# Setting up the Configurations

Here we describe the different parameters set in the baseline configuration file:

- _frame_dir:_ Directory where frames are stored.
- _save_dir:_ Directory to save checkpoints, dataset information, etc.
- _labels_dir:_ Directory where dataset labels are stored.
- _label_source:_ Which label file to use per game folder: `"ball"` (v1 `Labels-ball.json`), `"v2"` (`Labels-v2.json`), or `"both"` (merge events from both files). Default: `"ball"`.
- _label_map:_ Optional dict to remap raw label strings into your `class.txt` label set. Example: `{ "Kick-off": "Kick-off" }`. Unmapped/unknown labels are ignored.
- _store_mode:_ 'store' if it's the first time running the script to prepare and store dataset information, or 'load' to load previously stored information.
- _task_: either 'classification' or 'spotting'
- _batch_size:_ Batch size.
- _clip_len:_ Length of the clips in number of frames.
- _stride:_ Sampling one out of every _stride_ frames when reading from _frame_dir_.
- _dataset:_ Name of the dataset ('soccernetball').
- _epoch_num_frames:_ Number of frames used per epoch.
- _feature_arch:_ Feature extractor architecture (e.g. 'rny002_gsf', 'rny004', or 'rny008_gsf' from `timm` library). Check `model/model_classification.py` to see accepted models at this points. Of course, you can choose to change them to other models or use your own.
- _learning_rate:_ Learning rate.
- _num_classes:_ Number of classes for the current dataset.
- _num_epochs:_ Number of epochs for training.
- _warm_up_epochs:_ Number of warm-up epochs.
- _only_test:_ Boolean indicating whether only inference or training + inference.
- _device:_ Either "cuda" or "cpu".
- _num_workers:_ Number of workers.

You are free to create new configurations and add the necessary parameters once you modify the baseline. At the very least, you'll need to modify `frame_dir`, `save_dir`, and `labels_dir` as they are set to work in our own computation servers. 
