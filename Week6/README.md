# Master in Computer Vision (Barcelona) 2025/26
# Project 2 (Task 2) @ C6 - Video Analysis

This repository provides the starter code for Task 2 of Project 2: Action spotting on the SoccerNet Ball Action Spotting 2025 (SN-BAS-2025) dataset.

The installation of dependencies, how to obtain the dataset, and instructions on running the spotting baseline are detailed next.

## Dependencies

You can install the required packages for the project using the following command, with `requirements.txt` specifying the versions of the various packages:

```
pip install -r requirements.txt
```

## Getting the dataset and data preparation

Refer to the README files in the [data/soccernetball](/data/soccernetball) directory for instructions on how to download the SNABS2025 dataset, preparation of directories, and extraction of the video frames.


## Running the baseline for Task 2

The `main_spotting.py` is designed to train and evaluate the baseline using the settings specified in a configuration file. You can run `main_spotting.py` using the following command:

```
python3 main_spotting.py --model <model_name>
```

Here, `<model_name>` can be chosen freely but must match the name of a configuration file (e.g. `baseline.json`) located in the config directory [config](/config/). For example, to chose the baseline model, you would run: `python3 main_spotting.py --model baseline`.

For additional details on configuration options using the configuration file, refer to the README in the [config](/config/) directory.

## Important notes

- Before running the model, ensure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant [config](/config/) files.
- Make sure to run the `main_spotting.py` with the `mode` parameter set to `store` at least once to generate the clips and save them. After this initial run, you can set the `mode` to `load` to reuse the same clips in subsequent executions.

## Support

For any issues related to the code, please email [aclapes@ub.edu](mailto:aclapes@ub.edu) and CC [arturxe@gmail.com](mailto:arturxe@gmail.com).

---

## Best Model

The best model from our ablation study is a **BiGRU** temporal head on top of the RegNet-Y (200MF) backbone.

| Resource | Link |
|---|---|
| Config (`.json`) | [Download from Google Drive](https://drive.google.com/file/d/1CFfQsDhea_EOQesX3AxOBZPI_N-secT5/view?usp=share_link) |
| Checkpoint (`.pt`) | [Download from Google Drive](https://drive.google.com/file/d/1KhFG1xkFEhU9JlNSdJKVozH-z2iLj2oz/view?usp=share_link) |
