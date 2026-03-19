# References

## Core Method Inspiration

1. Xuanmeng Zhang, Weijian Deng, Jiani Hu, Xiang Long, Dong Chen, Fang Wen. DG-Net++: Improved Baseline for Domain Generalization in Person Re-Identification. IEEE CVPR Workshops, 2020.
2. DG-Net-PP repository: https://github.com/NVlabs/DG-Net-PP

## Dataset and Evaluation

1. Zheng Tang, Milind Naphade, Ming-Yu Liu, Xiaodong Yang, Stan Birchfield, Shuo Wang, Ratnesh Kumar, David Anastasiu, Jenq-Neng Hwang. CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification. CVPR, 2019.
2. Milind Naphade et al. The 5th AI City Challenge. CVPR Workshops, 2021.
3. Official CityFlowV2 / AI City Challenge dataset documentation is included in `../AI_CITY_CHALLENGE_2022_TRAIN/ReadMe.txt`.

## Single-Camera Tracking Baselines Present in the Dataset

1. Deep SORT: Nicolai Wojke, Alex Bewley, Dietrich Paulus. Simple Online and Realtime Tracking with a Deep Association Metric. ICIP, 2017.
2. MOANA: Zheng Tang, Jenq-Neng Hwang. MOANA: An Online Learned Adaptive Appearance Model for Robust Multiple Object Tracking in 3D. IEEE Access, 2019.
3. Tracklet Clustering baseline referenced by the dataset documentation.

## Metrics

1. IDF1 is used for ranking in Track 1.
2. The evaluator also reports IDP and IDR, plus other MOTChallenge metrics.
3. MOT metrics background: Anton Milan, Laura Leal-Taixe, Ian Reid, Stefan Roth, Konrad Schindler. MOT16: A Benchmark for Multi-Object Tracking. arXiv, 2016.
