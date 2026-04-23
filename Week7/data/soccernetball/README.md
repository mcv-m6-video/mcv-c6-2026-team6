# Setting up SoccerNet Ball Action Spotting

This directory contains the dataset splits converted from the original **SoccerNet Ball Action Spotting** dataset, available at:  
👉 [https://www.soccer-net.org/tasks/ball-action-spotting](https://www.soccer-net.org/tasks/ball-action-spotting)

To download the videos, follow the instructions provided in the [SoccerNet Hugging Face repository](https://huggingface.co/datasets/SoccerNet/SN-BAS-2025).

Once downloaded, you can extract frames and generate the desired folder structure using the provided script: **`extract_frames_snb.py`**. Run the script with the following command:


```
python extract_frames_snb.py --video_dir <video_dir>
        --out_dir <out_dir>
        --width 398 --height 224
        --sample_fps 25 --num_workers 5
```

Replace `<video_dir>` with the path to the directory containing the downloaded and unzipped videos, and `<out_dir>` with the path where you want the extracted frames to be saved. We recommend placing `<out_dir>` outside `<video_dir>`.

By default, frames will be extracted at a resolution of **398x224**, and the folder and frame naming convention will follow this structure:


```
<out_dir>
└───398x224
    └───england_efl
        └───2019-2020
            └───2019-10-01 - Blackburn Rovers - Nottingham Forest
                └───frame0.jpg
                └───frame1.jpg
                └───...
            └───2019-10-01 - Brentford - Bristol City
                └───frame0.jpg
                └───frame1.jpg
                └───...
```

---
