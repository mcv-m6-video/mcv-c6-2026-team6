"""
File containing the function to load all the frame datasets.
"""

#Standard imports
import os

#Local imports
from util.dataset import load_classes
from dataset.frame import ActionSpotDataset

#Constants
DEFAULT_STRIDE = 2      # Sampling stride (if greater than 1, frames are skipped) / Effectively reduces FPS
DEFAULT_OVERLAP = 0.9   # Temporal overlap between sampled clips (for traiing and validation only)

def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_len = args.epoch_num_frames // args.clip_len
    stride = args.stride if hasattr(args, "stride") else DEFAULT_STRIDE
    overlap = args.overlap if hasattr(args, "overlap") else DEFAULT_OVERLAP

    dataset_kwargs = {
        'stride': stride, 'overlap': overlap, 'dataset': args.dataset, 'labels_dir': args.labels_dir, 'task': args.task,
    }
    if hasattr(args, "label_source"):
        dataset_kwargs["label_source"] = args.label_source
    if hasattr(args, "label_map"):
        dataset_kwargs["label_map"] = args.label_map

    print('Dataset size:', dataset_len)

    train_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, dataset_len, **dataset_kwargs)
    train_data.print_info()

    val_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, dataset_len // 4, **dataset_kwargs)
    val_data.print_info()     

    dataset_kwargs['overlap'] = 0

    test_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'test.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, None, pad_len = 0, **dataset_kwargs)
    test_data.print_info()

        
    return classes, train_data, val_data, test_data