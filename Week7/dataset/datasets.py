"""
File containing the function to load all the frame datasets.
"""

#Standard imports
import os

#Local imports
from util.dataset import load_classes
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset

#Constants
DEFAULT_STRIDE = 2      # Sampling stride (if greater than 1, frames are skipped) / Effectively reduces FPS
DEFAULT_OVERLAP = 0.9   # Temporal overlap between sampled clips (for traiing and validation only)

def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_len = args.epoch_num_frames // args.clip_len
    stride = args.stride if "stride" in args else DEFAULT_STRIDE
    overlap = args.overlap if "overlap" in args else DEFAULT_OVERLAP

    # W7-3A: Temporal Gaussian Label Smoothing
    tgls_enabled = getattr(args, 'tgls_enabled', False)
    tgls_sigma   = getattr(args, 'tgls_sigma', 0.55)
    tgls_window  = getattr(args, 'tgls_window', 5)

    dataset_kwargs = {
        'stride': stride, 'overlap': overlap, 'dataset': args.dataset,
        'labels_dir': args.labels_dir, 'task': args.task,
        'tgls_enabled': tgls_enabled, 'tgls_sigma': tgls_sigma,
        'tgls_window': tgls_window,
        'num_classes': args.num_classes,
    }

    print('Dataset size:', dataset_len)

    train_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, dataset_len, **dataset_kwargs)
    train_data.print_info()

    val_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, dataset_len // 4, **dataset_kwargs)
    val_data.print_info()     

    # Test set: no TGLS (labels are only needed for evaluation targets, not loss)
    test_kwargs = {
        'stride': stride, 'overlap': 0, 'dataset': args.dataset,
        'labels_dir': args.labels_dir, 'task': args.task,
    }
    test_data = ActionSpotVideoDataset(classes, os.path.join('data', args.dataset, 'test.json'),
        args.frame_dir, args.clip_len, **test_kwargs)
    test_data.print_info()
        
    return classes, train_data, val_data, test_data