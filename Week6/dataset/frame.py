#!/usr/bin/env python3

"""
File containing classes related to the frame datasets.
"""

#Standard imports
from util.io import load_json
import os
import random
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import pickle
import math

#Constants

# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5
FPS_SN = 25


class ActionSpotDataset(Dataset):

    def __init__(
            self,
            classes,                    # dict of class names to idx
            game_file,                 # path to label json
            frame_dir,                  # path to frames
            store_dir,                  # path to store files (with frames path and labels per clip)
            store_mode,                 # 'store' or 'load'
            clip_len,                   # Number of frames per clip
            dataset_len,                # Number of clips
            stride=1,                   # Downsample frame rate
            overlap=1,                  # Overlap between clips (in proportion to clip_len)
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
                                        # and end of videos
            dataset = 'soccernetball',     # Dataset name
            labels_dir = None,          # Directory with labels for SoccerNetBall
            task = 'classification'     # Classification or localization
    ):
        self._src_file = game_file
        self._games = load_json(game_file)
        self._split = game_file.split('/')[-1].split('.')[0]
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._games)}
        self._dataset = dataset
        assert dataset == 'soccernetball'
        self._store_dir = store_dir
        self._store_mode = store_mode
        assert store_mode in ['store', 'load']
        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        assert overlap >= 0 and overlap <= 1
        self._clip_sampling_step = 1 if overlap == 1 else int((1 - overlap) * clip_len * stride)
        self._pad_len = pad_len
        assert pad_len >= 0     
        self._labels_dir = labels_dir
        self._task = task
        assert task in ['classification', 'spotting']

        #Frame reader class
        self._frame_reader = FrameReader(frame_dir, dataset = dataset)

        #Store or load clips
        if self._store_mode == 'store':
            self._store_clips()
        elif self._store_mode == 'load':
            self._load_clips()

        if dataset_len is None:
            self._dataset_len = len(self._frame_paths)
        else:
            self._dataset_len = dataset_len

        self._total_len = len(self._frame_paths)

    def _store_clips(self):
        #Initialize frame paths list
        self._frame_paths = []
        self._labels_store = []

        for video in tqdm(self._games):
            video_len = int(video['num_frames'])

            #Load labels
            video_half = 1
            labels_file = load_json(os.path.join(self._labels_dir, video['video'] + '/Labels-ball.json'))['annotations']

            for base_idx in range(-self._pad_len * self._stride, max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride), self._clip_sampling_step):

                frames_paths = self._frame_reader.load_paths(video['video'], base_idx, base_idx + self._clip_len * self._stride, stride=self._stride)

                labels = []

                for event in labels_file:
                    event_half = int(event['gameTime'][0])
                    if event_half == video_half:
                        event_frame = int(int(event['position']) / 1000 * FPS_SN) #miliseconds to frames
                        label_idx = (event_frame - base_idx) // self._stride #position of event in clip

                        if (label_idx >= 0 and label_idx < self._clip_len):
                            label = self._class_dict[event['label']]
                            labels.append({'label': label, 'label_idx': label_idx}) #add label and position to list (for both classification and location)

                if frames_paths[1] != -1: #in case no frames were available

                    self._frame_paths.append(frames_paths)
                    self._labels_store.append(labels)

        #Save to store
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'SPLIT' + self._split) #store clips information of dataset with LEN and SPLIT information

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        with open(store_path + '/frame_paths.pkl', 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(store_path + '/labels.pkl', 'wb') as f:
            pickle.dump(self._labels_store, f)
        print('Stored clips to ' + store_path)
        return
    
    def _load_clips(self):
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'SPLIT' + self._split)
        
        with open(store_path + '/frame_paths.pkl', 'rb') as f:
            self._frame_paths = pickle.load(f)
        with open(store_path + '/labels.pkl', 'rb') as f:
            self._labels_store = pickle.load(f)
        print('Loaded clips from ' + store_path)
        return

    def _get_one(self):
        #Get random index
        idx = random.randint(0, self._total_len - 1)

        #Get frame_path and labels dict
        frames_path = self._frame_paths[idx]
        dict_label = self._labels_store[idx]      

        #Load frames
        frames = self._frame_reader.load_frames(frames_path, pad=True, stride=self._stride)

        #Process labels
        if self._task == 'spotting':
            labels = np.zeros(self._clip_len, np.int64)
            for label in dict_label:
                labels[label['label_idx']] = label['label']

        elif self._task == 'classification':
            labels = np.zeros(len(self._class_dict), np.int64) #C classes
            for label in dict_label:
                labels[label['label']-1] = 1 # labels start at 1

        return {'frame': frames, 'contains_event': int(np.sum(labels) > 0),
                'label': labels}

    def __getitem__(self, unused):
        ret = self._get_one()

        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._games)



class FrameReader:

    def __init__(self, frame_dir, dataset):
        self._frame_dir = frame_dir
        self.dataset = dataset

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path)
        return img
    
    def load_paths(self, video_name, start, end, stride=1):

        path = os.path.join(self._frame_dir, video_name)

        found_start = -1
        pad_start = 0
        pad_end = 0
        for frame_num in range(start, end, stride):

            if frame_num < 0:
                pad_start += 1
                continue

            if pad_end > 0:
                pad_end += 1
                continue

            frame = frame_num
            frame_path = os.path.join(path, 'frame' + str(frame) + '.jpg')
            base_path = path
            ndigits = -1
            
            exist_frame = os.path.exists(frame_path)
            if exist_frame & (found_start == -1):
                found_start = frame

            if not exist_frame:
                pad_end += 1

        ret = [base_path, found_start, pad_start, pad_end, ndigits, (end-start) // stride]

        return ret
    
    def load_frames(self, paths, pad=False, stride=1):
        base_path = paths[0]
        start = paths[1]
        pad_start = paths[2]
        pad_end = paths[3]
        ndigits = paths[4]
        length = paths[5]

        ret = []
        if ndigits == -1:
            path = os.path.join(base_path, 'frame')
            _ = [ret.append(self.read_frame(path + str(start + j * stride) + '.jpg')) for j in range(length - pad_start - pad_end)]

        else:
            path = base_path + '/'
            _ = [ret.append(self.read_frame(path + str(start + j * stride).zfill(ndigits) + '.jpg')) for j in range(length - pad_start - pad_end)]

        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if pad_start > 0 or (pad and pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, pad_start, pad_end if pad else 0))            

        return ret
    
    def load_frames_test(self, video_name, start, end, pad=False, stride=1):
        ret = []
        n_pad_start = 0
        n_pad_end = 0

        for frame_num in range(start, end, stride):

            if frame_num < 0:
                n_pad_start += 1
                continue

            frame_path = os.path.join(self._frame_dir, video_name, 'frame' + str(frame_num) + '.jpg')
                
            try:
                img = self.read_frame(frame_path)
                ret.append(img)
            except RuntimeError:
                # print('Missing file!', frame_path)
                n_pad_end += 1

        if len(ret) == 0:
            return -1 # Return -1 if no frames were loaded
        
        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))
        return ret

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._games])
        print('{} : {} videos, {} frames ({} stride)'.format(
            self._src_file, len(self._games), num_frames, self._stride)
        )


def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames'] for x in labels])
        print('{} : {} videos, {} frames'.format(
            src_file, len(labels), num_frames))

class ActionSpotVideoDataset(Dataset):

    def __init__(
            self,
            classes,
            game_file,
            frame_dir,
            clip_len,
            overlap=0,
            stride=1,
            pad_len=DEFAULT_PAD_LEN,
            dataset = 'soccernetball',
            labels_dir = None,
            task = 'spotting'
    ):
        self._src_file = game_file
        self._games = load_json(game_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._games)}
        self._dataset = dataset
        assert dataset == 'soccernetball'
        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        assert overlap >= 0 and overlap <= 1
        self._clip_sampling_step = 1 if overlap == 1 else int((1 - overlap) * clip_len * stride)
        self._pad_len = pad_len
        assert pad_len >= 0
        self._labels_dir = labels_dir
        self._task = task
        assert task == 'spotting'

        self._frame_reader = FrameReader(frame_dir, dataset = dataset)

        self._clips = []
        for l in self._games:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, int(l['num_frames'] - (self._clip_sampling_step * stride))), \
                # Need to ensure that all clips have at least one frame
                self._clip_sampling_step
            ):
                has_clip = True
                self._clips.append((l['video'], i))
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):

        video_name, start = self._clips[idx]

        frames = self._frame_reader.load_frames_test(
            video_name, start, start + self._clip_len * self._stride, pad=True,
            stride=self._stride)

        return {'video': video_name, 'start': start // self._stride,
                'frame': frames}
    
    @property
    def videos(self):
        return sorted([
            (v['video'], math.ceil(v['num_frames'] / self._stride),
                FPS_SN / self._stride) for v in self._games])

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._games])

        print('{} : {} videos, {} frames ({} stride)'.format(
            self._src_file, len(self._games), num_frames, self._stride)
        )