"""
File containing main evaluation functions
"""

#Standard imports
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import json
import os
from SoccerNet.Evaluation.ActionSpotting import average_mAP

#Local imports
from dataset.frame import FPS_SN

#Constants
INFERENCE_BATCH_SIZE = 4

def evaluate(model, dataset, batch_size=INFERENCE_BATCH_SIZE, nms_window = 5):
    
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(dataset._class_dict)), np.float32), #scores matrix TxC (T with used stride)
            np.zeros(video_len, np.int32)) #support matrix T

    for clip in tqdm(DataLoader(
            dataset, num_workers=batch_size * 2, pin_memory=True,
            batch_size=batch_size
    )):
        # Batched by dataloader
        batch_pred_scores = model.predict(clip['frame']) # remove background class

        for i in range(clip['frame'].shape[0]):
            video = clip['video'][i]
            scores, support = pred_dict[video]
            pred_scores = batch_pred_scores[i]
            start = clip['start'][i].item()
            if start < 0:
                pred_scores = pred_scores[-start:, :]
                start = 0
            end = start + pred_scores.shape[0]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:end - start, :]

            scores[start:end, :] += pred_scores[:, 1:] # remove background class
            support[start:end] += (pred_scores.sum(axis=1) != 0) * 1

    # For mAP evalaution
    detections_numpy = list()

    # Get detections_numpy (from predictions and applying NMS)
    for video, video_len, _ in dataset.videos:
        scores, support = pred_dict[video]
        support[support == 0] = 1
        scores = scores / support[:, np.newaxis] # mean over support predictions
        pred = apply_NMS(scores, nms_window, 0.05) # apply NMS
        detections_numpy.append(pred)

    targets_numpy = list()
    closests_numpy = list()
    # Get targets_numpy and closests_numpy (from ground truth)
    for video, video_len, _ in dataset.videos:
        targets = np.zeros((video_len, len(dataset._class_dict)), np.float32)
        labels = json.load(open(os.path.join(dataset._labels_dir, video, 'Labels-ball.json')))

        for annotation in labels["annotations"]:

            event = dataset._class_dict[annotation["label"]]
            frame = int(FPS_SN / dataset._stride * ( int(annotation["position"])/1000 )) #with the current framerate

            frame = min(frame, video_len-1)
            targets[frame, event-1] = 1

        targets_numpy.append(targets)

        closest_numpy = np.zeros(targets.shape) - 1
        # Get the closest action index
        for c in np.arange(targets.shape[-1]):
            indexes = np.where(targets[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = targets[indexes[i], c]
        closests_numpy.append(closest_numpy)

    # Compute the performances
    mAP, AP_per_class, _, _, _, _ = (
        average_mAP(targets_numpy, detections_numpy, closests_numpy, FPS_SN / dataset._stride, deltas=np.array([1]))
    )

    return mAP, AP_per_class
    


def apply_NMS(predictions, window, thresh=0.0):

    nf, nc = predictions.shape
    for i in range(nc):
        aux = predictions[:,i]
        aux2 = np.zeros(nf) -1
        while(np.max(aux) >= thresh):
            # Get the max remaining index and value
            max_value = np.max(aux)
            max_index = np.argmax(aux)
            # detections_NMS[max_index,i] = max_value

            nms_from = int(np.maximum(-(window/2)+max_index,0))
            nms_to = int(np.minimum(max_index+int(window/2), len(aux)))

            aux[nms_from:nms_to] = -1
            aux2[max_index] = max_value
        predictions[:,i] = aux2

    return predictions
