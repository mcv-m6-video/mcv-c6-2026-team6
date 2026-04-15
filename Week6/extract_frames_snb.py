import os
import argparse
import cv2
import moviepy.editor
from tqdm import tqdm
from multiprocessing import Pool
cv2.setNumThreads(0)
from util.dataset import read_fps

'''
This script extracts frames from SoccerNetv2 Ball Action Spotting dataset by introducing the path where the downloaded videos are (at 720p resolution), the path to
write the frames, the sample fps, and the number of workers to use. The script will create a folder for each video in the out_dir path and save the frames as .jpg files in
the desired resolution.

python extract_frames_snb.py --video_dir video_dir
        --out_dir out_dir
        --sample_fps 25 --num_workers 5
'''

# Constants
FRAME_RETRY_THRESHOLD = 1000
DEFAULT_SAMPLE_FPS = 25
DEFAULT_HEIGHT = 224
DEFAULT_WIDTH = 398


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='Path to the downloaded videos')
    parser.add_argument('-o', '--out_dir',
                        help='Path to write frames. Dry run if None.')
    parser.add_argument('--sample_fps', type=int, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT)
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH)
    parser.add_argument('--recalc_fps', action='store_true') # Debug option
    parser.add_argument('-j', '--num_workers', type=int,
                        default=os.cpu_count() // 4)

    return parser.parse_args()

def get_duration(video_path):
    # Copied from SoccerNet repo
    return moviepy.editor.VideoFileClip(video_path).duration


def worker(args):
    video_name, video_path, out_dir, width, height, sample_fps, recalc_fps = args

    def get_stride(src_fps):
        if sample_fps <= 0:
            stride = 1
        else:
            stride = int(src_fps / sample_fps)
        return stride

    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)

    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    oh = height
    ow = width

    time_in_s = get_duration(video_path)

    fps_path = None
    if out_dir is not None:
        fps_path = os.path.join(out_dir, 'fps.txt')
        if os.path.exists(fps_path):
            print('Already done:', video_name)
            vc.release()
            return

        os.makedirs(out_dir, exist_ok=True)

    not_done = True
    while not_done:
        stride = get_stride(fps)
        est_out_fps = fps / stride
        print('{} -- effective fps: {} (stride: {})'.format(
            video_name, est_out_fps, stride))

        i = 0
        while True:
            ret, frame = vc.read()
            if not ret:
                # fps and num_frames are wrong
                if i != num_frames:
                    print('Failed to decode: {} -- {} / {}'.format(
                        video_path, i, num_frames))

                    if i + FRAME_RETRY_THRESHOLD < num_frames:
                        num_frames = i
                        adj_fps = num_frames / time_in_s
                        if get_stride(adj_fps) == stride:
                            # Stride would not change so nothing to do
                            not_done = False
                        else:
                            print('Retrying:', video_path)
                            # Stride changes, due to large error in fps.
                            # Use adjusted fps instead.
                            vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            fps = adj_fps
                    else:
                        not_done = False
                else:
                    not_done = False
                break

            if i % stride == 0:
                if not recalc_fps:
                    if frame.shape[0] != oh or frame.shape[1] != ow:
                        frame = cv2.resize(frame, (ow, oh))
                    if out_dir is not None:
                        frame_path = os.path.join(out_dir, 'frame{}.jpg'.format(i))
                        cv2.imwrite(frame_path, frame)
            i += 1
    vc.release()

    out_fps = fps / get_stride(fps)
    if fps_path is not None:
        with open(fps_path, 'w') as fp:
            fp.write(str(out_fps))
    print('{} - done'.format(video_name))


def main(args):
    video_dir = args.video_dir
    out_dir = args.out_dir
    width = args.width
    height = args.height
    sample_fps = args.sample_fps
    recalc_fps = args.recalc_fps
    num_workers = args.num_workers

    # global RECALC_FPS_ONLY
    # RECALC_FPS_ONLY = recalc_fps

    worker_args = []
    for league in os.listdir(video_dir):
        league_dir = os.path.join(video_dir, league)
        if os.path.isfile(league_dir) or league == "ExtraLabelsActionSpotting500games":
            continue
        for season in os.listdir(league_dir):
            season_dir = os.path.join(league_dir, season)
            if os.path.isfile(season_dir):
                continue
            for game in os.listdir(season_dir):
                game_dir = os.path.join(season_dir, game)
                if os.path.isfile(game_dir):
                    continue
                for video_file in os.listdir(game_dir):
                    if (video_file.endswith('720p.mp4') | video_file.endswith('720p.mkv')): # Only 720p videos
                        worker_args.append((
                            os.path.join(league, season, game, video_file),
                            os.path.join(game_dir, video_file),
                            os.path.join(out_dir, league, season, game) if out_dir else None,
                            width,
                            height,
                            sample_fps,
                            recalc_fps
                        ))

    with Pool(num_workers) as p:
        for _ in tqdm(p.imap_unordered(worker, worker_args), total=len(worker_args)):
            pass
    print('Done!')


if __name__ == '__main__':
    args = get_args()
    args.out_dir = os.path.join(args.out_dir, f"{args.width}x{args.height}")
    main(args)