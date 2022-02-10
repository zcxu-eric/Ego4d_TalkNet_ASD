from collections import defaultdict
import os, json, csv, sys
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser


def consistent(a, b):
    return abs(a/30-b) < 0.039


csv.field_size_limit(sys.maxsize)


def generate_from_pretained_tracker(split='test'):

    asd_records = []

    with open('data/track_results/v.txt', 'r') as f:
        videos = f.readlines()
    res2video = { i:video.split('/')[-1][:-5] for i, video in enumerate(videos) }
    video2res = { video.split('/')[-1][:-5]:i for i, video in enumerate(videos) }
    with open(f'data/split/{split}.list', 'r') as f:
        videos = [line.strip() for line in f.readlines()]
    tracklets = [f'data/track_results/results/{video2res[v]}.txt' for v in videos]
    for tracklet in tqdm(sorted(tracklets)):
        if not os.path.exists(tracklet):
            print(tracklet)
            continue
        global_tracks = defaultdict(list)
        with open(tracklet, 'r') as f:
            res = f.readlines()
        for record in res:
            frame, pid, x1, y1, x2, y2 = record.split()
            global_tracks[pid].append({
                'frame': int(frame), 
                'x1': int(x1), 
                'y1': int(y1), 
                'x2': int(x2), 
                'y2': int(y2),
                'pid': int(pid), 
                'video': res2video[int(tracklet.split('/')[-1][:-4])]
            })
        
        local_tracks = defaultdict(list)
        for pid, frames in global_tracks.items():
            count = -1
            last_frame = -2
            track_length = 0
            frames.sort(key=lambda x:x['frame'])
            for f in frames:
                if (f['frame'] > last_frame + 1) or (track_length > 300):
                    count += 1
                    track_length = 0
                last_frame = f['frame']
                video = f['video']
                trackid = f'{video}:{pid}:{count}'
                f.pop('video')
                local_tracks[trackid].append(f)
                track_length += 1

        for track_id, frames in local_tracks.items():
            with open(f'data/infer/bbox/{track_id}.json', 'w+') as f:
                json.dump(frames, f)
            record = []
            # [trackid (video+trackid),  length of tracklets,  fps,  labels,  frame]
            record.append([track_id, len(frames), 30.0, [0], frames[0]['frame']])
            asd_records.extend(record)

    asd_records = pd.DataFrame(asd_records)
    asd_records.to_csv(f'data/infer/csv/active_speaker_{split}.csv', header=None, index=False, sep='\t')


def load_label(data):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    return res


if __name__ == '__main__':

    os.makedirs('data/infer', exist_ok=True)
    os.makedirs('data/infer/csv', exist_ok=True)
    os.makedirs('data/infer/bbox', exist_ok=True)

    parser = ArgumentParser()
    parser.add_argument('--evalDataType', type=str, default="test", help='Choose the dataset for evaluation, val or test')
    args = parser.parse_args()
    generate_from_pretained_tracker(split=args.evalDataType)
