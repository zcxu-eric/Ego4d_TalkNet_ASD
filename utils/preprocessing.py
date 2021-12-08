from collections import defaultdict
import os, json
from tqdm import tqdm
import pandas as pd


def generate(split='train'):
    with open('data/json/av.json', 'r') as f:
        ori_annot = json.load(f)
    annotation = { c_annot['clip_uid']: c_annot for v_annot in ori_annot['videos'] for c_annot in v_annot['clips'] }

    with open(f'data/split/{split}.list', 'r') as f:
        videos = [line.strip() for line in f.readlines()]

    asd_records = []
    for video in tqdm(videos):
        active_speaker = defaultdict(list)
        tracks = defaultdict(list)
        for person in annotation[video]['persons']:

            if person['person_id'] == 'camera_wearer':
                continue

            segments = person['voice_segments']
            for segments in segments:
                active_speaker[person['person_id']].extend(list(range(segments['start_frame'], segments['end_frame']+1)))

            bboxes = person['tracking_paths']
            for bbox in bboxes:
                if 'visual_anchor' in bbox['track_id']:
                    continue
                tracks[bbox['track_id']] = { 'person': person['person_id'], 'frames': bbox['track'] }
                    
        for track in tracks:
            label = []
            bbox = {}
            record = []
            frames = tracks[track]['frames']

            for frame in frames:
                label.append(int(frame['frame'] in active_speaker[tracks[track]['person']]))
                bbox[int(frame['frame'])] = frame
            step = list(range(0, (frames[-1]['frame']-frames[0]['frame']+1), 300))
            track_id = video+':'+track
            for i, start in enumerate(step):
                if len(frames[start:start+300]) > 1:
                    record.append([track_id+':'+str(i), len(frames[start:start+300]), 30.0, label[start:start+300], frames[start]['frame']])
                with open(f'data/ego4d/bbox/{track_id}:{i}.json', 'w+') as f:
                    json.dump(frames[start:start+300], f)
            asd_records.extend(record)

    asd_records = pd.DataFrame(asd_records)
    asd_records.to_csv(f'data/ego4d/csv/active_speaker_{split}.csv', header=None, index=False, sep='\t')


if __name__ == '__main__':

    os.makedirs('data/ego4d', exist_ok=True)
    os.makedirs('data/ego4d/bbox', exist_ok=True)
    os.makedirs('data/ego4d/csv', exist_ok=True)

    generate(split='train')
    generate(split='val')
    generate(split='test')