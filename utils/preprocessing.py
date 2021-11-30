from collections import defaultdict
import os, json, csv, sys
from tqdm import tqdm
import pandas as pd


def consistent(a, b):
    return abs(a/30-b) < 0.039


csv.field_size_limit(sys.maxsize)


def generate(split='train'):
    annotation = pd.read_csv('data/csv/AV_step123_v1_batch1_export.csv')

    with open(f'data/split/{split}.list', 'r') as f:
        videos = [line.strip() for line in f.readlines()]

    asd_records = []
    for video in tqdm(videos):
        active_speaker = defaultdict(list)
        tracks = defaultdict(list)
        vad = annotation[annotation['clip_uid']==video]["voice_activity"]
        faces = annotation[annotation['clip_uid']==video]["bbox"]
        try:
            vad = json.loads(vad.iloc[0])[0]
            faces = json.loads(faces.iloc[0])[0]
        except:
            continue
        for v in vad['payload']:
            if 'label' in v:
                spkid = v['label'].strip(':')
                active_speaker[spkid].extend(list(range(v['start_frame'], v['end_frame'])))
        for f in faces['payload']:
            if 'track_id' in f:
                spkid = f['track_id'].strip(':')
                if 'dict_attributes' in f:
                    f['Person ID'] = f['dict_attributes']['Person ID'].split('_')[-1]
                    tracks[spkid].append(f)
                    
        for t in tracks:
            tracks[t].sort(key=lambda x:x['frameNumber'])
            frames = tracks[t]
            label = []
            bbox = {}
            record = []
            for frame in frames:
                label.append(int(frame['frameNumber'] in active_speaker[frame['Person ID']]))
                bbox[int(frame['frameNumber'])] = frame
            step = list(range(0, (frames[-1]['frameNumber']-frames[0]['frameNumber']+1), 300))
            track_id = video+':'+os.path.basename(t).split('.')[0]
            for i, start in enumerate(step):
                if len(frames[start:start+300]) > 1:
                    record.append([track_id+':'+str(i), len(frames[start:start+300]), 30.0, label[start:start+300], frames[start]['frameNumber']])
                with open(f'data/ego4d/bbox/{track_id}:{i}.json', 'w+') as f:
                    json.dump(frames[start:start+300], f)
            asd_records.extend(record)

    asd_records = pd.DataFrame(asd_records)
    asd_records.to_csv(f'data/ego4d/csv/active_speaker_{split}.csv', header=None, index=False, sep='\t')


def load_label(data):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    return res


if __name__ == '__main__':

    os.makedirs('data/ego4d', exist_ok=True)
    os.makedirs('data/ego4d/bbox', exist_ok=True)
    os.makedirs('data/ego4d/csv', exist_ok=True)

    generate(split='train')
    generate(split='val')
    generate(split='test')