import glob, json, os
from argparse import ArgumentParser

def generate_results(split='val'):

    os.makedirs('output/final', exist_ok=True)
    with open('data/track_results/v.txt', 'r') as f:
        videos = f.readlines()
    video2res = {  v.split('/')[-1].split('.')[0]:i for i,v in enumerate(videos) }
    res2video = {  i:v.split('/')[-1].split('.')[0] for i,v in enumerate(videos) }
    with open(f'data/split/{split}.list', 'r') as f:
        videos = [line.strip() for line in f.readlines()]
    tracklets = [f'data/track_results/results/{video2res[v]}.txt' for v in videos]

    for t in tracklets:
        if not os.path.exists(t):
            print(t)
            continue
        name = res2video[int(t.split('/')[-1][:-4])]
        asdres = glob.glob(f'output/results/{name}*.json')
        pidre = {}
        for asd in asdres:
            with open(asd, 'r') as f:
                lines = json.load(f)
                for line in lines:
                    identifier = '{}:{}'.format(line['frameNumber'], line['pid'])
                    pidre[identifier] = line
        with open(t, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            data = line.split()
            identifier = '{}:{}'.format(data[0], data[1])
            if identifier in pidre:
                new_lines.append('{} {} {}\n'.format(line[:-1], pidre[identifier]['score'], pidre[identifier]['label']))
            else:
                print(t, line)
                new_lines.append('{} {} {}\n'.format(line[:-1], 0, 0))
        with open(f'output/final/{name}.txt', 'w+') as f:
            f.writelines(new_lines)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--evalDataType', type=str, default="test", help='Choose the dataset for evaluation, val or test')
    args = parser.parse_args()
    generate_results(args.evalDataType)