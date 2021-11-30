import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile

def init_args(args):
    # The details for the following folders/files can be found in the annotation of the function 'preprocess_AVA' below
    args.modelSavePath    = os.path.join(args.savePath, 'model')
    args.scoreSavePath    = os.path.join(args.savePath, 'score.txt')
    args.trialPath     = os.path.join(args.dataPath, 'csv')
    args.audioOrigPath = os.path.join(args.dataPath, 'orig_audios')
    args.visualOrigPath= os.path.join(args.dataPath, 'orig_videos')
    args.audioPath     = os.path.join(args.dataPath, '../wave')
    args.visualPath    = os.path.join(args.dataPath, '../video_imgs')
    args.trainTrial    = os.path.join(args.trialPath, 'active_speaker_train.csv')
    args.evalTrial = os.path.join(args.trialPath, f'active_speaker_{args.evalDataType}.csv')
    os.makedirs(args.modelSavePath, exist_ok = True)
    return args
 

def download_pretrain_model_AVA():
    if os.path.isfile('data/pretrain_AVA.model') == False:
        Link = "1NVIkksrD3zbxbDuDbPc_846bLfPSZcZm"
        cmd = "gdown --id %s -O %s"%(Link, 'data/pretrain_AVA.model')
        subprocess.call(cmd, shell=True, stdout=None)