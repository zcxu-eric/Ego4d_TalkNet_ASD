import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader, test_loader
from utils.tools import *
from talkNet import talkNet

def main():
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "TalkNet Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=25,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=1000,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # Data path
    parser.add_argument('--dataPath',  type=str, default="data/infer", help='Save path of Ego4d dataset')
    parser.add_argument('--savePath',     type=str, default="exps/exp")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="test", help='Choose the dataset for evaluation, val or test')
    parser.add_argument('--checkpoint',  type=str, default="data/pretrain_AVA.model", help='Model checkpoint')
    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    loader = test_loader(trialFileName = args.evalTrial, \
                         audioPath     = args.audioPath, \
                         visualPath    = args.visualPath)
    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread)

    
    download_pretrain_model_AVA()
    s = talkNet(**vars(args))
    pretrained_model = args.checkpoint
    s.loadParameters(pretrained_model)
    print("Model %s loaded from previous state!"%(pretrained_model))
    s.predict_network(loader = valLoader, **vars(args))

if __name__ == '__main__':
    main()
