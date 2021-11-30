import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader
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
    parser.add_argument('--batchSize',    type=int,   default=1800,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # Data path
    parser.add_argument('--dataPath',  type=str, default="data/ego4d", help='Save path of Ego4d dataset')
    parser.add_argument('--savePath',     type=str, default="exps/exp")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [data/pretrain_AVA.model]')
    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    loader = train_loader(trialFileName = args.trainTrial, \
                          audioPath     = args.audioPath, \
                          visualPath    = args.visualPath, \
                          batchSize     = args.batchSize)
    trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread)

    loader = val_loader(trialFileName = args.evalTrial, \
                        audioPath     = args.audioPath, \
                        visualPath    = args.visualPath)
    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = 16)

    if args.evaluation == True:
        download_pretrain_model_AVA()
        s = talkNet(**vars(args))
        s.loadParameters('data/pretrain_AVA.model')
        print("Model %s loaded from previous state!"%('data/pretrain_AVA.model'))
        acc = s.evaluate_network(loader = valLoader, **vars(args))
        print("ACC %2.2f%%"%(acc))
        quit()

    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = talkNet(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = talkNet(epoch = epoch, **vars(args))
        download_pretrain_model_AVA()
        s.loadParameters('data/pretrain_AVA.model')

    acc = []
    scoreFile = open(args.scoreSavePath, "a+")

    while(1):        
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        
        if epoch % args.testInterval == 0:        
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            acc.append(s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args)))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, bestACC %2.2f%%"%(epoch, acc[-1], max(acc)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, bestACC %2.2f%%\n"%(epoch, lr, loss, acc[-1], max(acc)))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

if __name__ == '__main__':
    main()
