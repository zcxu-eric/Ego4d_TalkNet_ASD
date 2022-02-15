import torch, math
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, tqdm, json

from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel

class talkNet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, **kwargs):
        super(talkNet, self).__init__()        
        self.model = talkNetModel().cuda()
        self.lossAV = lossAV().cuda()
        self.lossA = lossA().cuda()
        self.lossV = lossV().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']        
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()
            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda()) # feedForward
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
            audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
            outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
            outsA = self.model.forward_audio_backend(audioEmbed)
            outsV = self.model.forward_visual_backend(visualEmbed)
            labels = labels[0].reshape((-1)).cuda() # Loss
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels)
            nlossA = self.lossA.forward(outsA, labels)
            nlossV = self.lossV.forward(outsV, labels)
            nloss = nlossAV + 0.4 * nlossA + 0.4 * nlossV
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  
        sys.stdout.write("\n")      
        return loss/num, lr

    def evaluate_network(self, loader, **kwargs):
        self.eval()
        predScores = []
        top1 = 0
        index = 0
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():                
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                audioEmbed, visualEmbed = self.model.forward_cross_attention(audioEmbed, visualEmbed)
                outsAV= self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)  
                labels = labels[0].reshape((-1)).cuda()             
                _, predScore, _, prec = self.lossAV.forward(outsAV, labels)
                top1 += prec 
                index += len(labels)
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
        return 100 * (top1/index)
    
    def predict_network(self, loader, **kwargs):
        self.eval()
        os.makedirs('output/results', exist_ok=True)

        for audioFeature, visualFeature, trackid in tqdm.tqdm(loader):
            durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
            videoFeature = numpy.array(visualFeature[0, 0, ...])
            audioFeature = audioFeature[0, 0, ...]
            # length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
            length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / (100*30/25), videoFeature.shape[0])
            allScore = [] # Evaluation use TalkNet
            for duration in durationSet:
                batchSize = int(math.ceil(length / duration))
                scores = []
                with torch.no_grad():
                    for i in range(batchSize):
                        inputA = torch.FloatTensor(audioFeature[int(i * duration * (100*30/25)): int((i+1) * duration * (100*30/25)), :]).unsqueeze(0).cuda()
                        inputV = torch.FloatTensor(videoFeature[i * duration * 30: (i+1) * duration * 30,:,:]).unsqueeze(0).cuda()
                        embedA = self.model.forward_audio_frontend(inputA)
                        embedV = self.model.forward_visual_frontend(inputV)	
                        embedA, embedV = self.model.forward_cross_attention(embedA, embedV)
                        out = self.model.forward_audio_visual_backend(embedA, embedV)
                        score = self.lossAV.forward(out, labels = None)
                        scores.extend(score)
                allScore.append(scores)
            allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
            with open(f'{kwargs["dataPath"]}/bbox/{trackid[0]}.json', 'r') as f:
                bbox = json.load(f)
            for i, frame in enumerate(bbox):
                frame['score'] = str(allScore[i])
                frame['label'] = int(allScore[i].item()>-3.0)
            with open(f'output/results/{trackid[0]}.json', 'w+') as f:
                json.dump(bbox, f)

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
