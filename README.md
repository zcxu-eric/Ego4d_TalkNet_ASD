## Ego4d Audio-visual Diarization Benchmark: Active Speaker Detection 
This repository contains the code adapted from TalkNet, an active speaker detection model to detect 'whether the face in the screen is speaking or not?'. For more details, please refer to [[Paper](https://arxiv.org/pdf/2107.06592.pdf)]    [[Video_English](https://youtu.be/C6bpAgI9zxE)]    [[Video_Chinese](https://www.bilibili.com/video/bv1Yw411d7HG)].


***

### Dependencies

Start from building the environment
```
sudo apt-get install ffmpeg
conda create -n TalkNet python=3.7.9 anaconda
conda activate TalkNet
pip install -r requirement.txt
```

Start from the existing environment
```
pip install -r requirement.txt
```

***

## TalkNet in Ego4d dataset

### Data preparation

Download data manifest (`manifest.csv`) and annotations (`av.json`) for audio-visual diarization benchmark following the Ego4D download [instructions](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md).

Note: the default folder to save videos and annotations is ```./data```, please create symbolic links in ```./data``` if you save them in another directory. The structure should be like this:

data/
* csv/
  * manifest.csv
* json/
  * av.json
* split/
  * test.list
  * train.list
  * val.list
  * full.list
* videos/
  * 00407bd8-37b4-421b-9c41-58bb8f141716.mp4
  * 007beb60-cbab-4c9e-ace4-f5f1ba73fccf.mp4
  * ...
  
Run the following script to download videos and generate clips:
```
python utils/download_clips.py
```

Run the following scripts to preprocess the videos and annotations:

```
bash scripts/extract_frame.sh
bash scripts/extract_wave.sh
python utils/preprocessing.py
```

### Training
Then you can train TalkNet on Ego4s using:
```
python trainTalkNet.py
```
The results will be saved in `exps/exp`:

`exps/exp/score.txt`: output score file

`exps/exp/model/model_00xx.model`: trained model

`exps/exps/val_res.csv`: prediction for val set.

### Pretrained model

The model pretrained on AVA will automatically be downloaded into `data/pretrain_AVA.model`.

Our [model](https://drive.google.com/drive/folders/1lNQxdlCtFVYQoKBYA0EoPoiw_Mtc4JTO?usp=sharing) trained on Ego4d performs `ACC:79.27%` on test set. 


***

## Inference

### Data preparation

We can predict active speakers for each person given the face tracks. Please put the tracking results in ``./data/track_results``. The structure should be like this:

data/
* track_results/
  * results/
    * 0.txt
    * 1.txt
    * ...
  * v.txt

Run the following script to make the tracking results compatible with dataloader (specify subset from ```['full', 'val', 'test']```):
```
python utils/process_tracking_result.py --evalDataType ${SUBSET}
```

#### Usage

Run the following script, specify the checkpoint and subset:

```
python inferTalkNet.py --checkpoint ${MODEL_PATH} --evalDataType ${SUBSET}
```

Finally, run the postprocessing script to make the predictions compatible with other components in this diarization benchmark:
```
python utils/postprocess.py --evalDataType ${SUBSET}
```

### Citation

Please cite the following paper if our code is helpful to your research.
```
@article{grauman2021ego4d,
  title={Ego4d: Around the world in 3,000 hours of egocentric video},
  author={Grauman, Kristen and Westbury, Andrew and Byrne, Eugene and Chavis, Zachary and Furnari, Antonino and Girdhar, Rohit and Hamburger, Jackson and Jiang, Hao and Liu, Miao and Liu, Xingyu and others},
  journal={arXiv preprint arXiv:2110.07058},
  year={2021}
}
```
