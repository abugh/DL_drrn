# RDN-pytorch
This is an unofficial implementation of "Residual Dense Network for Super Resolution (RDN)", CVPR 2018 in Pytorch. 

You can get the official RDN implementation [here](https://github.com/yulunzhang/RDN).

This implementation is modified from the implementation of [DRRN](https://github.com/jt827859032/DRRN-pytorch)

## Usage
### Training
```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--clip CLIP] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED] [--optimizer OPTIMIZER]
               [--batchNormalize] [--NetStructure NETSTRUCTURE]
               [--BlockNum BLOCKNUM] [--BlockSize BLOCKSIZE]
               [--DRRNsize DRRNSIZE] [--ESPCN]
               
optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        Training batch size
  --nEpochs NEPOCHS     Number of epochs to train for
  --lr LR               Learning Rate, Default=0.1
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default=5
  --cuda                Use cuda?
  --resume RESUME       Path to checkpoint, Default=None
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --clip CLIP           Clipping Gradients, Default=0.01
  --threads THREADS     Number of threads for data loader to use, Default=1
  --momentum MOMENTUM   Momentum, Default=0.9
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        Weight decay, Default=1e-4
  --pretrained PRETRAINED
                        path to pretrained model, Default=None
  --optimizer OPTIMIZER
                        SGD or Adam?, Default=SGD
  --batchNormalize      use batch Normalize?
  --NetStructure NETSTRUCTURE
                        DenseNet or DRRN?, Default=DenseNet
  --BlockNum BLOCKNUM   DenseNet BlockNum, Default=8
  --BlockSize BLOCKSIZE
                        DenseNet BlockSize, Default=4
  --DRRNsize DRRNSIZE   DRRNsize, Default=25
  --ESPCN               use ESPCN?
```

### Evaluation
```
usage: eval.py [-h] [--cuda] [--model MODEL] [--dataset DATASET]
               [--scale SCALE]

PyTorch DRRN Evaluation

optional arguments:
  -h, --help         show this help message and exit
  --cuda             use cuda?
  --model MODEL      model path
  --dataset DATASET  dataset name, Default: Set5
```
An example of training usage is shown as follows:
```
python eval.py --cuda
```

### Prepare Training dataset
  - the training data is generated with Matlab Bicubic Interpolation, please refer [Code for Data Generation](/data/generate_trainingset_x234.m) for creating training files.
  
### Performance
  - We provide some ***rough*** pre-trained DRRN and RDN [models](/model) trained on [291](/data/Train_291) images with data augmentation. The model can achieve a better performance with some smart optimization strategies(SGD with momentum and Adam). For the DRRN adn RDN implementation, you can manually modify the number of recursive blocks [here](/drrn.py#L26:18).
  - The same adjustable gradient clipping's implementation as original paper.
  - No bias is used in this implementation.
  - Batch normalization is used in this implementation,but you can cancel it.
  - Performance in PSNR on Set5 

