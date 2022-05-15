# Comparison between optimizers
**ZHITONG XU (zx581)**

# Goal
The goal of this project is to show how different optimizers approach to find global minimum. I picked SGD, SGD with momentum, Adagrad, RMSProp and Adam to show how apply exponential moving average on learning rate and gradient will help during training. And see if Adaptive methods are bad at generalization compared with a finetuned SGD with momentum.

# Code Structure
This repo contains main.py, dataset.py. It create a CNN with 2 convlution layers and 1 fully connected layer. The dataset used is Fashion MNIST. The default learning rate is 0.01, batch size is 64, momentum is 0.8(I set optimal hyperparameter for SGD with momentum as default). Those parameters can be changed using ```main.py --lr=* --momentum=* --batch-size=* --epochs=* --optimizer=*```. dataset.py serves as data loader, where it checks it ./Dataset contains Fashion MNIST, if not then it will download to that directory. Output will be a .csv file, I upload all output into this repo.

# Result
Result is present in project.pptx, and there is also a blog.pdf file, where I include some interesting staff about optimizers that I read during this semester. Here is a chart from presentation slides:
```
    optimizers    TTA(98%)      coeff var
    SGD           1058          12.59%
    Momentum      615           10.98%
    Adagrad       1268          13.87%
    RMSProp       1317          15.11%
    Adam          855           16.04%
```

# Reference
1. https://arxiv.org/pdf/1608.03983.pdf
2. https://arxiv.org/pdf/1711.05101.pdf
3. https://arxiv.org/pdf/1705.08292.pdf
