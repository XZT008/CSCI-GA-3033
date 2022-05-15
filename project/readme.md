# Comparison between optimizers
**ZHITONG XU (zx581)**

The goal of this project is to show how different ooptimizers approach to find global minimum. I picked SGD, SGD with momentum, Adagrad, RMSProp and Adam to show how apply exponential moving average on learning rate and gradient will help during training.

This repo contains main.py. It create a CNN with 2 convlution layer and 1 fully connected layer. The dataset used is Fashion MNIST. The default learning rate is 0.01, batch size is 64, momentum is 0.8. Those parameters can be changed using --lr --momentum --batch.
