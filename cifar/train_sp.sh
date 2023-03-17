#!/usr/bin/zsh

nohup python train_sp.py train cifar10_rnn_gate_110 > /nas1-nfs1/data/rds190000/progress.log 2>&1 &!