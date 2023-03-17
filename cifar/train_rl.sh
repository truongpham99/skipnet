#!/usr/bin/zsh

nohup python train_rl.py train cifar10_rnn_gate_rl_110 > /nas1-nfs1/data/rds190000/progress_rl.log 2>&1 &!