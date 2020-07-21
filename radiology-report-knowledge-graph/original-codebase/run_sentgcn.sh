#! /usr/bin/env bash

cmd=$1

if [[ $cmd -eq 0 ]]; then
python train_sentgcn.py --name sentgcn_t401v2t3_pree80_lr1e-4_ctx --pretrained /share/project/modimnet/gcnmodels/gcnclassifier_v1_ones3_t401v2t3_lr1e-6_e80.pth --train-folds 401 --val-folds 2 --test-folds 3 --gpus 0 --batch-size 8 --decoder-lr 1e-4 --vocab-path data/vocab.pkl
fi

if [[ $cmd -eq 1 ]]; then
python train_sentgcn.py --name sentgcn_t340v1t2_pree77_lr1e-4_ctx --pretrained /share/project/modimnet/gcnmodels/gcnclassifier_v1_ones3_t401v2t3_lr1e-6_e77.pth --train-folds 340 --val-folds 1 --test-folds 2 --gpus 0 --batch-size 8 --decoder-lr 1e-4 --vocab-path data/vocab.pkl
fi

if [[ $cmd -eq 2 ]]; then
python train_sentgcn.py --name sentgcn_t234v0t1_pree126_lr1e-4_ctx --pretrained /share/project/modimnet/gcnmodels/gcnclassifier_v1_ones3_t401v2t3_lr1e-6_e126.pth --train-folds 234 --val-folds 0 --test-folds 1 --gpus 0 --batch-size 8 --decoder-lr 1e-4 --vocab-path data/vocab.pkl
fi

if [[ $cmd -eq 3 ]]; then
python train_sentgcn.py --name sentgcn_t123v4t0_pree144_lr1e-4_ctx --pretrained /share/project/modimnet/gcnmodels/gcnclassifier_v1_ones3_t401v2t3_lr1e-6_e144.pth --train-folds 123 --val-folds 4 --test-folds 0 --gpus 0 --batch-size 8 --decoder-lr 1e-4 --vocab-path data/vocab.pkl
fi

if [[ $cmd -eq 4 ]]; then
python train_sentgcn.py --name sentgcn_t012v3t4_pree92_lr1e-4_ctx --pretrained /share/project/modimnet/gcnmodels/gcnclassifier_v1_ones3_t401v2t3_lr1e-6_e92.pth --train-folds 012 --val-folds 3 --test-folds 4 --gpus 0 --batch-size 8 --decoder-lr 1e-4 --vocab-path data/vocab.pkl
fi

