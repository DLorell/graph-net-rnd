#! /usr/bin/env bash

<<<<<<< HEAD
source ~/miniconda3/bin/activate radreport

python train_gcnclassifier.py --name gcnclassifier_v1_ones3_t401v2t3_lr1e-6 --pretrained="" --checkpoint="/scratch/lorell/gcnmodels/gcnclassifier_v1_ones3_t401v2t3_lr1e-6_e144.pth" --dataset-dir data --train-folds 401 --val-folds 2 --test-folds 3 --lr 1e-6 --batch-size 8 --gpus 0 --num-epochs 150 --log-freq 5
=======
python train_gcnclassifier.py --name gcnclassifier_v1_ones3_t401v2t3_lr1e-6 --pretrained="" --dataset-dir data --train-folds 401 --val-folds 2 --test-folds 3 --lr 1e-6 --batch-size 2 --gpus 0 --num-epochs 150
>>>>>>> 4c14b0d8e6af6cb26f2799f8fc67242c0a60b47d
