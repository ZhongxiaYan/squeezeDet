#!/bin/sh
python2 ./src/train.py --dataset=KITTI --data_path=./data/KITTI --image_set=train --train_dir="logs/squeezedet_plus_ttq_two_layers/train" --net=squeezeDet+ --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.0_SR_0.750.pkl --summary_step=100 --checkpoint_step=500 --gpu=0 --max_steps=15000
