MODEL=logs/squeezedet_plus_ttq_two_layers
NET=squeezeDet+
python2 ./src/eval.py --dataset=KITTI --data_path=./data/KITTI --image_set=val --eval_dir="$MODEL/val" --checkpoint_path="$MODEL/train" --net=$NET --gpu=0
