# Ternarizing SqueezeDet
We attempt ternarization on layers of SqueezeDet. See below for the paper on SqueezeDet. We cloned the repo from https://github.com/BichenWuUCB/squeezeDet and made modifications.

## _SqueezeDet:_ Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving
By Bichen Wu, Forrest Iandola, Peter H. Jin, Kurt Keutzer (UC Berkeley & DeepScale)

This repository contains a tensorflow implementation of SqueezeDet, a convolutional neural network based object detector described in our paper: https://arxiv.org/abs/1612.01051. If you find this work useful for your research, please consider citing:

    @inproceedings{squeezedet,
        Author = {Bichen Wu and Forrest Iandola and Peter H. Jin and Kurt Keutzer},
        Title = {SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving},
        Journal = {arXiv:1612.01051},
        Year = {2016}
    }
    
## Installation

The following instructions are written for Linux-based distros.

- Clone the SqueezeDet repository:

  ```Shell
  git clone https://github.com/ZhongxiaYan/squeezeDet.git
  ```
  Let's call the top level directory of SqueezeDet `$SQDT_ROOT`. 

## Virtual Environment and Demo
Optional, refer to https://github.com/BichenWuUCB/squeezeDet for details.

## Download Data and Pretrained Model
- Download KITTI object detection dataset: [images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) and [labels](http://www.cvlibs.net/download.php?file=data_object_label_2.zip). Put them under `$SQDT_ROOT/data/KITTI/`. Unzip them, then you will get two directories:  `$SQDT_ROOT/data/KITTI/training/` and `$SQDT_ROOT/data/KITTI/testing/`. 

- Now we need to split the training data into a training set and a vlidation set. 

  ```Shell
  cd $SQDT_ROOT/data/KITTI/
  mkdir ImageSets
  cd ./ImageSets
  ls ../training/image_2/ | grep ".png" | sed s/.png// > trainval.txt
  ```
  `trainval.txt` contains indices to all the images in the training data. In our experiments, we randomly split half of indices in `trainval.txt` into `train.txt` to form a training set and rest of them into `val.txt` to form a validation set. For your convenience, we provide a script to split the train-val set automatically. Simply run
  
    ```Shell
  cd $SQDT_ROOT/data/
  python random_split_train_val.py
  ```
  
  then you should get the `train.txt` and `val.txt` under `$SQDT_ROOT/data/KITTI/ImageSets`. 

  When above two steps are finished, the structure of `$SQDT_ROOT/data/KITTI/` should at least contain:

  ```Shell
  $SQDT_ROOT/data/KITTI/
                    |->training/
                    |     |-> image_2/00****.png
                    |     L-> label_2/00****.txt
                    |->testing/
                    |     L-> image_2/00****.png
                    L->ImageSets/
                          |-> trainval.txt
                          |-> train.txt
                          L-> val.txt
  ```

- Next, download the SqueezeNet CNN model pretrained for ImageNet classification:
  ```Shell
  cd $SQDT_ROOT/data/
  wget https://www.dropbox.com/s/fzvtkc42hu3xw47/SqueezeNet.tgz
  tar -xzvf SqueezeNet.tgz
  ```

- Compile the official evaluation script of KITTI dataset
  ```Shell
  cd $SQDT_ROOT/src/dataset/kitti-eval
  make
  ```

## Defining a Model
Let's call our model `$model`. We define the model architecture in `$SQDT_ROOT/models/$model/load_model.py`. The model is a class that is returned by the function `load_model`, and the class must contain member variables used in `$SQDT_ROOT/src/run.py`. Refer to `$SQDT_ROOT/models/squeezedet_plus/load_model.py` as an example.

### Defining a Config
We can define a specific set of hyperparameters (name this set `$config`) for `$model` in `$SQDT_ROOT/models/$model/$config/config.json`. The model may load in and use these defined configs. Refer to the `default` config for `squeezedet_plus` model as an example.

## Training
We train a model with a specific set of configurations at a time.
  ```Shell
  cd $SQDT_ROOT/
  python src/run.py models/$model/$config/
  ```
Training logs and checkpoints are saved to the directory `python src/run.py models/$model/$config/train` .

## Evaluation
You can evaluate a model with a set of configs by running
  ```Shell
  cd $SQDT_ROOT/
  python src/run.py models/$model/$config/ --train False
  ```
Evaluation logs and checkpoints are saved to the directory `python src/run.py models/$model/$config/val` .

You can run evaluation simultaneously with training for a GPU with 8G+ memory.

Finally, to monitor training and evaluation process, you can use tensorboard by
  ```Shell
  tensorboard --logdir=models/$model/$config/
  ```
