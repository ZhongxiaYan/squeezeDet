from __future__ import absolute_import, division, print_function

import os, sys
import joblib

from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton

def load_model(mc):
    return SqueezeDetPlus(mc)

class SqueezeDetPlus(ModelSkeleton):
    def __init__(self, mc):
        with tf.device('/gpu:0'):
            ModelSkeleton.__init__(self, mc)

            self._add_forward_graph()
            self._add_interpretation_graph()
            self._add_loss_graph()
            self._add_train_graph()
            self._add_viz_graph()

    def _add_forward_graph(self):
        mc = self.mc
        if mc.LOAD_PRETRAINED_MODEL:
            assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), 'Cannot find pretrained model at %s' % mc.PRETRAINED_MODEL_PATH
            self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

        conv1 = self._conv_layer('conv1', self.image_input, filters=96, size=7, stride=2, padding='VALID', freeze=True)
        pool1 = self._pooling_layer('pool1', conv1, size=3, stride=2, padding='VALID')

        fire2 = self._fire_layer('fire2', pool1, s1x1=96, e1x1=64, e3x3=64, freeze=True)

        fire3 = self._fire_layer('fire3', fire2, s1x1=96, e1x1=64, e3x3=64, freeze=True)
        fire4 = self._fire_layer('fire4', fire3, s1x1=192, e1x1=128, e3x3=128, freeze=True)
        pool4 = self._pooling_layer('pool4', fire4, size=3, stride=2, padding='VALID')

        fire5 = self._fire_layer('fire5', pool4, s1x1=192, e1x1=128, e3x3=128, freeze=True)
        fire6 = self._fire_layer('fire6', fire5, s1x1=288, e1x1=192, e3x3=192, freeze=True)
        fire7 = self._fire_layer('fire7', fire6, s1x1=288, e1x1=192, e3x3=192, freeze=True)

        fire8 = self._fire_layer('fire8', fire7, s1x1=384, e1x1=256, e3x3=256, freeze=True)
        pool8 = self._pooling_layer('pool8', fire8, size=3, stride=2, padding='VALID')

        fire9 = self._fire_layer('fire9', pool8, s1x1=384, e1x1=256, e3x3=256, freeze=True)

        # Two extra fire modules that are not trained before
        fire10 = self._fire_layer_ternary('fire10_ternary', fire9, s1x1=384, e1x1=256, e3x3=128, t3x3=128)
        fire11 = self._fire_layer('fire11', fire10, s1x1=384, e1x1=256, e3x3=256, freeze=True)
        dropout11 = tf.nn.dropout(fire11, self.keep_prob, name='drop11')

        num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
        self.preds = self._conv_layer('conv12', dropout11, filters=num_output, size=3, stride=1, padding='SAME', xavier=False, relu=False, stddev=0.0001)

    def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.01, freeze=False):
        """
        Fire layer constructor.

        Args:
          layer_name: layer name
          inputs: input tensor
          s1x1: number of 1x1 filters in squeeze layer.
          e1x1: number of 1x1 filters in expand layer.
          e3x3: number of 3x3 filters in expand layer.
          freeze: if true, do not train parameters in this layer.
        Returns:
          fire layer operation.
        """
        mc = self.mc
        sq1x1 = self._conv_layer(
            layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
            padding='SAME', stddev=stddev, freeze=freeze)
        ex1x1 = self._conv_layer(
            layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
            padding='SAME', stddev=stddev, freeze=freeze)
        ex3x3 = self._conv_layer(
            layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
            padding='SAME', stddev=stddev, freeze=freeze)
        return tf.concat([ex1x1, ex3x3], 3, name=layer_name + '/concat')

    def _fire_layer_ternary(self, layer_name, inputs, s1x1, e1x1, e3x3, t3x3, stddev=0.01, freeze=False):
        """
        Fire layer constructor.

        Args:
          layer_name: layer name
          inputs: input tensor
          s1x1: number of 1x1 filters in squeeze layer.
          e1x1: number of 1x1 filters in expand layer.
          e3x3: number of 3x3 filters in expand layer.
          freeze: if true, do not train parameters in this layer.
        Returns:
          fire layer operation.
        """
        mc = self.mc
        sq1x1 = self._conv_layer(
            layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
            padding='SAME', stddev=stddev, override_ternary=True)
        ex1x1 = self._conv_layer(
            layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
            padding='SAME', stddev=stddev, override_ternary=True)
        ex3x3 = self._conv_layer(
            layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
            padding='SAME', stddev=stddev, override_ternary=True)
        tx3x3 = self._conv_layer(
            layer_name+'/expand3x3_t', sq1x1, filters=t3x3, size=3, stride=1,
            padding='SAME', stddev=stddev)
        return tf.concat([ex1x1, ex3x3, tx3x3], 3, name=layer_name + '/concat')
