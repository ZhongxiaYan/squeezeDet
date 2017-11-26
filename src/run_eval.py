from __future__ import absolute_import, division, print_function

import os, sys, time, json, threading
from easydict import EasyDict as edict

import numpy as np
import tensorflow as tf
import cv2

from dataset import kitti
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform, Timer
from mlb_util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', None, """Neural net architecture.""")
tf.app.flags.DEFINE_string('config', None, """Configuration for the specified model.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.Graph().as_default() as g:
        model_path = Model + FLAGS.model + '/'
        sys.path.append(model_path)
        from load_model import load_model

        config_dir = model_path + FLAGS.config + '/'
        config_path = config_dir + 'config.json'
        with open(config_path, 'r+') as f:
            mc = edict(json.load(f))

        model = load_model(mc)

        imdb = kitti('test', './data/KITTI', mc)

        train_dir = config_dir + 'train/'
        test_dir = config_dir + 'test/'
        make_dir(test_dir)

        # add summary ops and placeholders
        ap_names = []
        for cls in imdb.classes:
            for diff in 'easy', 'medium', 'hard':
                ap_names.append('APs/%s_%s' % (cls, diff))

        eval_summary_ops = []
        eval_summary_phs = {}
        for name in ap_names + ['APs/mAP', 'timing/im_detect', 'timing/im_read', 'timing/post_proc', 'num_det_per_image']:
            ph = tf.placeholder(tf.float32)
            eval_summary_ops.append(tf.summary.scalar(name, ph))
            eval_summary_phs[name] = ph

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))

        summary_writer = tf.summary.FileWriter(test_dir)
        while True:
            ckpts = set()
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if not ckpt or ckpt.model_checkpoint_path in ckpts:
                print('Wait %ss for new checkpoints to be saved ... ' % 60)
                time.sleep(60)
            else:
                ckpts.add(ckpt.model_checkpoint_path)
                print('Evaluating %s...' % ckpt.model_checkpoint_path)
                eval_checkpoint(saver, test_dir, ckpt.model_checkpoint_path, summary_writer, eval_summary_ops, eval_summary_phs, imdb, model)

def eval_checkpoint(saver, test_dir, checkpoint_path, summary_writer, eval_summary_ops, eval_summary_phs, imdb, model):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:

        saver.restore(sess, checkpoint_path)
        global_step = ckpt_path.split('/')[-1].split('-')[-1]

        num_images = len(imdb.image_idx)

        all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]

        _t = {'im_detect': Timer(), 'im_read': Timer(), 'misc': Timer()}

        num_detection = 0.0
        for i in xrange(num_images):
            _t['im_read'].tic()
            images, scales = imdb.read_image_batch(shuffle=False)
            _t['im_read'].toc()

            _t['im_detect'].tic()
            det_boxes, det_probs, det_class = sess.run(
                [model.det_boxes, model.det_probs, model.det_class],
                feed_dict={model.image_input:images})
            _t['im_detect'].toc()

            _t['misc'].tic()
            for j in xrange(len(det_boxes)): # batch
                # rescale
                det_boxes[j, :, 0::2] /= scales[j][0]
                det_boxes[j, :, 1::2] /= scales[j][1]

                det_bbox, score, det_class = model.filter_prediction(
                det_boxes[j], det_probs[j], det_class[j])

                num_detection += len(det_bbox)
                for c, b, s in zip(det_class, det_bbox, score):
                    all_boxes[c][i].append(bbox_transform(b) + [s])
            _t['misc'].toc()

            print('im_detect: %s/%s im_read: %.3fs detect: %.3fs misc: %.3fs' % (
                i + 1, num_images, _t['im_read'].average_time, _t['im_detect'].average_time, _t['misc'].average_time))

        print('Evaluating detections...')
        aps, ap_names = imdb.evaluate_detections(test_dir, global_step, all_boxes)

        print('Evaluation summary:')
        print('  Average number of detections per image: %s:' % (num_detection / num_images))
        print('  Timing:')
        print('    im_read: %.3fs detect: %.3fs misc: %.3fs' % (_t['im_read'].average_time, _t['im_detect'].average_time, _t['misc'].average_time))
        print ('  Average precisions:')
        feed_dict = {}
        for cls, ap in zip(ap_names, aps):
            feed_dict[eval_summary_phs['APs/' + cls]] = ap
            print ('    %s: %.3f' % (cls, ap))

        print ('    Mean average precision: %.3f' % np.mean(aps)))
        feed_dict[eval_summary_phs['APs/mAP']] = np.mean(aps)
        feed_dict[eval_summary_phs['timing/im_detect']] = _t['im_detect'].average_time
        feed_dict[eval_summary_phs['timing/im_read']] = _t['im_read'].average_time
        feed_dict[eval_summary_phs['timing/post_proc']] = _t['misc'].average_time
        feed_dict[eval_summary_phs['num_det_per_image']] = num_detection / num_images

        print ('Analyzing detections...')
        stats, ims = imdb.do_detection_analysis_in_eval(test_dir, global_step)

        eval_summary_str = sess.run(eval_summary_ops, feed_dict=feed_dict)
        for sum_str in eval_summary_str:
            summary_writer.add_summary(sum_str, global_step)
            
if __name__ == '__main__':
    tf.app.run()
