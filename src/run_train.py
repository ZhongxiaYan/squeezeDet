from __future__ import absolute_import, division, print_function

import os, sys, time, json, threading
from easydict import EasyDict as edict

import numpy as np
import tensorflow as tf
import cv2

from dataset import kitti
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform
from mlb_util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', None, """Neural net architecture.""")
tf.app.flags.DEFINE_string('config', None, """Configuration for the specified model.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.Graph().as_default():
        model_dir = Model + FLAGS.model + '/'
        sys.path.append(model_dir)
        from load_model import load_model

        config_dir = model_dir + FLAGS.config + '/'
        config_path = config_dir + 'config.json'
        with open(config_path, 'r+') as f:
            mc = edict(json.load(f))

        mc.IS_TRAINING = True
        mc.PRETRAINED_MODEL_PATH = model_dir + 'pretrained.pkl'
        model = load_model(mc)

        train_dir = config_dir + 'train/'
        make_dir(train_dir)

        imdb = kitti('train', Root + 'data/KITTI', mc)
        
        save_model_statistics(model, train_dir)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(train_dir)

        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt:
            print('Loading checkpoint:', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint. Initialize from scratch')
            sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()

        if mc.NUM_THREAD > 0:
            enq_threads = []
            for _ in range(mc.NUM_THREAD):
                enq_thread = threading.Thread(target=_enqueue, args=[model, sess, coord, imdb])
                enq_thread.start()
                enq_threads.append(enq_thread)

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        run_options = tf.RunOptions(timeout_in_ms=60000)

        step = tf.train.global_step(sess, model.global_step)
        while step < mc.MAX_STEPS:
            if coord.should_stop():
                sess.run(model.FIFOQueue.close(cancel_pending_enqueues=True))
                coord.request_stop()
                coord.join(threads)
                break

            start_time = time.time()
            if step % mc.SUMMARY_STEP == 0:
                summary_step(model, sess, imdb, summary_writer)
            else:
                ops = [model.train_op, model.loss, model.conf_loss, model.bbox_loss, model.class_loss]
                if mc.NUM_THREAD > 0:
                    _, loss_value, conf_loss, bbox_loss, class_loss = sess.run(ops, options=run_options)
                else:
                    feed_dict, _, _, _ = _load_data(imdb, load_to_placeholder=False)
                    _, loss_value, conf_loss, bbox_loss, class_loss = sess.run(ops, feed_dict=feed_dict)
            duration = time.time() - start_time
            
            assert not np.isnan(loss_value), 'Model diverged. Total loss: %s, conf_loss: %s, bbox_loss: %s, class_loss: %s' % (loss_value, conf_loss, bbox_loss, class_loss)

            step = tf.train.global_step(sess, model.global_step)
            if step % mc.PRINT_STEP == 0:
                images_per_sec = mc.BATCH_SIZE / duration
                print('step %d, loss = %.2f' % (step, loss_value)

            # Save the model checkpoint periodically.
            if step % mc.CHECKPOINT_STEP == 0 or step == mc.MAX_STEPS:
                saver.save(sess, train_dir + 'model.ckpt', global_step=step)

def save_model_statistics(model, train_dir):
    stats_file = train_dir + 'model_metrics.txt'
    # save model size, flops, activations by layers
    with open(stats_file, 'w+') as f:
        for counter, label in (model.model_size_counter, '# params'), (model.activation_counter, 'Activation size'), (model.flop_counter, '# flops'):
            count = 0
            f.write('%s by layer:\n' % label)
            for c in counter:
                f.write('\t%s: %s\n' % (c[0], c[1]))
                count += c[1]
            f.write('\ttotal: %s\n\n' % count)
    print('Model statistics saved to %s' % stats_file)

def _load_data(imdb, load_to_placeholder=True):
    image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, bbox_per_batch = imdb.read_batch()

    label_indices, bbox_indices, box_delta_values, mask_indices, box_values = [], [], [], [], []
    aidx_set = set()
    for i in xrange(len(label_per_batch)): # batch_size
        for j in xrange(len(label_per_batch[i])): # number of annotations
            if (i, aidx_per_batch[i][j]) not in aidx_set:
                aidx_set.add((i, aidx_per_batch[i][j]))
                label_indices.append([i, aidx_per_batch[i][j], label_per_batch[i][j]])
                mask_indices.append([i, aidx_per_batch[i][j]])
                bbox_indices.extend([[i, aidx_per_batch[i][j], k] for k in xrange(4)])
                box_delta_values.extend(box_delta_per_batch[i][j])
                box_values.extend(bbox_per_batch[i][j])

    if load_to_placeholder:
        image_input = model.ph_image_input
        input_mask = model.ph_input_mask
        box_delta_input = model.ph_box_delta_input
        box_input = model.ph_box_input
        labels = model.ph_labels
    else:
        image_input = model.image_input
        input_mask = model.input_mask
        box_delta_input = model.box_delta_input
        box_input = model.box_input
        labels = model.labels

    feed_dict = {
        image_input: image_per_batch,
        input_mask: np.reshape(
            sparse_to_dense(
                mask_indices, [mc.BATCH_SIZE, mc.ANCHORS],
                [1.0] * len(mask_indices)),
            [mc.BATCH_SIZE, mc.ANCHORS, 1]),
        box_delta_input: sparse_to_dense(
            bbox_indices, 
            [mc.BATCH_SIZE, mc.ANCHORS, 4], box_delta_values),
        box_input: sparse_to_dense(
            bbox_indices, 
            [mc.BATCH_SIZE, mc.ANCHORS, 4], box_values),
        labels: sparse_to_dense(
            label_indices, 
            [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
            [1.0] * len(label_indices)),
    }

    return feed_dict, image_per_batch, label_per_batch, bbox_per_batch

def _enqueue(model, sess, coord, imdb):
    try:
        while not coord.should_stop():
            feed_dict, _, _, _ = _load_data(imdb)
            sess.run(model.enqueue_op, feed_dict=feed_dict)
    except Exception, e:
        coord.request_stop(e)
    
def _draw_box(im, box_list, label_list, color=(0,255,0), cdict={}, form='center'):
    assert form in ['center', 'diagonal'], 'bounding box format not accepted: %s.' % form

    for bbox, label in zip(box_list, label_list):
        if form == 'center':
            bbox = bbox_transform(bbox)

        xmin, ymin, xmax, ymax = [int(b) for b in bbox]

        l = label.split(':')[0] # text before "CLASS: PROB"
        c = cdict.get(l, color)

        # draw box
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
        # draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, label, (xmin, ymax), font, 0.3, c, 1)

def _viz_prediction_result(model, images, bboxes, labels, batch_det_bbox, batch_det_class, batch_det_prob):
    mc = model.mc
    for i in range(len(images)):
        # draw ground truth
        _draw_box(images[i], bboxes[i], [mc.CLASS_NAMES[idx] for idx in labels[i]], (0, 255, 0))

        # draw prediction
        det_bbox, det_prob, det_class = model.filter_prediction(batch_det_bbox[i], batch_det_prob[i], batch_det_class[i])

        keep_idx    = [idx for idx in range(len(det_prob)) if det_prob[idx] > mc.PLOT_PROB_THRESH]
        det_bbox    = [det_bbox[idx] for idx in keep_idx]
        det_prob    = [det_prob[idx] for idx in keep_idx]
        det_class   = [det_class[idx] for idx in keep_idx]

        _draw_box(images[i], det_bbox, [mc.CLASS_NAMES[idx] + ': %.2f'% prob for idx, prob in zip(det_class, det_prob)], (0, 0, 255))

def summary_step(model, sess, imdb, summary_writer):
    feed_dict, image_per_batch, label_per_batch, bbox_per_batch = _load_data(imdb, load_to_placeholder=False)
    op_list = [
        model.train_op, model.loss, summary_op, model.det_boxes,
        model.det_probs, model.det_class, model.conf_loss,
        model.bbox_loss, model.class_loss
    ]
    _, loss_value, summary_str, det_boxes, det_probs, det_class, conf_loss, bbox_loss, class_loss = sess.run(op_list, feed_dict=feed_dict)

    _viz_prediction_result(model, image_per_batch, bbox_per_batch, label_per_batch, det_boxes, det_class, det_probs)
    image_per_batch = bgr_to_rgb(image_per_batch)
    viz_summary = sess.run(model.viz_op, feed_dict={model.image_to_show: image_per_batch})

    summary_writer.add_summary(summary_str, step)
    summary_writer.add_summary(viz_summary, step)
    print('conf_loss: %s, bbox_loss: %s, class_loss: %s' % (conf_loss, bbox_loss, class_loss))
    
if __name__ == '__main__':
    tf.app.run()
