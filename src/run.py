from __future__ import absolute_import, division, print_function

import os, sys, time, json, threading
from easydict import EasyDict as edict

import numpy as np
import tensorflow as tf

from dataset import kitti
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform, Timer
from mlb_util import *
from config.kitti_squeezeDetPlus_config import kitti_squeezeDetPlus_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

save_roots = {
    'jeffzhang' : '/media/jeffzhang/1FBD76B338533CA0/squeezeDet',
    'ubuntu' : ''
}

flags = tf.app.flags
flags.DEFINE_string('gpu', '0', 'gpu id.')
flags.DEFINE_boolean('train', True, 'True for training phase, false for evaluation.')
flags.DEFINE_string('save_root', save_roots[os.environ['USER']], 'Checkpoints will be saved in subdirectories of this root. Symlinks will point to train subdirectories.')
flags.DEFINE_boolean('debug', False, 'If true, train and validate for 1 iteration.')
flags.DEFINE_string('eval_start_checkpoint', '', 'Evaluate all checkpoints after this number')

FLAGS = flags.FLAGS

def process_model_path(model_path):
    config_dir = os.path.abspath(model_path)
    assert config_dir.startswith(Models), 'Invalid config directory %s' % config_dir
    return config_dir[len(Models):].split('/')[:2]

def main(argv):
    model_name, config_name = process_model_path(argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    model_dir = Models + model_name + '/'

    mc = kitti_squeezeDetPlus_config()
    config_dir = model_dir + config_name + '/'
    config_path = config_dir + 'config.json'
    with open(config_path, 'r+') as f: # load custom params
        for key, value in json.load(f).items():
            mc[key] = value

    mc.IS_TRAINING = FLAGS.train
    mc.PRETRAINED_MODEL_PATH = model_dir + 'pretrained.pkl'
    train_dir = config_dir + 'train/'
    test_dir = config_dir + 'val/'
    if FLAGS.save_root:
        for dname in ['train', 'val']:
            orig_dir = os.path.join(config_dir, dname)
            save_dir = os.path.join(FLAGS.save_root, 'models', model_name, config_name, dname)
            make_dir(save_dir)

            if not os.path.exists(orig_dir):
                os.symlink(save_dir, orig_dir)
                print('Creating symlink %s -> %s' % (orig_dir, save_dir))
            elif not os.path.islink(orig_dir):
                raise RuntimeError('%s exists but is not a link. Cannot create new link to %s' % (orig_dir, save_dir))

    if FLAGS.debug:
        mc.PRINT_STEP = mc.SUMMARY_STEP = mc.CHECKPOINT_STEP = mc.MAX_STEPS = 1

    if mc.IS_TRAINING:
        kitti_set = 'train'
        summary_dir = train_dir
    else:
        kitti_set = 'val'
        summary_dir = test_dir
        mc.BATCH_SIZE = 1
    imdb = kitti(kitti_set, Root + 'data/KITTI', mc)
    summary_writer = tf.summary.FileWriter(summary_dir)

    sys.path.append(model_dir)
    from load_model import load_model
    model = load_model(mc)

    def _load_data(load_to_placeholder=True):
        image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, bbox_per_batch = imdb.read_batch()

        label_indices, bbox_indices, box_delta_values, mask_indices, box_values = [], [], [], [], []
        aidx_set = set()
        for i in range(len(label_per_batch)):
            for j in range(len(label_per_batch[i])):
                if (i, aidx_per_batch[i][j]) not in aidx_set:
                    aidx_set.add((i, aidx_per_batch[i][j]))
                    label_indices.append(
                        [i, aidx_per_batch[i][j], label_per_batch[i][j]])
                    mask_indices.append([i, aidx_per_batch[i][j]])
                    bbox_indices.extend(
                        [[i, aidx_per_batch[i][j], k] for k in range(4)])
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
                    [1.0]*len(mask_indices)),
                [mc.BATCH_SIZE, mc.ANCHORS, 1]),
            box_delta_input: sparse_to_dense(
                bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
                box_delta_values),
            box_input: sparse_to_dense(
                bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
                box_values),
            labels: sparse_to_dense(
                label_indices,
                [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
                [1.0] * len(label_indices)),
        }
        return feed_dict, image_per_batch, label_per_batch, bbox_per_batch

    def _enqueue(sess, coord):
        try:
            while not coord.should_stop():
                feed_dict, _, _, _ = _load_data()
                sess.run(model.enqueue_op, feed_dict=feed_dict)
        except Exception, e:
            if not sess.run(model.FIFOQueue.is_closed()):
                coord.request_stop(e)


    if mc.IS_TRAINING:
        save_model_statistics(model, train_dir + 'model_metrics.txt')

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
        summary_op = tf.summary.merge_all()

        if 'RESTORE_MODEL' in mc:
            checkpoint_path = Models + mc.RESTORE_MODEL
            restore_training_variables = False
        else:
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt:
                checkpoint_path = ckpt.model_checkpoint_path
            else:
                checkpoint_path = None
            restore_training_variables = True
        if checkpoint_path:
            restore_variables = tf.global_variables() if restore_training_variables else tf.model_variables()
            saver = tf.train.Saver(var_list=get_checkpoint_variables(checkpoint_path, restore_variables), max_to_keep=None)
            print('Loading checkpoint:', checkpoint_path)
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_path)
        else:
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
            print('No checkpoint. Initialize from scratch')
            sess.run(tf.global_variables_initializer())
        
        coord = tf.train.Coordinator()

        if mc.NUM_THREAD > 0:
            enq_threads = []
            for _ in range(mc.NUM_THREAD):
                enq_thread = threading.Thread(target=_enqueue, args=[sess, coord])
                enq_thread.start()
                enq_threads.append(enq_thread)

        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        run_options = tf.RunOptions(timeout_in_ms=60000)

        step = tf.train.global_step(sess, model.global_step)
        try:
            while step < mc.MAX_STEPS and not coord.should_stop():
                start_time = time.time()
                if step % mc.SUMMARY_STEP == 0:
                    feed_dict, image_per_batch, label_per_batch, bbox_per_batch = _load_data(load_to_placeholder=False)
                    op_list = [
                        model.train_op, model.loss, summary_op, model.det_boxes,
                        model.det_probs, model.det_class, model.conf_loss,
                        model.bbox_loss, model.class_loss
                    ]
                    _, loss_value, summary_str, det_boxes, det_probs, det_class, conf_loss, bbox_loss, class_loss = sess.run(op_list, feed_dict=feed_dict)

                    viz_prediction_result(model, image_per_batch, bbox_per_batch, label_per_batch, det_boxes, det_class, det_probs)
                    image_per_batch = bgr_to_rgb(image_per_batch)
                    viz_summary = sess.run(model.viz_op, feed_dict={model.image_to_show: image_per_batch})

                    summary_writer.add_summary(summary_str, step)
                    summary_writer.add_summary(viz_summary, step)
                    print('step %s, conf_loss: %s, bbox_loss: %s, class_loss: %s' % (step, conf_loss, bbox_loss, class_loss))
                else:
                    ops = [model.train_op, model.loss, model.conf_loss, model.bbox_loss, model.class_loss]
                    if mc.NUM_THREAD > 0:
                        _, loss_value, conf_loss, bbox_loss, class_loss = sess.run(ops, options=run_options)
                    else:
                        feed_dict, _, _, _ = _load_data(load_to_placeholder=False)
                        _, loss_value, conf_loss, bbox_loss, class_loss = sess.run(ops, feed_dict=feed_dict)
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged. Total loss: %s, conf_loss: %s, bbox_loss: %s, class_loss: %s' % (loss_value, conf_loss, bbox_loss, class_loss)

                step = tf.train.global_step(sess, model.global_step)
                if step % mc.PRINT_STEP == 0:
                    print('step %d, loss = %.2f' % (step, loss_value))
                if step % mc.CHECKPOINT_STEP == 0 or step >= mc.MAX_STEPS:
                    print('Saving checkpoint')
                    saver.save(sess, train_dir + 'model.ckpt', global_step=step)
        except KeyboardInterrupt:
            pass
        finally:
            print('Finished training')
            sess.run(model.FIFOQueue.close(cancel_pending_enqueues=True))
            coord.request_stop()
            coord.join(threads)
    else:
        ap_names = []
        for cls in imdb.classes:
            for diff in 'easy', 'medium', 'hard':
                ap_names.append('APs/%s_%s' % (cls, diff))

        eval_summary_phs = {}
        for name in ap_names + ['APs/mAP', 'timing/im_detect', 'timing/im_read', 'timing/post_proc', 'num_det_per_image']:
            eval_summary_phs[name] = tf.placeholder(tf.float32)
        eval_summary_ops = [tf.summary.scalar(name, ph) for name, ph in eval_summary_phs.items()]

        def eval_checkpoint_path(checkpoint_path):
            print('Evaluating %s...' % checkpoint_path)
            global_step = checkpoint_path.split('/')[-1].split('-')[-1]
            if os.path.exists(os.path.join(test_dir, 'detection_files_' + str(global_step))):
                print('Already evaluated')
                return
            eval_checkpoint(model, imdb, summary_writer, test_dir, checkpoint_path, eval_summary_phs, eval_summary_ops)            

        if FLAGS.eval_start_checkpoint:
            start_step = int(FLAGS.eval_start_checkpoint)
            checkpoints = tf.train.get_checkpoint_state(train_dir).all_model_checkpoint_paths
            for checkpoint_path in checkpoints:
                step = int(checkpoint_path.split('model.ckpt-')[1])
                if step >= start_step:
                    eval_checkpoint_path(checkpoint_path)

        while True:
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if not ckpt:
                print('Wait %ss for new checkpoints to be saved ... ' % 30)
                time.sleep(30)
            else:
                eval_checkpoint_path(ckpt.model_checkpoint_path)

def get_checkpoint_variables(checkpoint_path, graph_variables):
    print(checkpoint_path)
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    saved_variables = set(reader.get_variable_to_shape_map().keys())
    graph_variables = { v.name.encode('ascii').split(':')[0] : v for v in graph_variables }
    var_list = { n : v for n, v in graph_variables.items() if n in saved_variables }
    return var_list

def viz_prediction_result(model, images, bboxes, labels, batch_det_bbox, batch_det_class, batch_det_prob):
    mc = model.mc
    for i in range(len(images)):
        # draw ground truth
        draw_box(images[i], bboxes[i], [mc.CLASS_NAMES[idx] for idx in labels[i]], (0, 255, 0))

        # draw prediction
        det_bbox, det_prob, det_class = model.filter_prediction(batch_det_bbox[i], batch_det_prob[i], batch_det_class[i])

        keep_idx    = [idx for idx in range(len(det_prob)) if det_prob[idx] > mc.PLOT_PROB_THRESH]
        det_bbox    = [det_bbox[idx] for idx in keep_idx]
        det_prob    = [det_prob[idx] for idx in keep_idx]
        det_class   = [det_class[idx] for idx in keep_idx]

        draw_box(images[i], det_bbox, [mc.CLASS_NAMES[idx] + ': %.2f'% prob for idx, prob in zip(det_class, det_prob)], (0, 0, 255))

def eval_checkpoint(model, imdb, summary_writer, test_dir, checkpoint_path, eval_summary_phs, eval_summary_ops):
    saver = tf.train.Saver(get_checkpoint_variables(checkpoint_path, tf.global_variables()))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)
                      
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
                feed_dict={ model.image_input : images })
            _t['im_detect'].toc()

            _t['misc'].tic()
            for j in xrange(len(det_boxes)): # batch
                # rescale
                det_boxes[j, :, 0::2] /= scales[j][0]
                det_boxes[j, :, 1::2] /= scales[j][1]

                det_bbox, score, det_class = model.filter_prediction(det_boxes[j], det_probs[j], det_class[j])
                num_detection += len(det_bbox)
                for c, b, s in zip(det_class, det_bbox, score):
                    all_boxes[c][i].append(bbox_transform(b) + [s])
            _t['misc'].toc()

            print('im_detect: %s/%s im_read: %.3fs detect: %.3fs misc: %.3fs' % (i + 1, num_images, _t['im_read'].average_time, _t['im_detect'].average_time, _t['misc'].average_time))

        print('Evaluating detections...')
        aps, ap_names = imdb.evaluate_detections(test_dir, global_step, all_boxes)

        print('Evaluation summary:')
        print('  Average number of detections per image: %s:' % (num_detection / num_images))
        print('  Timing:')
        print('    im_read: %.3fs detect: %.3fs misc: %.3fs' % (_t['im_read'].average_time, _t['im_detect'].average_time, _t['misc'].average_time))
        print('  Average precisions:')
        feed_dict = {}
        for cls, ap in zip(ap_names, aps):
            feed_dict[eval_summary_phs['APs/' + cls]] = ap
            print('    %s: %.3f' % (cls, ap))

        print('    Mean average precision: %.3f' % np.mean(aps))
        feed_dict[eval_summary_phs['APs/mAP']] = np.mean(aps)
        feed_dict[eval_summary_phs['timing/im_detect']] = _t['im_detect'].average_time
        feed_dict[eval_summary_phs['timing/im_read']] = _t['im_read'].average_time
        feed_dict[eval_summary_phs['timing/post_proc']] = _t['misc'].average_time
        feed_dict[eval_summary_phs['num_det_per_image']] = num_detection / num_images

        print('Analyzing detections...')
        stats, ims = imdb.do_detection_analysis_in_eval(test_dir, global_step)

        eval_summary_str = sess.run(eval_summary_ops, feed_dict=feed_dict)
        for sum_str in eval_summary_str:
            summary_writer.add_summary(sum_str, global_step)
                      
if __name__ == '__main__':
    tf.app.run()
