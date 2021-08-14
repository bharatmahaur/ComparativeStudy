from Resnet_model import resnet101
from utils.data_utils import parser
from utils.plot_utils import ploy_rect
from utils.common_utils import make_summary
import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys
import cv2
import csv
import os

# define clasify: background and car
LABELS = {
    1: 'Person',
    2: 'Bird',
    3: 'Cat',
    4: 'Cow',
    5: 'Dog',
    6: 'Horse',
    7: 'Sheep',
    8: 'Aeroplane',
    9: 'Bicycle',
    10: 'Boat',
    11: 'Bus',
    12: 'Car',
    13: 'Motorbike',
    14: 'Train',
    15: 'Bottle',
    16: 'Chair',
    17: 'Diningtable',
    18: 'Pottedplant',
    19: 'Sofa',
    20: 'TV'
}

TRAIN_TFRECORDS = './tfrecords/voc2012/train*'
VAL_TFRECORDS = './tfrecords/voc2012/val.tfrecords'

# RESTORE_PATH = './yolov3_weight/yolov3.ckpt'
# RESTORE_PATH = './checkpoint/good/RetinaNet.ckpt-80830'

SAVE_DIR = './checkpoint/RetinaNet.ckpt'

TRAIN_NUM = 13600
VAL_NUM = 3400
NUM_CLASS = 20
NUM_ANCHORS = 9

TRAIN_EVAL_INTERNAL = 1360
VAL_EVAL_INTERNAL = 1
SAVE_INTERNAL = 1

EPOCH = 500
BATCH_SIZE = 10
IMAGE_SIZE = [224, 224]
SHUFFLE_SIZE = 500
NUM_PARALLEL = 10

# learning rate and optimizer
OPTIMIZER = 'adam'
LEARNING_RATE_INIT = 1e-4
LEARNING_RATE_TYPE = 'exponential'
LEARNING_RATE_DECAY_STEPS = 300
LEARNING_RATE_DECAY_RATE = 0.96
LEARNING_RATE_MIN = 1e-6

# variables part
RESTORE_PART = ['yolov3/darknet53_body']
UPDATE_PART = ['yolov3/RetinaNet_head']

with tf.Graph().as_default():
    train_files = tf.train.match_filenames_once(TRAIN_TFRECORDS)
    train_dataset = tf.data.TFRecordDataset(train_files, buffer_size=5)
    train_dataset = train_dataset.map(lambda x : parser(x, IMAGE_SIZE, 'train'), num_parallel_calls=NUM_PARALLEL)
    train_dataset = train_dataset.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(BATCH_SIZE)

    val_files = [VAL_TFRECORDS]
    val_dataset = tf.data.TFRecordDataset(val_files)
    val_dataset = val_dataset.map(lambda x : parser(x, IMAGE_SIZE, 'val'), num_parallel_calls=NUM_PARALLEL)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(BATCH_SIZE)

    # create a public iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)

    image, boxes, labels = iterator.get_next()
    image.set_shape([None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

    with tf.variable_scope("yolov3"):
        retinaNet = resnet101(NUM_CLASS, NUM_ANCHORS)
        feature_maps, pred_class, pred_boxes = retinaNet.forward(image, is_training=True)
        anchors = retinaNet.generate_anchors(feature_maps)

        loss, FL_loss, reg_loss = retinaNet.compute_loss(anchors, pred_class, pred_boxes, labels, boxes)
        pred_boxes, pred_labels, pred_scores = retinaNet.predict(anchors, pred_class, pred_boxes)
        eval_result = retinaNet.evaluate(pred_boxes, pred_labels, pred_scores, boxes, labels)

    global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    # Define the learning rate and optimizer
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_INIT, global_step, LEARNING_RATE_DECAY_STEPS, LEARNING_RATE_DECAY_RATE, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Summary
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('focal_loss', FL_loss)
    tf.summary.scalar('regression_loss', reg_loss)
    tf.summary.scalar('learning_rate', learning_rate)

    # Restore and update var (if finetuning)
    # restore_vars = tf.contrib.framework.get_variables_to_restore(include=RESTORE_PART)
    # update_vars = tf.contrib.framework.get_variables_to_restore(include=UPDATE_PART)
    # saver_restore = tf.train.Saver(var_list=restore_vars)

    # Restore all vars (if continue training)
    # saver_restore = tf.train.Saver()
    
    # average model 
    ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=global_step)
    ema_op = ema.apply(tf.trainable_variables())

    # Update the BN vars
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # train_step = optimizer.minimize(loss, global_step=global_step, var_list=update_vars+restore_vars)
    train_step = optimizer.minimize(loss, global_step=global_step)
    with tf.control_dependencies(update_ops):
        with tf.control_dependencies([train_step, ema_op]):
            train_op = tf.no_op(name='train')

    # Set session config    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.80

    with tf.Session(config=config) as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        # saver_restore.restore(sess, RESTORE_PATH)
        saver = tf.train.Saver()

        write_op = tf.summary.merge_all()
        write_train = tf.summary.FileWriter('./log/train', sess.graph)
        write_val = tf.summary.FileWriter('./log/val')

        print('\n------------- start to train --------------\n')
        
        for epoch in range(EPOCH):
            best_precision = 0
            sess.run(iterator.make_initializer(train_dataset))
            while True:
                try:
                    _, summary_, loss_, FL_loss_, reg_loss_, global_step_, eval_result_ = sess.run([train_op, write_op, loss, FL_loss, reg_loss, global_step, eval_result])
                    if math.isnan(loss_):
                        sys.exit(0)

                    write_train.add_summary(summary_, global_step=global_step_)
                    info = "EPOCH: {}, global_step: {}, loss: {:.3f}, focal_loss: {:.3f}, reg_loss: {:.3f}".format(epoch, global_step_, loss_, FL_loss_, reg_loss_)
                    print(info)

                    # evaluate on the train dataset
                    if (global_step_ + 1) % TRAIN_EVAL_INTERNAL == 0:
                        # Calculate the precision and recall
                        TP_array, pred_label_array, gt_label_array = eval_result_
                        precision_ = np.sum(TP_array) / (np.sum(pred_label_array) + 1e-6)
                        recall_ = np.sum(TP_array) / (np.sum(gt_label_array) + 1e-6)

                        info = "===> batch precision: {:.3f}, batch recall: {:.3f} <===".format(precision_, recall_)
                        print(info)
                except tf.errors.OutOfRangeError:
                    break
            
            # if (epoch + 1) % SAVE_INTERNAL == 0:
            #     saver.save(sess, SAVE_DIR, global_step_)
            
            if (epoch + 1) % VAL_EVAL_INTERNAL == 0:
                sess.run(iterator.make_initializer(val_dataset))
                TP_list, pred_label_list, gt_label_list, loss_list = [], [], [], []

                while True:
                    try:
                        loss_, eval_result_ = sess.run([loss, eval_result])

                        TP_array, pred_label_array, gt_label_array = eval_result_
                        TP_list.append(TP_array)
                        pred_label_list.append(pred_label_array)
                        gt_label_list.append(gt_label_array)
                        loss_list.append(loss_)
                    except tf.errors.OutOfRangeError:
                        break    

                precision = np.sum(np.array(TP_list)) / (np.sum(np.array(pred_label_list)) + 1e-6)
                recall = np.sum(np.array(TP_list)) / (np.sum(np.array(gt_label_list)) + 1e-6)

                loss_mean = np.mean(np.array(loss_list))

                info = "===> Epoch: {}, precision: {:.3f}, recall: {:.3f}".format(epoch, precision, recall)
                print(info)

                write_val.add_summary(make_summary('val_recall', recall), global_step=epoch)
                write_val.add_summary(make_summary('val_precision', precision), global_step=epoch)

                # Save the best model
                if precision > best_precision:
                    best_precision = precision
                    saver.save(sess, SAVE_DIR, epoch)
