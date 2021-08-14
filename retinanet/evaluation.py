from Resnet_model import resnet101
from utils.data_utils import parser
from utils.plot_utils import ploy_rect
import tensorflow as tf 
import argparse
import numpy as np
import cv2

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
Arg_parser = argparse.ArgumentParser(description="RetinaNet image test")

Arg_parser.add_argument("--Val_tfrecords", type=str, default='./tfrecords/voc2012/val.tfrecords', help="The path of val tfrecords")

Arg_parser.add_argument("--Restore_path", type=str, default='./checkpoint/good/RetinaNet.ckpt-80830', help="The checkpoint you restore")

Arg_parser.add_argument("--Compare", type=bool, default=False, help="If you want to see the comparision between gruondtruth and prediction")

args = Arg_parser.parse_args()

NUM_CLASS = 20
NUM_ANCHORS = 9

BATCH_SIZE = 10
IMAGE_SIZE = [224, 224]
SHUFFLE_SIZE = 500
NUM_PARALLEL = 10

with tf.Graph().as_default():
    val_files = [args.Val_tfrecords]
    val_dataset = tf.data.TFRecordDataset(val_files)
    val_dataset = val_dataset.map(lambda x : parser(x, IMAGE_SIZE, 'val'), num_parallel_calls=NUM_PARALLEL)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(BATCH_SIZE)

    iterator = val_dataset.make_one_shot_iterator()

    image, boxes, labels = iterator.get_next()
    image.set_shape([None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

    with tf.variable_scope("yolov3"):
        retinaNet = resnet101(NUM_CLASS, NUM_ANCHORS)
        feature_maps, pred_class, pred_boxes = retinaNet.forward(image, is_training=False)
        anchors = retinaNet.generate_anchors(feature_maps)

        pred_boxes, pred_labels, pred_scores = retinaNet.predict(anchors, pred_class, pred_boxes)
        eval_result = retinaNet.evaluate(pred_boxes, pred_labels, pred_scores, boxes, labels)

    saver_restore = tf.train.Saver()

    # Set session config    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.90

    with tf.Session(config=config) as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver_restore.restore(sess, args.Restore_path)

        TP_list, pred_label_list, gt_label_list = [], [], []

        while True:
            try:
                if args.Compare:
                    # To see the true boxes and pred boxes
                    PIXEL_MEANS = np.array([[[122.7717, 115.9465, 102.9801]]])
                    PIXEL_STDV = [[[0.229, 0.224, 0.2254]]]
                    image_, pred_boxes_, pred_labels_, pred_scores_, boxes_ = sess.run([image, pred_boxes, pred_labels, pred_scores, boxes])
                    for i, single_boxes in enumerate(pred_boxes_):
                        ori_image = image_[i]
                        ori_image = (ori_image * PIXEL_STDV) + PIXEL_MEANS / 255.0

                        # rescale the coordinates to the original image
                        single_boxes[:, 0] = single_boxes[:, 0] * 416.0
                        single_boxes[:, 1] = single_boxes[:, 1] * 416.0
                        single_boxes[:, 2] = single_boxes[:, 2] * 416.0
                        single_boxes[:, 3] = single_boxes[:, 3] * 416.0

                        for j, box in enumerate(single_boxes):
                            if np.sum(box) != 0.0:
                                ploy_rect(ori_image, box, (255, 0, 0))
                                print("********************************")
                                print(box[0], " ", box[1], " ", box[2], " ", box[3], LABELS[pred_labels_[i, j]], pred_scores_[i, j])

                        # Get the trye box
                        true_boxes = boxes_[i]
                        true_boxes[:, 0] = true_boxes[:, 0]
                        true_boxes[:, 1] = true_boxes[:, 1]
                        true_boxes[:, 2] = true_boxes[:, 2]
                        true_boxes[:, 3] = true_boxes[:, 3]

                        for j, box in enumerate(true_boxes):
                            if np.sum(box) != 0.0:
                                ploy_rect(ori_image, box, (0, 0, 255))
                                print("********************************")
                                print(box[0], " ", box[1], " ", box[2], " ", box[3])

                        cv2.namedWindow('Detection_result', 0)
                        cv2.imshow('Detection_result', ori_image)
                        cv2.waitKey(0)
                else:
                    eval_result_ = sess.run(eval_result)

                    TP_array, pred_label_array, gt_label_array = eval_result_
                    TP_list.append(TP_array)
                    pred_label_list.append(pred_label_array)
                    gt_label_list.append(gt_label_array)
            except tf.errors.OutOfRangeError:
                break    

        precision = np.sum(np.array(TP_list)) / np.sum(np.array(pred_label_list))
        recall = np.sum(np.array(TP_list)) / np.sum(np.array(gt_label_list))

        info = "===> precision: {:.3f}, recall: {:.3f}".format(precision, recall)
        print(info)

