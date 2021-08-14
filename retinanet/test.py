from utils.plot_utils import ploy_rect
from utils.common_utils import augmentation
from Resnet_model import resnet101
import argparse
import tensorflow as tf 
import numpy as np  
import argparse
import random
import cv2
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

parser = argparse.ArgumentParser(description="RetinaNet image test")

parser.add_argument("--Input_dir", type=str, default='./demo_image', help="The directory of your test images")

parser.add_argument("--Single_test", type=bool, default=False, help="If you want to test a single image or more than one")

parser.add_argument("--Input_image", type=str, default='./demo_image/2007_000664.jpg', help="The path of your image")

parser.add_argument("--Output_dir", type=str, default="./demo_result/", help="The save path of your result")

parser.add_argument("--Restore_path", type=str, default='./checkpoint/good/RetinaNet.ckpt-80830', help="The checkpoint you restore")

args = parser.parse_args()

IMAGE_SHAPE = [224, 224]
PIXEL_MEANS = tf.constant([[[102.9801, 115.9465, 122.7717]]]) / 255.0
PIXEL_STDV = tf.constant([[[0.2254, 0.224, 0.229]]])

random.seed(0)
COLOR_MAP = {}
for i in range(21):
    COLOR_MAP[i] = [random.randint(0, 255) for RGB in range(3)]

if not args.Single_test:
    images = []
    ori_images = []
    result_path = []
    image_size = []
    for image_path in os.listdir(args.Input_dir):
        full_path = os.path.join(args.Input_dir, image_path)
        if os.path.isfile(full_path):
            ori_image = cv2.imread(full_path)
        
            ori_images.append(ori_image)
            image_size.append(list(ori_image.shape[:2]))

            image = tf.image.convert_image_dtype(ori_image, tf.float32)
            image = tf.image.resize(image, IMAGE_SHAPE)
            image = (image - PIXEL_MEANS) / PIXEL_STDV
            images.append(image)

            name = args.Output_dir + os.path.splitext(image_path)[0] + '_result.jpg'
            result_path.append(name)
else:
    ori_image = cv2.imread(args.Input_image)
    image_size = ori_image.shape[:2]

    image = tf.image.convert_image_dtype(ori_image, tf.float32)
    image = tf.image.resize(image, IMAGE_SHAPE)
    image = (image - PIXEL_MEANS) / PIXEL_STDV
    images = tf.expand_dims(image, axis=0)

    result_path = args.Output_dir + os.path.splitext(args.Input_image)[0] + '_result.jpg'

images = tf.convert_to_tensor(images)

input_data = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])

retinaNet = resnet101(20, 9)
with tf.variable_scope('yolov3'):
    feature_maps, pred_class, pred_boxes = retinaNet.forward(input_data, is_training=False)
    anchors = retinaNet.generate_anchors(feature_maps)
    pred_boxes, pred_labels, pred_scores = retinaNet.predict(anchors, pred_class, pred_boxes)

saver_to_restore = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7

with tf.Session(config=config) as sess:
    saver_to_restore.restore(sess, args.Restore_path)

    boxes_, scores_, labels_ = sess.run([pred_boxes, pred_scores, pred_labels], feed_dict={input_data:images.eval()})

    if not args.Single_test:
        for i, single_boxes in enumerate(boxes_):
            ori_image = ori_images[i]

            # rescale the coordinates to the original image
            single_boxes[:, 0] = single_boxes[:, 0] * float(image_size[i][0])
            single_boxes[:, 1] = single_boxes[:, 1] * float(image_size[i][1])
            single_boxes[:, 2] = single_boxes[:, 2] * float(image_size[i][0])
            single_boxes[:, 3] = single_boxes[:, 3] * float(image_size[i][1])

            for j, box in enumerate(single_boxes):
                if np.sum(box) != 0.0:
                    ploy_rect(ori_image, box, COLOR_MAP[labels_[i][j]])
                    print("********************************")
                    print(box[0], " ", box[1], " ", box[2], " ", box[3], LABELS[labels_[i, j]], scores_[i, j])
                
            cv2.namedWindow('Detection_result', 0)
            cv2.imshow('Detection_result', ori_image)
            cv2.imwrite(result_path[i], ori_image)
            cv2.waitKey(0)
    else:
        boxes_ = boxes_[0]
        scores_ = scores_[0]
        labels_ = labels_[0]
        
        # rescale the coordinates to the original image
        boxes_[:, 0] = boxes_[:, 0] * float(image_size[0])
        boxes_[:, 1] = boxes_[:, 1] * float(image_size[1])
        boxes_[:, 2] = boxes_[:, 2] * float(image_size[0])
        boxes_[:, 3] = boxes_[:, 3] * float(image_size[1])

        for j, box in enumerate(boxes_):
            if np.sum(box) != 0.0:
                ploy_rect(ori_image, box, COLOR_MAP[labels_[j]])
                print("********************************")
                print(box[0], " ", box[1], " ", box[2], " ", box[3], LABELS[labels_[j]], scores_[j])
            
        cv2.namedWindow('Detection_result', 0)
        cv2.imshow('Detection_result', ori_image)
        cv2.imwrite(result_path, ori_image)
        cv2.waitKey(0)
