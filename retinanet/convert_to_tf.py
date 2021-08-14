import os
import numpy as np
import tensorflow as tf
import cv2
import argparse
import shutil
import xml.etree.cElementTree as ET

"""
We divide the image into train, val and test.
train 60%
val 20%
test 20%
"""
# define clasify: background and car
LABELS = {
    'person': (1, 'Person'),
    'bird': (2, 'Bird'),
    'cat': (3, 'Cat'),
    'cow': (4, 'Cow'),
    'dog': (5, 'Dog'),
    'horse': (6, 'Horse'),
    'sheep': (7, 'Sheep'),
    'aeroplane': (8, 'Aeroplane'),
    'bicycle': (9, 'Bicycle'),
    'boat': (10, 'Boat'),
    'bus': (11, 'Bus'),
    'car': (12, 'Car'),
    'motorbike': (13, 'Motorbike'),
    'train': (14, 'Train'),
    'bottle': (15, 'Bottle'),
    'chair': (16, 'Chair'),
    'diningtable': (17, 'Diningtable'),
    'pottedplant': (18, 'Pottedplant'),
    'sofa': (19, 'Sofa'),
    'tvmonitor': (20, 'TV')
}

parser = argparse.ArgumentParser(description="Convert the images and labels to tfrecords")

parser.add_argument("--image_path", type=str, default="/home/ley/Documents/VOCdevkit/VOC2012/JPEGImages", help="The path of your images")

parser.add_argument("--annotation_path", type=str, default="/home/ley/Documents/VOCdevkit/VOC2012/Annotations", help="The path of your annotations")

parser.add_argument("--train_tf", type=str, default="./tfrecords/voc2012/train", help="The path of your train tfrecords")

parser.add_argument("--val_tf", type=str, default="./tfrecords/voc2012/val.tfrecords", help="The path of your val tfrecords")

parser.add_argument("--test_tf", type=str, default="./tfrecords/voc2012/test.tfrecords", help="The path of your test tfrecords")

args = parser.parse_args()

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def read_file(image_path, annotation_path):
    image_name = []
    annotation_name = []
    for JPEG_name in os.listdir(image_path):
        JPEG_path = os.path.join(image_path, JPEG_name)
        xml_name = os.path.splitext(JPEG_name)[0] + ".xml"
        xml_path = os.path.join(annotation_path, xml_name)
        
        if os.path.exists(xml_path):
            image_name.append(JPEG_path)
            annotation_name.append(xml_path)

    return image_name, annotation_name

def get_info(annotation):
    boxes = []
    labels = []

    tree = ET.parse(annotation)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text

        labels.append((int(LABELS[name][0])))
        boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])

    return labels, boxes

def convert_to_TF(sess, image, boxes, labels):
    boxes = np.array(boxes, dtype=np.float32).tostring()
    labels = np.array(labels, dtype=np.int64).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image),
        'labels': _bytes_feature(labels),
        'boxes' : _bytes_feature(boxes),
    }))
    return example

def main():
    with tf.Session() as sess:
        image_name, annotation_name = read_file(args.image_path, args.annotation_path)
        total_num = len(image_name)
        each_num = total_num // 5
        for i, image in enumerate(image_name):
            if i % each_num == 0 and i < total_num * 0.6:
                train_tf = args.train_tf + '-' + str(i // each_num) + '.tfrecords'
                writer = tf.python_io.TFRecordWriter(train_tf)

            if i == total_num * 0.6: 
                print('Now convert val tfrecords')
                writer = tf.python_io.TFRecordWriter(args.val_tf) 

            if i == total_num * 0.8: 
                print('Now convert test tfrecords')
                writer = tf.python_io.TFRecordWriter(args.test_tf) 

            if i >= total_num * 0.8:
                shutil.copy(image_name[i], os.path.join('./test/image', os.path.basename(image_name[i])))
                shutil.copy(annotation_name[i], os.path.join('./test/annotation', os.path.basename(annotation_name[i])))
                
            image = tf.gfile.FastGFile(image_name[i], 'rb').read()
            labels, boxes = get_info(annotation_name[i])
            example = convert_to_TF(sess, image, boxes, labels)
            writer.write(example.SerializeToString())
            print('Convert TFRecord: %i' % i)
        writer.close()


if __name__ == "__main__":
    main()