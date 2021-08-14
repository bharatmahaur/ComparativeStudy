import tensorflow as tf 
import numpy as np 
import random
import cv2

PIXEL_MEANS = np.array([[[122.7717, 115.9465, 102.9801]]])
PIXEL_STDV = [[[0.229, 0.224, 0.2254]]]

def normlize(image, mean=PIXEL_MEANS):
    image = (image - mean / 255.0) / PIXEL_STDV
    return image

def flip_left_right(image, boxes, labels):
    width = tf.cast(tf.shape(image)[1], tf.float32)
    image  = tf.image.flip_left_right(image)

    xmin = 0 - boxes[:, 2] + width
    ymin = boxes[:, 1]
    xmax = 0 - boxes[:, 0] + width
    ymax = boxes[:, 3]
    boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    return image, boxes, labels

def flip_down_up(image, boxes, labels):
    height = tf.cast(tf.shape(image)[0], tf.float32)
    image  = tf.image.flip_up_down(image)

    xmin = boxes[:, 0]
    ymin = 0 - boxes[:, 3] + height
    xmax = boxes[:, 2]
    ymax = 0 - boxes[:, 1] + height
    boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    return image, boxes, labels


def distort_color(image, boxes, labels):
    def nothing(image):
        return image 
    
    sequence = [0, 1, 2, 3]
    sequence = tf.random_shuffle(sequence)
    for i in range(4):
        image = tf.cond(tf.equal(sequence[i], 0), lambda: tf.image.random_brightness(image, max_delta=32./255), lambda: nothing(image))
        image = tf.cond(tf.equal(sequence[i], 1), lambda: tf.image.random_saturation(image, lower=0.8, upper=1.2), lambda: nothing(image))
        image = tf.cond(tf.equal(sequence[i], 2), lambda: tf.image.random_hue(image, max_delta=0.2), lambda: nothing(image)) 
        image = tf.cond(tf.equal(sequence[i], 3), lambda: tf.image.random_contrast(image, lower=0.8, upper=1.2), lambda: nothing(image))     
    return image, boxes, labels


def crop(image, boxes, labels, min_object_covered=0.5, aspect_ratio_range=[0.5, 2.0], area_range=[0.3, 1.0]):
    h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
    xmin, ymin, xmax, ymax = tf.unstack(boxes, axis=1)
    bboxes = tf.stack([ymin/h, xmin/w, ymax/h, xmax/w], axis=1)
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    begin, size, dist_boxes = tf.image.sample_distorted_bounding_box(
                                    tf.shape(image),
                                    bounding_boxes=tf.expand_dims(bboxes, axis=0),
                                    min_object_covered=min_object_covered,
                                    aspect_ratio_range=aspect_ratio_range,
                                    area_range=area_range,
                                    max_attempts=50)
    # NOTE dist_boxes with shape: [ymin, xmin, ymax, xmax] and in values in range(0, 1)
    # Employ the bounding box to distort the image.
    croped_box = [dist_boxes[0,0,1]*w, dist_boxes[0,0,0]*h, dist_boxes[0,0,3]*w, dist_boxes[0,0,2]*h]

    croped_xmin = tf.clip_by_value(xmin, croped_box[0], croped_box[2])-croped_box[0]
    croped_ymin = tf.clip_by_value(ymin, croped_box[1], croped_box[3])-croped_box[1]
    croped_xmax = tf.clip_by_value(xmax, croped_box[0], croped_box[2])-croped_box[0]
    croped_ymax = tf.clip_by_value(ymax, croped_box[1], croped_box[3])-croped_box[1]

    image = tf.slice(image, begin, size)
    boxes = tf.stack([croped_xmin, croped_ymin, croped_xmax, croped_ymax], axis=1)

    return image, boxes, labels


def resize_image_and_correct_boxes(image, boxes, labels, image_size):
    origin_image_size = tf.cast(tf.shape(image)[0:2], tf.float32)
    def w_long():
        new_w = image_size[1]
        new_h = tf.cast(origin_image_size[0] / origin_image_size[1] * image_size[1], tf.int32)
        return [new_h, new_w]

    def h_long():
        new_h = image_size[0]
        new_w = tf.cast(origin_image_size[1] / origin_image_size[0] * image_size[0], tf.int32)  
        return [new_h, new_w]

    new_size = tf.cond(tf.less(origin_image_size[0] / image_size[0], origin_image_size[1] / image_size[1]), 
                        w_long, h_long)

    image = tf.image.resize_images(image, new_size)
    offset_h = tf.cast((image_size[0] - new_size[0]) / 2, tf.int32)
    offset_w = tf.cast((image_size[1] - new_size[1]) / 2, tf.int32)
    image = tf.image.pad_to_bounding_box(image, offset_h, offset_w, image_size[0], image_size[1])
    
    # correct the boxes
    xmin = tf.clip_by_value(boxes[:, 0] / origin_image_size[1], 0.0, 1.0) * tf.cast(new_size[1], tf.float32) + tf.cast(offset_w, tf.float32)
    ymin = tf.clip_by_value(boxes[:, 1] / origin_image_size[0], 0.0, 1.0) * tf.cast(new_size[0], tf.float32) + tf.cast(offset_h, tf.float32)
    xmax = tf.clip_by_value(boxes[:, 2] / origin_image_size[1], 0.0, 1.0) * tf.cast(new_size[1], tf.float32) + tf.cast(offset_w, tf.float32)
    ymax = tf.clip_by_value(boxes[:, 3] / origin_image_size[0], 0.0, 1.0) * tf.cast(new_size[0], tf.float32) + tf.cast(offset_h, tf.float32)

    # if the object is not in the dist_box, just remove it 
    mask = tf.logical_not(tf.logical_or(tf.equal(xmin, xmax), tf.equal(ymin, ymax)))        
    xmin = tf.boolean_mask(xmin, mask)
    ymin = tf.boolean_mask(ymin, mask)
    xmax = tf.boolean_mask(xmax, mask)
    ymax = tf.boolean_mask(ymax, mask)
    labels = tf.boolean_mask(labels, mask)

    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return image, boxes, labels


def data_augmentation(image, boxes, labels):
    def nothing(image, boxes, labels):
        return image, boxes, labels

    image, boxes, labels = tf.cond(tf.equal(tf.cast(tf.random_uniform(shape=[]) * 2, tf.int64), 0), lambda: flip_left_right(image, boxes, labels), lambda: nothing(image, boxes, labels))
    image, boxes, labels = tf.cond(tf.equal(tf.cast(tf.random_uniform(shape=[]) * 2, tf.int64), 0), lambda: flip_down_up(image, boxes, labels), lambda: nothing(image, boxes, labels))
    # image, boxes, labels = tf.cond(tf.equal(tf.cast(tf.random_uniform(shape=[]) * 2, tf.int64), 0), lambda: distort_color(image, boxes, labels), lambda: nothing(image, boxes, labels))
    # image, boxes, labels = crop(image, boxes, labels)

    return image, boxes, labels

def preprocess(image, boxes, labels, image_size, mode):
    if len(image.get_shape().as_list()) != 3:
        raise ValueError('Input image must have 3 shapes H W C')
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # data augmentation for train data
    if mode == 'train':  
        image, boxes, labels = data_augmentation(image, boxes, labels)
    
    image = normlize(image)

    image, boxes, labels = resize_image_and_correct_boxes(image, boxes, labels, image_size)

    # Pad the boxes and labels to 20
    pad_num = 60 - tf.shape(boxes)[0]
    boxes = tf.pad(boxes, [[0, pad_num], [0, 0]], "CONSTANT")
    labels = tf.pad(labels, [[0, pad_num]], "CONSTANT")
    return image, boxes, labels


def parser(serialized_example, image_size, mode):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image' : tf.FixedLenFeature([], dtype = tf.string),
            'boxes' : tf.FixedLenFeature([], dtype = tf.string),
            'labels' : tf.FixedLenFeature([], dtype = tf.string),
        })

    image = features['image']
    boxes = features['boxes']
    labels = features['labels']

    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

    boxes = tf.decode_raw(boxes, tf.float32)
    boxes = tf.reshape(boxes, shape=[-1, 4])

    labels = tf.decode_raw(labels, tf.int64)
    labels = tf.reshape(labels, shape=[-1])
   
    return preprocess(image, boxes, labels, image_size, mode)