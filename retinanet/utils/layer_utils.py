import tensorflow as tf
import math 
import tensorflow.contrib.slim as slim

# How to understand this func? Is it different from slim.conv2d?????????
def conv2d(inputs, filters, kernel_size, strides=1):
    # if the stride = 2, make the size of feature map is the half of the input
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    if strides > 1: 
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def build_block(inputs, filters, first=False, max_pool=False, stride=1):
    short_cut = inputs
    if first == True:
        net = slim.conv2d(inputs, filters, 1, stride=2)
        short_cut = slim.conv2d(short_cut, filters * 4, 1, stride=2, activation_fn=None)
    else:
        net = slim.conv2d(inputs, filters, 1, stride=stride)
        if max_pool == True:
            short_cut = slim.conv2d(short_cut, filters * 4, 1, stride=1, activation_fn=None)

    net = slim.conv2d(net, filters, 3, stride=stride)
    net = slim.conv2d(net, filters * 4, 1, stride=stride, activation_fn=None)

    net = net + short_cut
    net = tf.nn.relu(net)
    return net

def resnet101_body(inputs):
    # Conv1
    with tf.variable_scope('Conv1'):
        net = slim.conv2d(inputs, 64, 7, stride=2)

    # Conv2_x
    with tf.variable_scope('Conv2_x'):
        net = slim.max_pool2d(net, 3, stride=2, padding='SAME')
        net = build_block(net, 64, max_pool=True)
        for i in range(2):
            net = build_block(net, 64, first=False)
    
    # Conv3_x
    with tf.variable_scope('Conv3_x'):
        net = build_block(net, 128, first=True)
        for i in range(3):
            net = build_block(net, 128, first=False)
    layer1 = net

    # Conv4_x
    with tf.variable_scope('Conv4_x'):
        net = build_block(net, 256, first=True)
        for i in range(22):
            net = build_block(net, 256, first=False)
    layer2 = net

    # Conv5_x
    with tf.variable_scope('Conv5_x'):
        net = build_block(net, 512, first=True)
        for i in range(2):
            net = build_block(net, 512, first=False)
    layer3 = net

    return layer1, layer2, layer3

def classNet(feature_map, num_classes, num_anchors):
    for i in range(4):
        feature_map = slim.conv2d(feature_map, 256, 3, stride=1, activation_fn=tf.nn.relu)
    output = slim.conv2d(feature_map, num_classes * num_anchors, 3, stride=1, normalizer_fn=None,
                            biases_initializer=tf.constant_initializer(-math.log((1 - 0.01) / 0.01)))

    return output

def boxNet(feature_map, num_anchors):
    for i in range(4):
        feature_map = slim.conv2d(feature_map, 256, 3, stride=1, activation_fn=tf.nn.relu)
    output = slim.conv2d(feature_map, 4 * num_anchors, 3, stride=1, normalizer_fn=None)

    return output

def resnet101_head(layer1, layer2, layer3, num_classes, num_anchors):
    inter1 = slim.conv2d(layer3, 256, 1, stride=1, normalizer_fn=None)
    feature_map5 = slim.conv2d(inter1, 256, 3, stride=1, normalizer_fn=None)

    inter1 = upsample(inter1, layer2.get_shape().as_list())
    inter2 = slim.conv2d(layer2, 256, 1, stride=1, normalizer_fn=None)
    concat1 = tf.concat([inter1, inter2], axis=-1)

    feature_map4 = slim.conv2d(concat1, 256, 3, stride=1)

    inter2 = upsample(inter2, layer1.get_shape().as_list())
    inter3 = slim.conv2d(layer1, 256, 1, stride=1, normalizer_fn=None)
    concat2 = tf.concat([inter2, inter3], axis=-1)

    feature_map3 = slim.conv2d(concat2, 256, 3, stride=1)

    feature_map6 = slim.conv2d(layer3, 256, 3, stride=2, activation_fn=tf.nn.relu)

    feature_map7 = slim.conv2d(feature_map6, 256, 3, stride=2)

    feature_maps = [feature_map3, feature_map4, feature_map5, feature_map6, feature_map7]
    
    # classNet and boxNet 
    class_pred = []
    box_pred = []
    with tf.variable_scope("classNet"):
        for level in range(5):
            grid_szie = feature_maps[level].get_shape().as_list()[1:3]
            class_output = classNet(feature_maps[level], num_classes + 1, num_anchors)
            class_output = tf.reshape(class_output, [-1, grid_szie[0] * grid_szie[1] * num_anchors, num_classes + 1])
            class_pred.append(class_output)
        class_pred = tf.concat(class_pred, axis=1)

    with tf.variable_scope("boxNet"):
        for level in range(5):
            grid_size = feature_maps[level].get_shape().as_list()[1:3]
            box_output = boxNet(feature_maps[level], num_anchors)
            box_output = tf.reshape(box_output, [-1, grid_size[0] * grid_size[1] * num_anchors, 4])
            box_pred.append(box_output)
        box_pred = tf.concat(box_pred, axis=1)

    return feature_maps, class_pred, box_pred



def upsample(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsample')
    return inputs