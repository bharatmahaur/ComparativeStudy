from tensorflow.core.framework import summary_pb2
import tensorflow as tf 
import numpy as np

BBOX_XFORM_CLIP = np.log(1000. / 16.)

def label_count(TP, pred_label, gt_label, class_num):
    TP_array = np.zeros(shape=[class_num + 1], dtype=np.int32)
    pred_label_array = np.zeros(shape=[class_num + 1], dtype=np.int32)
    gt_label_array = np.zeros(shape=[class_num + 1], dtype=np.int32)

    # If can use better method insted of "for" 
    for val in TP:
        TP_array[val] += 1

    for val in pred_label:
        pred_label_array[val] += 1

    for val in gt_label:
        gt_label_array[val] += 1

    TP_array = TP_array[1:]
    pred_label_array = pred_label_array[1:]
    gt_label_array = gt_label_array[1:]

    return TP_array, pred_label_array, gt_label_array

def calc_IOU(anchors, gt_boxes):
    """
    Maintain an efficient way to calculate the ios matrix 
    between ground truth true boxes and the anchors
    [ymin, xmin, ymax, xmax]
    anchors: [1049, 9, 4]
    gt_boxes: [V, 4]
    """
    anchors_yx_min = anchors[:, :, 0:2]
    anchors_yx_max = anchors[:, :, 2:4]

    gt_yx_min = gt_boxes[:, 0:2]
    gt_yx_max = gt_boxes[:, 2:4]

    # [1049, 9, 2] => [1049, 9, 1, 2]
    anchors_yx_min = tf.expand_dims(anchors_yx_min, axis=-2)
    anchors_yx_max = tf.expand_dims(anchors_yx_max, axis=-2)

    # [V, 2] => [1, V, 2]
    gt_yx_min = tf.expand_dims(gt_yx_min, axis=0)
    gt_yx_max = tf.expand_dims(gt_yx_max, axis=0)

    # BroadCast [1049, 9, 1, 2] & [1, V, 2] => [1049, 9, V, 2] 
    intersect_mins = tf.maximum(anchors_yx_min, gt_yx_min)
    intersect_maxs = tf.minimum(anchors_yx_max, gt_yx_max)
    intersect_hw = tf.maximum(intersect_maxs - intersect_mins, 0.0)
   
    # [1049, 9, V]
    intersect_area = intersect_hw[..., 0] * intersect_hw[..., 1]

    # [1049, 9, 1]
    anchors_hw = anchors_yx_max - anchors_yx_min 
    anchors_area = anchors_hw[..., 0] * anchors_hw[..., 1]

    # [1, V]
    gt_hw = gt_yx_max - gt_yx_min
    gt_area = gt_hw[..., 0] * gt_hw[..., 1]

    # [1049, 9, V]
    iou = intersect_area / (anchors_area + gt_area - intersect_area + 1e-10)
    
    return iou

def encode(best_index, anchors, gt_boxes, gt_labels, weights=[10., 10., 5., 5.]):
    # Only np.array can do this op!
    # gt_labels: [V], best_index: [1049, 9] => target_class: [1049, 9] 
    # gt_boxes: [V, 4], best_index: [1049, 9] => target_boxes: [1049, 9, 4]
    best_index = np.array(best_index)
    gt_boxes = np.array(gt_boxes)
    gt_labels = np.array(gt_labels) 

    target_class = gt_labels[best_index]
    target_boxes = gt_boxes[best_index]

    target_xcenter = (target_boxes[:, :, 1] + target_boxes[:, :, 3]) / 2.0
    target_ycenter = (target_boxes[:, :, 0] + target_boxes[:, :, 2]) / 2.0
    target_w = target_boxes[:, :, 3] - target_boxes[:, :, 1]
    target_h = target_boxes[:, :, 2] - target_boxes[:, :, 0]
    
    anchors_xcenter = (anchors[:, :, 1] + anchors[:, :, 3]) / 2.0
    anchors_ycenter = (anchors[:, :, 0] + anchors[:, :, 2]) / 2.0
    anchors_w = anchors[:, :, 3] - anchors[:, :, 1]
    anchors_h = anchors[:, :, 2] - anchors[:, :, 0]

    wx, wy, ww, wh = weights
    tx = wx * (target_xcenter - anchors_xcenter) / anchors_w
    ty = wy * (target_ycenter - anchors_ycenter) / anchors_h
    tw = ww * np.log(target_w / anchors_w)
    th = wh * np.log(target_h / anchors_h)

    target_reg = np.stack([tx, ty, tw, th], axis=-1)

    return target_class, target_reg

def decode(anchors, pred_boxes, weights=[10., 10., 5., 5.]):
    anchors_xcenter = (anchors[:, :, 1] + anchors[:, :, 3]) / 2.0
    anchors_ycenter = (anchors[:, :, 0] + anchors[:, :, 2]) / 2.0
    anchors_w = anchors[:, :, 3] - anchors[:, :, 1]
    anchors_h = anchors[:, :, 2] - anchors[:, :, 0]

    wx, wy, ww, wh = weights
    tx = pred_boxes[:, :, 0] / wx
    ty = pred_boxes[:, :, 1] / wy
    tw = pred_boxes[:, :, 2] / ww
    th = pred_boxes[:, :, 3] / wh

    # Prevent sending too large values into tf.exp
    tw = tf.minimum(tw, BBOX_XFORM_CLIP)
    th = tf.minimum(th, BBOX_XFORM_CLIP)
    
    detect_xcenter = tx * anchors_w + anchors_xcenter
    detect_ycenter = ty * anchors_h + anchors_ycenter
    detect_w = tf.exp(tw) * anchors_w
    detect_h = tf.exp(th) * anchors_h

    detect_ymin = detect_ycenter - 0.5 * detect_h
    detect_xmin = detect_xcenter - 0.5 * detect_w
    detect_ymax = detect_ycenter + 0.5 * detect_h
    detect_xmax = detect_xcenter + 0.5 * detect_w

    detect_boxes = tf.stack([detect_ymin, detect_xmin, detect_ymax, detect_xmax], axis=-1)

    return detect_boxes

def augmentation(anchors, batch_size):
    """
    You can use tf.tile or tf.meshgrid to augment the anchors! 
    """
    anchors = tf.expand_dims(anchors, axis=0)
    anchors = tf.tile(anchors, (batch_size, 1, 1, 1))
    return anchors

def assign_targets_oneimg(anchors, gt_boxes, gt_labels, num_class, num_anchors):
    """
    Assign the targets in one image
    """
    # Remove the padded boxes [60, 4] => [V, 4]
    mask = tf.not_equal(gt_boxes, 0.0)
    mask_float = tf.cast(mask, tf.float32)
    mask = tf.not_equal(tf.reduce_sum(mask_float, axis=-1), 0.0)
    gt_boxes = tf.boolean_mask(gt_boxes, mask, axis=0)
    gt_labels = tf.boolean_mask(gt_labels, mask, axis=0)

    # Avoid the xmin=xmax, ymin=ymax
    mask = tf.logical_not(tf.logical_or(tf.equal(gt_boxes[:, 0], gt_boxes[:, 2]), tf.equal(gt_boxes[:, 1], gt_boxes[:, 3])))
    gt_boxes = tf.boolean_mask(gt_boxes, mask, axis=0)
    gt_labels = tf.boolean_mask(gt_labels, mask, axis=0)

    # [1049, 9, V]
    iou = calc_IOU(anchors, gt_boxes)

    # [1049, 9]
    best_iou = tf.reduce_max(iou, axis=-1)
    best_index = tf.argmax(iou, axis=-1)

    target_mask = tf.greater_equal(best_iou, 0.5)
    background_mask = tf.less(best_iou, 0.4)
    ignore_mask = tf.logical_and(tf.logical_not(target_mask), tf.logical_not(background_mask))

    target_mask = tf.cast(target_mask, tf.float32)
    background_mask = tf.cast(background_mask, tf.float32)
    ignore_mask = tf.cast(ignore_mask, tf.float32)

    # [1049, 9]
    target_class, target_reg = tf.py_function(encode, [best_index, anchors, gt_boxes, gt_labels], [tf.int64, tf.float32])
    target_class = tf.reshape(target_class, (tf.shape(anchors)[0], tf.shape(anchors)[1]))
    target_reg = tf.reshape(target_reg, (tf.shape(anchors)[0], tf.shape(anchors)[1], 4))

    target_class = target_class * tf.cast(target_mask, tf.int64)
    target_reg = target_reg * tf.expand_dims(target_mask, axis=-1)

    # [1049, 9, num_class + 1]
    target_class = tf.one_hot(target_class, num_class + 1)

    target_class = tf.reshape(target_class, [-1, num_class + 1])
    target_reg = tf.reshape(target_reg, [-1, 4])
    target_mask = tf.reshape(target_mask, [-1])
    background_mask = tf.reshape(background_mask, [-1])
    ignore_mask = tf.reshape(ignore_mask, [-1])

    return [target_class, target_reg, target_mask, background_mask, ignore_mask]

def nms(detect_boxes, detect_scores, class_num, score_threshold=0.5, iou_thresh=0.5, max_boxes=50):
    boxes_list, labels_list, scores_list = [], [], []
    mask = tf.greater(detect_scores, score_threshold)
    for i in range(1, class_num + 1):
        filter_boxes = tf.boolean_mask(detect_boxes, mask[:, i])
        filter_scores = tf.boolean_mask(detect_scores[:, i], mask[:, i])

        # If it is essential to get tok 1000 to speed up ???????

        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_scores,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=iou_thresh)
        labels_list.append(tf.ones_like(tf.gather(filter_scores, nms_indices), tf.int64) * i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        scores_list.append(tf.gather(filter_scores, nms_indices))

    with tf.variable_scope("nms_output"):
        labels = tf.concat(labels_list, axis=0)
        scores = tf.concat(scores_list, axis=0)
        boxes = tf.concat(boxes_list, axis=0)

        padnum = 1000 - tf.shape(labels)[0]
        labels = tf.pad(labels, [[0, padnum]], "CONSTANT")
        scores = tf.pad(scores, [[0, padnum]], "CONSTANT")
        boxes = tf.pad(boxes, [[0, padnum], [0, 0]], "CONSTANT")

    return [boxes, labels, scores]

def eval_OneImg(pred_box, pred_label, pred_score, gt_box, gt_label, class_num, iou_thresh=0.5):
    # Remove the padded boxes and label
    remain_mask = tf.cast(tf.not_equal(pred_box, 0.0), tf.float32)
    remain_mask = tf.not_equal(tf.reduce_sum(remain_mask, axis=-1), 0.0)
    pred_box = tf.boolean_mask(pred_box, remain_mask)
    pred_label = tf.boolean_mask(pred_label, remain_mask)
    pred_score = tf.boolean_mask(pred_score, remain_mask)

    remain_mask = tf.cast(tf.not_equal(gt_box, 0.0), tf.float32)
    remain_mask = tf.not_equal(tf.reduce_sum(remain_mask, axis=-1), 0.0)
    gt_box = tf.boolean_mask(gt_box, remain_mask)
    gt_label = tf.boolean_mask(gt_label, remain_mask)

    # pred_box: [M, 4] pred_label: [M] pred_score: [M]
    # gt_box: [N, 4] gt_label: [N]

    # pred_box => [M, 1, 4] gt_box => [1, N, 4]
    pred_box = tf.expand_dims(pred_box, axis=1)
    gt_box = tf.expand_dims(gt_box, axis=0)

    pred_yxmin = pred_box[:, :, 0:2]
    pred_yxmax = pred_box[:, :, 2:4]

    gt_yxmin = gt_box[:, :, 0:2]
    gt_yxmax = gt_box[:, :, 2:4]

    # => [M, N, 2]
    intersect_mins = tf.maximum(pred_yxmin, gt_yxmin)
    intersect_maxs = tf.minimum(pred_yxmax, gt_yxmax)
    intersect_hw = tf.maximum(intersect_maxs - intersect_mins, 0.0)

    # [M, N]
    intersect_area = intersect_hw[:, :, 0] * intersect_hw[:, :, 1]

    # [M, 1]
    pred_hw = pred_yxmax - pred_yxmin
    pred_area = pred_hw[:, :, 0] * pred_hw[:, :, 1]

    # [1, N]
    gt_hw = gt_yxmax - gt_yxmin
    gt_area = gt_hw[:, :, 0] * gt_hw[:, :, 1]

    # [M, N]
    iou = intersect_area / (pred_area + gt_area - intersect_area + 1e-10)

    # [M]
    best_iou = tf.reduce_max(iou, axis=-1)
    best_index = tf.argmax(iou, axis=-1)

    iou_mask = tf.greater(best_iou, iou_thresh)

    # Get the label of the match gt_box, shape [M]
    pred_to_gt_label = tf.gather(gt_label, best_index)
    label_mask = tf.equal(pred_to_gt_label, pred_label)

    TP_mask = tf.logical_and(iou_mask, label_mask)
    TP = tf.boolean_mask(pred_label, TP_mask)

    TP_array, pred_label_array, gt_label_array = tf.py_function(label_count, inp=[TP, pred_label, gt_label, class_num], 
                                                    Tout=[tf.int32, tf.int32, tf.int32])
    TP_array = tf.reshape(TP_array, [class_num])
    pred_label_array = tf.reshape(pred_label_array, [class_num])
    gt_label_array = tf.reshape(gt_label_array, [class_num])

    return [TP_array, pred_label_array, gt_label_array]

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])