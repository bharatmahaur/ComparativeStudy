import tensorflow as tf 
import numpy as np
import tensorflow.contrib.slim as slim
from utils.layer_utils import resnet101_body, resnet101_head
from utils.common_utils import assign_targets_oneimg, augmentation, decode, nms, eval_OneImg

IMAGE_SHAPE = [224, 224]

class resnet101(object):

    def __init__(self, class_num, anchors_num, aspect_ratios=(2.0, 1.0, 0.5), 
                min_level=3, max_level=7, scales_per_octave=3, batch_norm_decay=0.999, reg_loss_weight=50.0):
        self.class_num = class_num
        self.anchors_num = anchors_num
        self.aspect_ratios = aspect_ratios
        self.min_level = min_level
        self.max_level = max_level
        self.scales_per_octave = scales_per_octave
        self.batch_norm_decay = batch_norm_decay
        self.reg_loss_weight = reg_loss_weight
    
    def forward(self, inputs, is_training=False, reuse=False):
        """
        The Inference of the retinaNet 

        Args: 
            The images [batch_size, H, W, C]

        Returns: 
            Feature_maps, class_pred, box_pred. feature_maps is a list and class_pred is [batch_size, anchors, num_class+1]
                 box_pred is [batch_size, anchors, 4]
        """
        # The input img_size, form: [height, weight]
        self.img_size = inputs.get_shape().as_list()[1:3]

        '''
        method1: resnet101, designed by myself, the problem is that it is not finetuning
        '''
        # Set the batch norm params
        batch_norm_param = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'fused': None,
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_param,                 
                                weights_initializer=tf.random_normal_initializer(stddev=0.01),
                                biases_initializer=tf.zeros_initializer(),
                                activation_fn=tf.nn.relu):
                with tf.variable_scope('resnet_body'):
                    layer1, layer2, layer3 = resnet101_body(inputs) 

                with tf.variable_scope('resnet_head'):
                    with slim.arg_scope([slim.conv2d], 
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_param,                        
                            activation_fn=None,
                            weights_initializer=tf.random_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            biases_initializer=tf.zeros_initializer()):                          
                        feature_maps, class_pred, box_pred = resnet101_head(layer1, layer2, layer3, self.class_num, self.anchors_num)
               
        return feature_maps, class_pred, box_pred
        
    def generate_anchors(self, feature_maps):
        """
        To generate the anchors

        Args: 
            [P3, P4, P5, P6, P7]

        Returns: 
            The anchors [N, 9, 4] and the structure is [ymin, xmin, ymax, xmax]
        """
        anchors_list = []
        for i, feature_map in enumerate(feature_maps):
            level = i + 3
            base_size = [2 ** level * 4, 2 ** level * 4]
            stride = [2 ** level, 2 ** level]
            grid_size = feature_map.get_shape().as_list()[1:3]

            # W / H = octave_scale
            octave_scale = [2 ** (float(scale) / self.scales_per_octave) for scale in range(self.scales_per_octave)]

            # Use np.arrary not for to create the anchor height and width, considering the speed 
            octave_grid, ratio_grid = np.meshgrid(octave_scale, self.aspect_ratios)
            octave_grid = np.reshape(octave_grid, -1)
            ratio_grid = np.reshape(ratio_grid, -1)      

            anchors_height = base_size[0] * octave_grid / np.sqrt(ratio_grid)
            anchors_width = base_size[1] * octave_grid * np.sqrt(ratio_grid)

            # Get a grid of box centers
            grid_x = np.arange(0, grid_size[1], 1, np.float32)
            grid_y = np.arange(0, grid_size[0], 1, np.float32)
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            
            # # If img_size % stride == 0, give the offset of 0.5(P3, P4, P5), else give the offset of 0(P6, P7) 
            # if (level < 6):
            #     x_centers = (grid_x + 0.5) * stride[1]
            #     y_centers = (grid_y + 0.5) * stride[0]
            # else:
            #     x_centers = grid_x * stride[1]
            #     y_centers = grid_y * stride[0]
            x_centers = (grid_x + 0.5) * stride[1]
            y_centers = (grid_y + 0.5) * stride[0]
            
            # Normalized 
            x_centers, anchors_width = x_centers / self.img_size[1], anchors_width / self.img_size[1]
            y_centers, anchors_height = y_centers / self.img_size[0], anchors_height / self.img_size[0]

            # Concat the x,y,h,w
            anchors_width, x_centers = np.meshgrid(anchors_width, x_centers)
            anchors_height, y_centers = np.meshgrid(anchors_height, y_centers)

            anchors = np.stack([x_centers, y_centers, anchors_width, anchors_height], axis=-1)
            ymin = anchors[:, :, 1] - 0.5 * anchors[:, :, 3]
            xmin = anchors[:, :, 0] - 0.5 * anchors[:, :, 2]
            ymax = anchors[:, :, 1] + 0.5 * anchors[:, :, 3]
            xmax = anchors[:, :, 0] + 0.5 * anchors[:, :, 2]
            anchors = np.stack([ymin, xmin, ymax, xmax], axis=-1)
            anchors_list.append(anchors)
        
        anchors = tf.cast(tf.concat(anchors_list, axis=0), tf.float32)
        return anchors
   
    def predict(self, anchors, pred_class, pred_boxes):
        batch_size = tf.shape(pred_class)[0]
        anchors = augmentation(anchors, batch_size)
        anchors = tf.reshape(anchors, [batch_size, -1, 4])

        detect_boxes = decode(anchors, pred_boxes)
        detect_scores = tf.sigmoid(pred_class)

        map_out = tf.map_fn(lambda x: nms(x[0], x[1], self.class_num), [detect_boxes, detect_scores],
                                          dtype=[tf.float32, tf.int64, tf.float32])
        with tf.variable_scope("predict_output"):
            pred_boxes = map_out[0]
            pred_labels = map_out[1]
            pred_scores = map_out[2]
        return pred_boxes, pred_labels, pred_scores
        
    def assign_targets(self, anchors, gt_boxes, gt_labels):
        """
        Assign gt targets
        Inputs:
            anchors: 3-D tensor of shape [1049, 9, 4]
            gt_boxes: 3-D tensor of shape [batch_size, 60, 4] containing coordinates of gt boxes
            gt_labels: 3-D tensors of shape [batch_size, 60] containing gt classes
        Returns:
            batch_cls_targets: class tensor with shape [batch_size, num_anchors, num_classes]
            batch_reg_target: box tensor with shape [batch_size, num_anchors, 4]
        """
        batch_size = tf.shape(gt_labels)[0]
        anchors = augmentation(anchors, batch_size)

        # Rescale the boxes 
        ymin = gt_boxes[:, :, 0] / self.img_size[0]
        xmin = gt_boxes[:, :, 1] / self.img_size[1]
        ymax = gt_boxes[:, :, 2] / self.img_size[0]
        xmax = gt_boxes[:, :, 3] / self.img_size[1]
        gt_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

        map_output = tf.map_fn(lambda x: assign_targets_oneimg(x[0], x[1], x[2], self.class_num, self.anchors_num),
                               [anchors, gt_boxes, gt_labels],
                               dtype=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

        target_class = map_output[0]
        target_reg = map_output[1]
        target_mask = map_output[2]
        background_mask = map_output[3]
        ignore_mask = map_output[4]

        return target_class, target_reg, target_mask, background_mask, ignore_mask

    def loss(self, pred_class, true_class, pred_boxes, true_boxes, target_mask, background_mask, alpha=0.25, gamma=2.0):
        with tf.variable_scope("loss"):
            with tf.variable_scope("focal_loss"):
                pos_anchors = tf.maximum(1.0, tf.reduce_sum(target_mask))

                target_mask = tf.expand_dims(target_mask, axis=-1)
                background_mask = tf.expand_dims(background_mask, axis=-1)

                pred_class = tf.cast(pred_class, tf.float32)
                true_class = tf.cast(true_class, tf.float32)

                prediction = tf.sigmoid(pred_class)
                prediction_t = tf.where(tf.equal(true_class, 1.0), prediction, 1.0 - prediction)
                alpha = tf.ones_like(true_class) * alpha
                alpha_t = tf.where(tf.equal(true_class, 1.0), alpha, 1.0 - alpha)

                FL = -alpha_t * tf.pow(1 - prediction_t, gamma) * tf.log(prediction_t)
                FL = FL * target_mask + FL * background_mask
                FL_mean = tf.reduce_sum(FL) / pos_anchors

            with tf.variable_scope("regression_loss"):
                delta_square = 9.0
                reg_diff = tf.abs(pred_boxes - true_boxes)
                reg_loss = tf.where(tf.less(reg_diff, 1.0 / delta_square), 0.5 * tf.pow(reg_diff, 2) * delta_square, reg_diff - 0.5 / delta_square)

                reg_loss = reg_loss * target_mask           
                reg_loss_mean = tf.reduce_sum(reg_loss) / pos_anchors

            regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            total_loss = FL_mean + reg_loss_mean + regularization_loss

        return total_loss, FL_mean, reg_loss_mean
    
    def compute_loss(self, anchors, pred_class, pred_boxes, labels, boxes):
        target_class, target_reg, target_mask, background_mask, ignore_mask = self.assign_targets(anchors, boxes, labels)
        loss = self.loss(pred_class, target_class, pred_boxes, target_reg, target_mask, background_mask)
        
        return loss

    def evaluate(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        # This function is to calculate the number of TP, pred_label and gt_label
        # Rescale the boxes 
        ymin = gt_boxes[:, :, 0] / self.img_size[0]
        xmin = gt_boxes[:, :, 1] / self.img_size[1]
        ymax = gt_boxes[:, :, 2] / self.img_size[0]
        xmax = gt_boxes[:, :, 3] / self.img_size[1]
        gt_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

        TP_array, pred_label_array, gt_label_array = tf.map_fn(lambda x: eval_OneImg(x[0], x[1], x[2], x[3], x[4], self.class_num), 
                                                                [pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels],
                                                                dtype=[tf.int32, tf.int32, tf.int32])
    
        TP_array = tf.reduce_sum(TP_array, axis=0)
        pred_label_array = tf.reduce_sum(pred_label_array, axis=0)
        gt_label_array = tf.reduce_sum(gt_label_array, axis=0)

        return TP_array, pred_label_array, gt_label_array
            
