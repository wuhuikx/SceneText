import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from nets import resnet_v1

tf.app.flags.DEFINE_integer('text_scale', 512, '')
FLAGS = tf.app.flags.FLAGS

def model(images, weight_decay=1e-5, is_training=True):

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]

            for i in range(4):
                print('Shape of f_{} = {}'.format(i, f[i].shape))

            h = [f[0], None, None, None]
            num_outputs = [None, 128, 64, 32]

            def unpool(data):
                return tf.image.resize_bilinear(data,
                        size=[tf.shape(data)[1]*2, tf.shape(data)[2]*2])

            def feature_merge(data, d_concat, num_output):
                concat_res = tf.concat([unpool(data), d_concat], axis=-1)
                conv1x1_res = slim.conv2d(concat_res, num_output, 1)
                conv3x3_res = slim.conv2d(conv1x1_res, num_output, 3)
                return conv3x3_res
            
            for i in range(1,4):
                h[i] = feature_merge(h[i-1], f[i], num_outputs[i])
            
            feature = slim.conv2d(h[3], 32, 3)
            F_score = slim.conv2d(feature, 1, 1,
                    activation_fn=tf.nn.sigmoid,
                    normalizer_fn=None)

            geo_map = slim.conv2d(feature, 4, 1,
                    activation_fn=tf.nn.sigmoid,
                    normalizer_fn=None) * FLAGS.text_scale
            angle_map = slim.conv2d(feature, 1, 1,
                    activation_fn=tf.nn.sigmoid,
                    normalizer_fn=None)
            angle_map = (angle_map - 0.5) * np.pi/2
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry



