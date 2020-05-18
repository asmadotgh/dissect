import tensorflow as tf
import sys
import os
from classifier.layer import *
from tensorflow.layers import flatten
import pdb


def batch_norm_layer(x, is_training, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                        is_training=is_training, scope=name)


def dense_layer(x, layer_size, isTrain, scope, dropout_p=0.8):
    with tf.name_scope(scope):
        x = batch_norm_layer(x, is_training=isTrain, name=scope + '_batch')
        # print("bn: ", x)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, layer_size, name=scope + '_dense')
        x = tf.layers.dropout(x, dropout_p, isTrain)
        # print("dense: ", x)
        return x


def pretrained_classifier(inputae, n_label, reuse, name='SimpleClassifier', isTrain=False):
    print(name, isTrain)
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # input: [n, 64, 64, 3]
        print(inputae)
        inputs = tf.layers.flatten(inputae)
        print(inputs)
        inputs = dense_layer(inputs, 32, isTrain, 'SCBlock1')  # [n, 32]
        print(inputs)
        inputs = dense_layer(inputs, 32, isTrain, 'SCBlock2')  # [n, 32]
        print(inputs)
        logit = dense_layer(inputs, n_label, isTrain, 'SCBlock3')  # [n, n_label]
        print(logit)
        if isTrain == False:
            # print(isTrain)
            logit = tf.stop_gradient(logit)
        prediction = tf.nn.sigmoid(logit)
        # pred_y = tf.argmax(prediction, 1)
        return logit, prediction
