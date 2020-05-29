import tensorflow as tf
import sys
import os
from classifier.layer import *
from tensorflow.layers import flatten
import pdb
from explainer.ops import dense, relu, global_sum_pooling, D_Resblock, D_FirstResblock

def classifier_realistic_64(inputs, n_label, name='realistic_classifier', isTrain=False):
    print(name, isTrain)
    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
        # input: [n, 64, 64, 3]
        inputs = tf.layers.flatten(inputs)
        print(inputs)
        inputs = dense('RealisticBlock1', inputs, 32)  # [n, 32]
        inputs = relu(inputs)
        print(inputs)
        inputs = dense('RealisticBlock2', inputs, 32,)  # [n, 32]
        inputs = relu(inputs)
        print(inputs)
        inputs = dense('RealisticBlock3', inputs, n_label)  # [n, n_label]
        logit = relu(inputs)
        print(logit)
        if isTrain == False:
            logit = tf.stop_gradient(logit)
        prediction = tf.nn.softmax(logit)
        return logit, prediction


def classifier_distinct_64(inputs, num_dims, name='distinct_classifier', isTrain=False):
    with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
        # input: [n, 64, 64, 6]
        print(inputs)
        inputs = D_FirstResblock("DistinctResBlock1", inputs, 64, None, is_down=True)  # [n, 32, 32, 64]
        print(inputs)
        inputs = D_Resblock("DistinctResBlock2", inputs, 128, None, is_down=True)  # [n, 16, 16, 128]
        print(inputs)
        inputs = relu(inputs)
        print(inputs)  # [n, 16, 16, 128]
        inputs = global_sum_pooling(inputs)  # [n, 128]
        print(inputs)
        logit = dense("DistinctDense", inputs, num_dims, None, is_sn=True)  # [n, num_dims]
        print(logit)
        if isTrain == False:
            logit = tf.stop_gradient(logit)
        prediction = tf.nn.sigmoid(logit)
        return logit, prediction
