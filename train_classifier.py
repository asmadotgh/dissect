import numpy as np
import pandas as pd
import sys
import os
import pdb
import yaml
import tensorflow as tf
from classifier.DenseNet import pretrained_classifier as celeba_classifier
from classifier.SimpleNet import pretrained_classifier as shapes_classifier
import argparse
import warnings
from data_loader.data_loader import CelebALoader, ShapesLoader
from utils import calc_accuracy, read_data_file

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


def train(config_path, overwrite_output_dir=None, overwrite_training_images=None, overwrite_training_labels=None):
    # TODO refactor to use ArrayLoader for loading image arrays instead of from file
    config = yaml.load(open(config_path))
    print(config)

    # ============= Experiment Folder=============
    if overwrite_output_dir is not None:
        output_dir = overwrite_output_dir
    else:
        output_dir = os.path.join(config['log_dir'], config['name'])
    try:
        os.makedirs(output_dir)
    except:
        pass
    try:
        os.makedirs(os.path.join(output_dir, 'logs'))
    except:
        pass
    # ============= Experiment Parameters =============
    if overwrite_training_labels is not None and overwrite_training_images is not None:
        OVERWRITE_TRAINING = True
    else:
        OVERWRITE_TRAINING = False
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    channels = config['num_channel']
    input_size = config['input_size']
    N_CLASSES = config['num_class']
    ckpt_dir_continue = config['ckpt_dir_continue']
    dataset = config['dataset']
    if dataset == 'CelebA':
        pretrained_classifier = celeba_classifier
        my_data_loader = CelebALoader(input_size=128)
    elif dataset == 'shapes':
        pretrained_classifier = shapes_classifier
        my_data_loader = ShapesLoader()
    elif dataset == 'CelebA64' or dataset == 'dermatology':
        pretrained_classifier = celeba_classifier
        my_data_loader = CelebALoader(input_size=64)
    if ckpt_dir_continue == '':
        continue_train = False
    else:
        continue_train = True
    # ============= Data =============
    try:
        categories, file_names_dict = read_data_file(config['image_label_dict'])
    except:
        print("Problem in reading input data file : ", config['image_label_dict'])
        sys.exit()
    data_train = np.load(config['train'])
    data_test = np.load(config['test'])
    print("The classification categories are: ")
    print(categories)
    print('The size of the training set: ', data_train.shape[0])
    print('The size of the testing set: ', data_test.shape[0])
    fp = open(os.path.join(output_dir, 'setting.txt'), 'w')
    fp.write('config_file:' + str(config_path) + '\n')
    fp.close()
    # ============= placeholder =============
    with tf.name_scope('input'):
        x_ = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x-input')
        y_ = tf.placeholder(tf.int64, [None, N_CLASSES], name='y-input')
        isTrain = tf.placeholder(tf.bool)
    # ============= Model =============
    if N_CLASSES == 1:
        y = tf.reshape(y_, [-1])
        y = tf.one_hot(y, 2, on_value=1.0, off_value=0.0, axis=-1)
        logit, prediction = pretrained_classifier(x_, n_label=2, reuse=False, name='classifier', isTrain=isTrain)
    else:
        logit, prediction = pretrained_classifier(x_, n_label=N_CLASSES, reuse=False, name='classifier',
                                                  isTrain=isTrain)
        y = y_
    classif_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logit) # TODO use softmax for binary classification
    classif_acc = calc_accuracy(prediction=prediction, labels=y)
    loss = tf.losses.get_total_loss()
    # ============= Optimization functions =============    
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
    # ============= summary =============    
    cls_loss = tf.summary.scalar('classif_loss', classif_loss)
    total_loss = tf.summary.scalar('total_loss', loss)
    cls_acc = tf.summary.scalar('classif_acc', classif_acc)
    sum_train = tf.summary.merge([cls_loss, total_loss, cls_acc])
    # ============= Variables =============
    # Note that this list of variables only include the weights and biases in the model.
    lst_vars = []
    for v in tf.global_variables():
        lst_vars.append(v)
    # ============= Session =============
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(var_list=lst_vars)
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter(output_dir + '/train', sess.graph)
    writer_test = tf.summary.FileWriter(output_dir + '/test', sess.graph)
    # ============= Checkpoints =============
    if continue_train:
        print("Before training, Load checkpoint ")
        print("Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(ckpt_dir_continue)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_dir_continue, ckpt_name))
            print(ckpt_name)
            print("Successful checkpoint upload")
        else:
            print("Failed checkpoint load")
            sys.exit()
    # ============= Training =============
    train_loss = []
    test_loss = []
    itr_train = 0
    itr_test = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        if OVERWRITE_TRAINING:
            perm = np.arange(overwrite_training_images.shape[0])
            np.random.shuffle(perm)
            training_images = overwrite_training_images[perm]
            training_labels = overwrite_training_labels[perm]
            num_batch = int(overwrite_training_images.shape[0] / BATCH_SIZE)
        else:
            perm = np.arange(data_train.shape[0])
            np.random.shuffle(perm)
            data_train = data_train[perm]
            num_batch = int(data_train.shape[0] / BATCH_SIZE)
        for i in range(0, num_batch):
            start = i * BATCH_SIZE
            if OVERWRITE_TRAINING:
                xs = training_images[start:start + BATCH_SIZE]
                ys = training_labels[start:start + BATCH_SIZE]
            else:
                ns = data_train[start:start + BATCH_SIZE]
                xs, ys = my_data_loader.load_images_and_labels(ns, image_dir=config['image_dir'], n_class=N_CLASSES,
                                                               file_names_dict=file_names_dict,
                                                               num_channel=channels, do_center_crop=True)
            [_, _loss, summary_str] = sess.run([train_step, loss, sum_train], feed_dict={x_: xs, isTrain: True, y_: ys})
            writer.add_summary(summary_str, itr_train)
            itr_train += 1
            total_loss += _loss
        total_loss /= i
        print("Epoch: " + str(epoch) + " loss: " + str(total_loss) + '\n')
        train_loss.append(total_loss)

        total_loss = 0.0
        perm = np.arange(data_test.shape[0])
        np.random.shuffle(perm)
        data_test = data_test[perm]
        num_batch = int(data_test.shape[0] / BATCH_SIZE)
        for i in range(0, num_batch):
            start = i * BATCH_SIZE
            ns = data_test[start:start + BATCH_SIZE]
            xs, ys = my_data_loader.load_images_and_labels(ns, image_dir=config['image_dir'], n_class=N_CLASSES,
                                                           file_names_dict=file_names_dict,
                                                           num_channel=channels, do_center_crop=True)
            [_loss, summary_str] = sess.run([loss, sum_train], feed_dict={x_: xs, isTrain: False, y_: ys})
            writer_test.add_summary(summary_str, itr_test)
            itr_test += 1
            total_loss += _loss
        total_loss /= i
        print("Epoch: " + str(epoch) + " Test loss: " + str(total_loss) + '\n')
        test_loss.append(total_loss)

        checkpoint_name = os.path.join(output_dir, 'cp1_epoch' + str(epoch) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        np.save(os.path.join(output_dir, 'logs', 'train_loss.npy'), np.asarray(train_loss))
        np.save(os.path.join(output_dir, 'logs', 'test_loss.npy'), np.asarray(test_loss))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    args = parser.parse_args()

    # ============= Load config =============
    config_path = args.config

    train(config_path)
