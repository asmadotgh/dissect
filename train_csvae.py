# train_csvae trains a conditional subspace VAE, conditioned on f(x)

import sys
import os


from classifier.DenseNet import pretrained_classifier as celeba_classifier
from classifier.SimpleNet import pretrained_classifier as shapes_classifier
from data_loader.data_loader import CelebALoader, ShapesLoader

from explainer.ops import KL, safe_log, convert_ordinal_to_binary

from explainer.networks_128 import EncoderZ as EncoderZ_128
from explainer.networks_128 import EncoderW as EncoderW_128
from explainer.networks_128 import Decoder as Decoder_128

from explainer.networks_64 import EncoderZ as EncoderZ_64
from explainer.networks_64 import EncoderW as EncoderW_64
from explainer.networks_64 import Decoder as Decoder_64

import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
from utils import save_images, read_data_file
from losses import *
import pdb
import yaml
import time
import scipy.io as sio
from datetime import datetime
import random
import warnings
import argparse


warnings.filterwarnings("ignore", category=DeprecationWarning)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--debug', '-d', action='store_true')
    args = parser.parse_args()

    # ============= Load config =============
    config_path = args.config
    config = yaml.load(open(config_path))
    print(config)

    # ============= Experiment Folder=============
    assets_dir = os.path.join(config['log_dir'], config['name'])
    log_dir = os.path.join(assets_dir, 'log')
    ckpt_dir = os.path.join(assets_dir, 'ckpt_dir')
    sample_dir = os.path.join(assets_dir, 'sample')
    test_dir = os.path.join(assets_dir, 'test')
    # make directory if not exist
    try:
        os.makedirs(log_dir)
    except:
        pass
    try:
        os.makedirs(ckpt_dir)
    except:
        pass
    try:
        os.makedirs(sample_dir)
    except:
        pass
    try:
        os.makedirs(test_dir)
    except:
        pass

    # ============= Experiment Parameters =============
    ckpt_dir_cls = config['cls_experiment']
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    channels = config['num_channel']
    input_size = config['input_size']
    NUMS_CLASS_cls = config['num_class']
    NUMS_CLASS = config['num_bins']
    target_class = config['target_class']

    # ADDED
    beta1 = config['beta1']
    beta2 = config['beta2']
    beta3 = config['beta3']
    beta4 = config['beta4']
    beta5 = config['beta5']

    z_dim = config['z_dim']
    w_dim = config['w_dim']
    ##

    save_summary = int(config['save_summary'])
    ckpt_dir_continue = config['ckpt_dir_continue']

    dataset = config['dataset']
    if dataset == 'CelebA':
        pretrained_classifier = celeba_classifier
        my_data_loader = CelebALoader()
        EncoderZ = EncoderZ_128
        EncoderW = EncoderW_128
        Decoder = Decoder_128

    elif dataset == 'shapes':
        pretrained_classifier = shapes_classifier
        if args.debug:
            my_data_loader = ShapesLoader(dbg_mode=True, dbg_size=config['batch_size'],
                                          dbg_image_label_dict=config['image_label_dict'])
        else:
            my_data_loader = ShapesLoader()
        EncoderZ = EncoderZ_64
        EncoderW = EncoderW_64
        Decoder = Decoder_64

    elif dataset == 'CelebA64' or dataset == 'dermatology':
        pretrained_classifier = celeba_classifier
        my_data_loader = CelebALoader(input_size=64)
        EncoderZ = EncoderZ_64
        EncoderW = EncoderW_64
        Decoder = Decoder_64

    if ckpt_dir_continue == '':
        continue_train = False
    else:
        ckpt_dir_continue = os.path.join(ckpt_dir_continue, 'ckpt_dir')
        continue_train = True

    # ============= Data =============
    try:
        categories, file_names_dict = read_data_file(config['image_label_dict'])
    except:
        print("Problem in reading input data file : ", config['image_label_dict'])
        sys.exit()
    data = np.asarray(list(file_names_dict.keys()))

    # TODO do we need descretized values?
    print("The classification categories are: ")
    print(categories)
    print('The size of the training set: ', data.shape[0])
    fp = open(os.path.join(log_dir, 'setting.txt'), 'w')
    fp.write('config_file:' + str(config_path) + '\n')
    fp.close()

    # ============= placeholder =============
    x_source = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x_source')
    y_s = tf.placeholder(tf.int32, [None, NUMS_CLASS], name='y_s')
    y_source = y_s[:, 0]
    train_phase = tf.placeholder(tf.bool, name='train_phase')

    y_t = tf.placeholder(tf.int32, [None, NUMS_CLASS], name='y_t')
    y_target = y_t[:, 0]


    # ============= CSVAE =============

    encoder_z = EncoderZ('encoder_z')
    encoder_w = EncoderW('encoder_w')
    decoder_x = Decoder('decoder_x')
    decoder_y = Decoder('decoder_y')

    # encode x to get mean, log variance, and samples from the latent subspace Z
    mu_z, logvar_z, z = encoder_z(x_source, z_dim)
    # encode x and y to get mean, log variance, and samples from the latent subspace W
    mu_w, logvar_w, w = encoder_w(x_source, y_source, w_dim)

    # pass samples of z and w to get predictions of x
    pred_x = decoder_x(tf.concat([w, z], axis=-1), tf.shape(x_source), activation='tanh')
    # get predicted labels based only on the latent subspace Z
    pred_y = decoder_y(z, tf.shape(y_source), activation='sigmoid')

    # ============= pre-trained classifier =============
    real_img_cls_logit_pretrained, real_img_cls_prediction = pretrained_classifier(x_source, NUMS_CLASS_cls,
                                                                                   reuse=False, name='classifier')
    # TODO generate a set of images with decoder and pass them through external classifier?
    # fake_img_cls_logit_pretrained, fake_img_cls_prediction = pretrained_classifier(fake_target_img, NUMS_CLASS_cls,
    #                                                                                reuse=True)
    # real_img_recons_cls_logit_pretrained, real_img_recons_cls_prediction = pretrained_classifier(fake_source_img,
    #                                                                                              NUMS_CLASS_cls,
    #                                                                                              reuse=True)

    # ============= pre-trained classifier loss =============
    real_p = tf.cast(y_target, tf.float32) * 1.0 / float(NUMS_CLASS - 1)
    # TODO do I need predicted probability?
    # fake_q = fake_img_cls_prediction[:, target_class]

    # ============= Loss =============
    # OPTIMIZATION:

    # Specified in section 4.1 of http://www.cs.toronto.edu/~zemel/documents/Conditional_Subspace_VAE_all.pdf
    # There are three components: M1, M2, N

    # 1.Optimize the first loss related to maximizing variational lower bound
    #   on the marginal log likelihood and minimizing mutual information

    # define two KL divergences:
    # KL divergence for label 1
    #    We want the latent subspace W for this label to be close to mean 0, var 0.01
    kl1 = KL(mu1=mu_w, logvar1=logvar_w,
             mu2=tf.zeros_like(mu_w), logvar2=tf.ones_like(logvar_w) * np.log(0.01))
    # KL divergence for label 0
    #    We want the latent subspace W for this label to be close to mean 3, var 0.01
    kl0 = KL(mu1=mu_w, logvar1=logvar_w, mu2=tf.ones_like(mu_w) * 3., logvar2=tf.ones_like(logvar_w) * np.log(0.01))

    loss_m1_1 = tf.reduce_sum(beta1 * tf.reduce_sum((x_source - pred_x) ** 2, axis=-1))  # corresponds to M1
    loss_m1_2 = tf.reduce_sum(
        beta2 * tf.where(tf.equal(y_source, tf.ones_like(y_source)), kl1, kl0))  # corresponds to M1
    loss_m1_3 = tf.reduce_sum(
        beta3 * KL(mu_z, logvar_z, tf.zeros_like(mu_z), tf.zeros_like(logvar_z)))  # corresponds to M1
    loss_m2 = tf.reduce_sum(beta4 * tf.reduce_sum(pred_y * safe_log(pred_y), axis=-1))  # corresponds to M2

    loss_m1 = loss_m1_1 + loss_m1_2 + loss_m1_3
    loss1 = loss_m1 + loss_m2

    # 2. Optimize second loss related to learning the approximate posterior

    loss_n = tf.reduce_sum(beta5 * tf.where(y_source == 1, -safe_log(pred_y[:, 1]), -safe_log(pred_y[:, 0])))  # N

    loss2 = loss_n

    optimizer_1 = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(loss1, var_list=decoder_x.var_list() +
                                                                                             encoder_w.var_list() +
                                                                                             encoder_z.var_list())
    optimizer_2 = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(loss2, var_list=decoder_y.var_list())

    # combine losses for tracking
    loss = loss1 + loss2

    # ============= summary =============
    real_img_sum = tf.summary.image('real_img', x_source)
    # TODO for each dimension, create a fake image at 3, and one fake image at all 0
    # fake_img_sum = tf.summary.image('fake_target_img', fake_target_img)
    # fake_source_img_sum = tf.summary.image('fake_source_img', fake_source_img)
    # fake_source_recons_img_sum = tf.summary.image('fake_source_recons_img', fake_source_recons_img)

    loss_m1_sum = tf.summary.scalar('losses/M1', loss_m1)
    loss_m1_1_sum = tf.summary.scalar('losses/M1/m1_1', loss_m1_1)
    loss_m1_2_sum = tf.summary.scalar('losses/M1/m1_2', loss_m1_2)
    loss_m1_3_sum = tf.summary.scalar('losses/M1/m1_3', loss_m1_3)
    loss_m2_sum = tf.summary.scalar('losses/M2', loss_m2)
    loss_n_sum = tf.summary.scalar('losses/N', loss_n)
    loss_sum = tf.summary.scalar('losses/total_loss', loss)

    part1_sum = tf.summary.merge(
        [loss_m1_sum, loss_m1_1_sum, loss_m1_2_sum, loss_m1_3_sum, loss_m2_sum])
    part2_sum = tf.summary.merge(
        [loss_n_sum, loss_sum, ])
    overall_sum = tf.summary.merge(
        [loss_sum, real_img_sum])

    # ============= session =============
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    # ============= Checkpoints =============
    if continue_train:
        print(" [*] before training, Load checkpoint ")
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(ckpt_dir_continue)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_dir_continue, ckpt_name))
            print(ckpt_dir_continue, ckpt_name)
            print("Successful checkpoint upload")
        else:
            print("Failed checkpoint load")
    else:
        print(" [!] before training, no need to Load ")

    # ============= load pre-trained classifier checkpoint =============
    class_vars = [var for var in slim.get_variables_to_restore() if 'classifier' in var.name]
    name_to_var_map_local = {var.op.name: var for var in class_vars}
    temp_saver = tf.train.Saver(var_list=name_to_var_map_local)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir_cls)
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    temp_saver.restore(sess, os.path.join(ckpt_dir_cls, ckpt_name))
    print("Classifier checkpoint loaded.................")
    print(ckpt_dir_cls, ckpt_name)

    # ============= Training =============
    counter = 1
    for e in range(1, EPOCHS + 1):
        np.random.shuffle(data)
        for i in range(data.shape[0] // BATCH_SIZE):
            if args.debug:
                image_paths = np.array([str(ind) for ind in my_data_loader.tmp_list])
            else:
                image_paths = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            img, labels = my_data_loader.load_images_and_labels(image_paths, image_dir=config['image_dir'], n_class=1,
                                                                file_names_dict=file_names_dict,
                                                                num_channel=channels, do_center_crop=True)

            labels = labels.ravel()
            target_labels = np.random.randint(0, high=NUMS_CLASS, size=BATCH_SIZE)

            labels = convert_ordinal_to_binary(labels, NUMS_CLASS)
            target_labels = convert_ordinal_to_binary(target_labels, NUMS_CLASS)

            my_feed_dict = {y_t: target_labels, x_source: img, train_phase: True,
                            y_s: labels}

            _, par1_loss, par1_summary_str, overall_sum_str = sess.run([optimizer_1, loss1, part1_sum, overall_sum],
                                                                       feed_dict=my_feed_dict)

            writer.add_summary(par1_summary_str, counter)
            writer.add_summary(overall_sum_str, counter)
            counter += 1

            _, part2_loss, part2_summary_str, overall_sum_str2 = sess.run([optimizer_2, loss2, part2_sum, overall_sum],
                                                                          feed_dict=my_feed_dict)
            writer.add_summary(part2_summary_str, counter)
            writer.add_summary(overall_sum_str2, counter)
            counter += 1

            def save_results(sess, step):
                num_seed_imgs = 8
                img, labels = my_data_loader.load_images_and_labels(image_paths[0:num_seed_imgs],
                                                                    image_dir=config['image_dir'], n_class=1,
                                                                    file_names_dict=file_names_dict,
                                                                    num_channel=channels,
                                                                    do_center_crop=True)
                 # TODO save a few samples
                # labels = np.repeat(labels, NUMS_CLASS * w_dim, 0)
                # labels = labels.ravel()
                # labels = convert_ordinal_to_binary(labels, NUMS_CLASS)
                # img_repeat = np.repeat(img, NUMS_CLASS * w_dim, 0)
                #
                # target_labels = np.asarray([np.asarray(range(NUMS_CLASS)) for j in range(num_seed_imgs * k_dim)])
                # target_labels = target_labels.ravel()
                # target_labels = convert_ordinal_to_binary(target_labels, NUMS_CLASS)
                #
                #
                # my_feed_dict = {y_t: target_labels, x_source: img_repeat, train_phase: False,
                #                 y_s: labels}
                #
                #
                # FAKE_IMG, fake_logits_ = sess.run([fake_target_img, fake_target_logits],
                #                                   feed_dict=my_feed_dict)
                #
                # output_fake_img = np.reshape(FAKE_IMG, [-1, w_dim, NUMS_CLASS, input_size, input_size, channels])
                #
                # # save samples
                # sample_file = os.path.join(sample_dir, '%06d.jpg' % step)
                # save_images(output_fake_img, sample_file, num_samples=num_seed_imgs,
                #             nums_class=NUMS_CLASS, k_dim=w_dim, image_size=input_size, num_channel=channels)
                # np.save(sample_file.split('.jpg')[0] + '_y.npy', labels)

            if counter % save_summary == 0:
                save_results(sess, counter)

            if counter % 500 == 0:
                saver.save(sess, ckpt_dir + "/model%2d.ckpt" % counter)


if __name__ == "__main__":
    train()
