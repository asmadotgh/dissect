# train_csvae trains a conditional subspace VAE, conditioned on f(x)

import sys
import os


from classifier.DenseNet import pretrained_classifier as celeba_classifier
from classifier.SimpleNet import pretrained_classifier as shapes_classifier
from data_loader.data_loader import CelebALoader, ShapesLoader

from explainer.ops import KL, safe_log

from explainer.networks_128 import EncoderZ as EncoderZ_128
from explainer.networks_128 import EncoderW as EncoderW_128
from explainer.networks_128 import DecoderX as DecoderX_128
from explainer.networks_128 import DecoderY as DecoderY_128

from explainer.networks_64 import EncoderZ as EncoderZ_64
from explainer.networks_64 import EncoderW as EncoderW_64
from explainer.networks_64 import DecoderX as DecoderX_64
from explainer.networks_64 import DecoderY as DecoderY_64

import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
from utils import save_image, read_data_file, make3d_tensor, make4d_tensor, convert_ordinal_to_binary
from losses import *
import pdb
import yaml
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
    MU_CLUSTER = config['mu_cluster']
    VAR_CLUSTER = config['var_cluster']
    TRAVERSAL_N_SIGMA = config['traversal_n_sigma']
    STEP_SIZE = 2*TRAVERSAL_N_SIGMA * VAR_CLUSTER/(NUMS_CLASS - 1)
    OFFSET = MU_CLUSTER - TRAVERSAL_N_SIGMA*VAR_CLUSTER
    target_class = config['target_class']

    # CSVAE parameters
    beta1 = config['beta1']
    beta2 = config['beta2']
    beta3 = config['beta3']
    beta4 = config['beta4']
    beta5 = config['beta5']
    z_dim = config['z_dim']
    w_dim = config['w_dim']

    save_summary = int(config['save_summary'])
    ckpt_dir_continue = config['ckpt_dir_continue']

    dataset = config['dataset']
    if dataset == 'CelebA':
        pretrained_classifier = celeba_classifier
        my_data_loader = CelebALoader()
        EncoderZ = EncoderZ_128
        EncoderW = EncoderW_128
        DecoderX = DecoderX_128
        DecoderY = DecoderY_128

    elif dataset == 'shapes':
        pretrained_classifier = shapes_classifier
        if args.debug:
            my_data_loader = ShapesLoader(dbg_mode=True, dbg_size=config['batch_size'],
                                          dbg_image_label_dict=config['image_label_dict'])
        else:
            my_data_loader = ShapesLoader()
        EncoderZ = EncoderZ_64
        EncoderW = EncoderW_64
        DecoderX = DecoderX_64
        DecoderY = DecoderY_64

    elif dataset == 'CelebA64' or dataset == 'dermatology':
        pretrained_classifier = celeba_classifier
        my_data_loader = CelebALoader(input_size=64)
        EncoderZ = EncoderZ_64
        EncoderW = EncoderW_64
        DecoderX = DecoderX_64
        DecoderY = DecoderY_64

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

    # CSVAE does not need discretizing categories. The default 2 is recommended.
    print("The classification categories are: ")
    print(categories)
    print('The size of the training set: ', data.shape[0])
    fp = open(os.path.join(log_dir, 'setting.txt'), 'w')
    fp.write('config_file:' + str(config_path) + '\n')
    fp.close()

    # ============= placeholder =============
    x_source = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x_source')
    y_s = tf.placeholder(tf.int32, [None, NUMS_CLASS_cls], name='y_s')
    y_source = y_s[:, NUMS_CLASS_cls-1]
    train_phase = tf.placeholder(tf.bool, name='train_phase')

    y_target = tf.placeholder(tf.int32, [None, w_dim], name='y_target')  # between 0 and NUMS_CLASS

    # ============= CSVAE =============

    encoder_z = EncoderZ('encoder_z')
    encoder_w = EncoderW('encoder_w')
    decoder_x = DecoderX('decoder_x')
    decoder_y = DecoderY('decoder_y')

    # encode x to get mean, log variance, and samples from the latent subspace Z
    mu_z, logvar_z, z = encoder_z(x_source, z_dim)
    # encode x and y to get mean, log variance, and samples from the latent subspace W
    mu_w, logvar_w, w = encoder_w(x_source, y_source, w_dim)

    # pass samples of z and w to get predictions of x
    pred_x = decoder_x(tf.concat([w, z], axis=-1))
    # get predicted labels based only on the latent subspace Z
    pred_y = decoder_y(z, NUMS_CLASS_cls)

    # Create and save a grid of images
    fake_img_traversal = tf.zeros([0, input_size, input_size, channels])
    for i in range(w_dim):
        for j in range(NUMS_CLASS):
            val = j * STEP_SIZE
            np_arr = np.zeros((BATCH_SIZE, w_dim))
            np_arr[:, i] = val
            tmp_w = tf.convert_to_tensor(np_arr, dtype=tf.float32)
            fake_img = decoder_x(tf.concat([tmp_w, z], axis=-1))
            fake_img_traversal = tf.concat([fake_img_traversal, fake_img], axis=0)
    fake_img_traversal_board = make4d_tensor(fake_img_traversal, channels, input_size, w_dim, NUMS_CLASS, BATCH_SIZE)
    fake_img_traversal_save = make3d_tensor(fake_img_traversal, channels, input_size, w_dim, NUMS_CLASS, BATCH_SIZE)


    # Create and save 2d traversal, this is relevant only for w_dim == 2
    fake_2d_img_traversal = tf.zeros([0, input_size, input_size, channels])
    for i in range(NUMS_CLASS):
        for j in range(NUMS_CLASS):
            val_0 = i * STEP_SIZE
            val_1 = j * STEP_SIZE
            np_arr = np.zeros((BATCH_SIZE, w_dim))
            np_arr[:, 0] = val_0
            np_arr[:, 1] = val_1
            tmp_w = tf.convert_to_tensor(np_arr, dtype=tf.float32)
            fake_2d_img = decoder_x(tf.concat([tmp_w, z], axis=-1))
            fake_2d_img_traversal = tf.concat([fake_2d_img_traversal, fake_2d_img], axis=0)
    fake_2d_img_traversal_board = make4d_tensor(fake_2d_img_traversal, channels, input_size, NUMS_CLASS, NUMS_CLASS, BATCH_SIZE)
    fake_2d_img_traversal_save = make3d_tensor(fake_2d_img_traversal, channels, input_size, NUMS_CLASS, NUMS_CLASS, BATCH_SIZE)

    # Create a single image based on y_target
    target_w = STEP_SIZE * tf.cast(y_target, dtype=tf.float32) + OFFSET
    fake_target_img = decoder_x(tf.concat([target_w, z], axis=-1))

    # ============= pre-trained classifier =============

    real_img_cls_logit_pretrained, real_img_cls_prediction = pretrained_classifier(x_source, NUMS_CLASS_cls,
                                                                                   reuse=False, name='classifier')
    fake_recon_cls_logit_pretrained, fake_recon_cls_prediction = pretrained_classifier(pred_x, NUMS_CLASS_cls,
                                                                                       reuse=True)
    fake_img_cls_logit_pretrained, fake_img_cls_prediction = pretrained_classifier(fake_img, NUMS_CLASS_cls,
                                                                                   reuse=True)

    # ============= predicted probabilities =============
    fake_target_p_tensor = tf.reduce_max(tf.cast(y_target, tf.float32) * 1.0 / float(NUMS_CLASS - 1), axis=1)

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
    #    We want the latent subspace W for this label to be close to mean MU_CLUSTER, var VAR_CLUSTER
    kl0 = KL(mu1=mu_w, logvar1=logvar_w, mu2=tf.ones_like(mu_w) * MU_CLUSTER, logvar2=tf.ones_like(logvar_w) * np.log(VAR_CLUSTER))

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
    fake_recon_img_sum = tf.summary.image('fake_recon_img', pred_x)
    fake_img_sum = tf.summary.image('fake_target_img', fake_target_img)
    fake_img_traversal_sum = tf.summary.image('fake_img_traversal', fake_img_traversal_board)
    fake_2d_img_traversal_sum = tf.summary.image('fake_2d_img_traversal', fake_2d_img_traversal_board)

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
        [loss_sum, real_img_sum, fake_recon_img_sum, fake_img_sum, fake_img_traversal_sum, fake_2d_img_traversal_sum])

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
            labels = np.eye(NUMS_CLASS_cls)[labels.astype(int)]

            target_labels_probs = np.random.randint(0, high=NUMS_CLASS, size=BATCH_SIZE)
            target_labels_w_ind = np.random.randint(0, high=w_dim, size=BATCH_SIZE)
            target_labels = np.eye(w_dim)[target_labels_w_ind] * np.repeat(np.expand_dims(target_labels_probs, axis=-1), w_dim, axis=1)

            my_feed_dict = {y_target: target_labels, x_source: img, train_phase: True, y_s: labels}

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
                num_seed_imgs = BATCH_SIZE
                img, labels = my_data_loader.load_images_and_labels(image_paths[0:num_seed_imgs],
                                                                    image_dir=config['image_dir'], n_class=1,
                                                                    file_names_dict=file_names_dict,
                                                                    num_channel=channels,
                                                                    do_center_crop=True)

                labels = labels.ravel()
                labels = np.eye(NUMS_CLASS_cls)[labels.astype(int)]

                target_labels_probs = np.random.randint(0, high=NUMS_CLASS, size=BATCH_SIZE)
                target_labels_w_ind = np.random.randint(0, high=w_dim, size=BATCH_SIZE)
                target_labels = np.eye(w_dim)[target_labels_w_ind] * np.repeat(
                    np.expand_dims(target_labels_probs, axis=-1), w_dim, axis=1)

                my_feed_dict = {y_target: target_labels, x_source: img, train_phase: False,
                                y_s: labels}

                sample_fake_img_traversal, sample_fake_2d_img_traversal = sess.run([fake_img_traversal_save, fake_img_traversal_save], feed_dict=my_feed_dict)

                # save samples
                sample_file = os.path.join(sample_dir, '%06d.jpg' % step)
                save_image(sample_fake_img_traversal, sample_file)

                sample_file = os.path.join(sample_dir, '%06d_2d.jpg' % step)
                save_image(sample_fake_2d_img_traversal, sample_file)

            batch_counter = int(counter/2)
            if batch_counter % save_summary == 0:
                save_results(sess, batch_counter)

            if batch_counter % 500 == 0:
                saver.save(sess, ckpt_dir + "/model%2d.ckpt" % batch_counter)


if __name__ == "__main__":
    train()
