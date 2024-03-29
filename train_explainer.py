# This is the baseline vanilla Explainer code, extended to support multiple dimensions
# However, nothing enforces distinctness of the "knobs"

# The same effect can be achieved by running train_discoverer with lambda_r always set to zero.
# However, this code is more memory efficient since it doesn't need to make the regularizer network

import sys
import os
from classifier.DenseNet import pretrained_classifier as celeba_classifier
from classifier.SimpleNet import pretrained_classifier as shapes_classifier
from data_loader.data_loader import ImageLabelLoader, ShapesLoader

from explainer.ops import safe_log
from explainer.networks_128 import Discriminator_Ordinal as Discriminator_Ordinal_128
from explainer.networks_128 import Generator_Encoder_Decoder as Generator_Encoder_Decoder_128

from explainer.networks_64 import Discriminator_Ordinal as Discriminator_Ordinal_64
from explainer.networks_64 import Generator_Encoder_Decoder as Generator_Encoder_Decoder_64

import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
from utils import save_images, read_data_file, convert_ordinal_to_binary
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
    target_class = config['target_class']
    lambda_GAN = config['lambda_GAN']
    lambda_cyc = config['lambda_cyc']
    lambda_cls = config['lambda_cls']
    save_summary = int(config['save_summary'])
    ckpt_dir_continue = config['ckpt_dir_continue']
    k_dim = config['k_dim']
    disentangle = k_dim > 1
    discriminate_evert_nth = config['discriminate_every_nth']
    generate_every_nth = config['generate_every_nth']
    dataset = config['dataset']
    if dataset == 'CelebA':
        pretrained_classifier = celeba_classifier
        my_data_loader = ImageLabelLoader()
        Discriminator_Ordinal = Discriminator_Ordinal_128
        Generator_Encoder_Decoder = Generator_Encoder_Decoder_128
    elif dataset == 'shapes':
        pretrained_classifier = shapes_classifier
        if args.debug:
            my_data_loader = ShapesLoader(dbg_mode=True, dbg_size=config['batch_size'],
                                          dbg_image_label_dict=config['image_label_dict'])
        else:
            my_data_loader = ShapesLoader()
        Discriminator_Ordinal = Discriminator_Ordinal_64
        Generator_Encoder_Decoder = Generator_Encoder_Decoder_64
    elif dataset == 'CelebA64' or dataset == 'dermatology':
        pretrained_classifier = celeba_classifier
        my_data_loader = ImageLabelLoader(input_size=64)
        Discriminator_Ordinal = Discriminator_Ordinal_64
        Generator_Encoder_Decoder = Generator_Encoder_Decoder_64
    elif dataset == 'synthderm':
        pretrained_classifier = celeba_classifier
        my_data_loader = ImageLabelLoader(input_size=64)
        Discriminator_Ordinal = Discriminator_Ordinal_64
        Generator_Encoder_Decoder = Generator_Encoder_Decoder_64

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

    if disentangle:
        y_regularizer = tf.placeholder(tf.int32, [None], name='y_regularizer')

    # ============= G & D =============
    G = Generator_Encoder_Decoder("generator")  # with conditional BN, SAGAN: SN here as well
    D = Discriminator_Ordinal("discriminator")  # with SN and projection

    real_source_logits = D(x_source, y_s, NUMS_CLASS, "NO_OPS")
    if disentangle:
        fake_target_img, fake_target_img_embedding = G(x_source,
                                                       y_regularizer * NUMS_CLASS + y_target, NUMS_CLASS * k_dim)
        fake_source_img, fake_source_img_embedding = G(fake_target_img,
                                                       y_regularizer * NUMS_CLASS + y_source, NUMS_CLASS * k_dim)
        fake_source_recons_img, x_source_img_embedding = G(x_source,
                                                           y_regularizer * NUMS_CLASS + y_source, NUMS_CLASS * k_dim)
    else:
        fake_target_img, fake_target_img_embedding = G(x_source,  y_target, NUMS_CLASS)
        fake_source_img, fake_source_img_embedding = G(fake_target_img, y_source, NUMS_CLASS)
        fake_source_recons_img, x_source_img_embedding = G(x_source, y_source, NUMS_CLASS)
    fake_target_logits = D(fake_target_img, y_t, NUMS_CLASS, None)

    # ============= pre-trained classifier =============
    real_img_cls_logit_pretrained, real_img_cls_prediction = pretrained_classifier(x_source, NUMS_CLASS_cls,
                                                                                   reuse=False, name='classifier')
    fake_img_cls_logit_pretrained, fake_img_cls_prediction = pretrained_classifier(fake_target_img, NUMS_CLASS_cls,
                                                                                   reuse=True)
    real_img_recons_cls_logit_pretrained, real_img_recons_cls_prediction = pretrained_classifier(fake_source_img,
                                                                                                 NUMS_CLASS_cls,
                                                                                                 reuse=True)

    # ============= pre-trained classifier loss =============
    real_p = tf.cast(y_target, tf.float32) * 1.0 / float(NUMS_CLASS - 1)
    fake_q = fake_img_cls_prediction[:, target_class]
    fake_evaluation = (real_p * safe_log(fake_q)) + ((1 - real_p) * safe_log(1 - fake_q))
    fake_evaluation = -tf.reduce_mean(fake_evaluation)

    recons_evaluation = (real_img_cls_prediction[:, target_class] * safe_log(
        real_img_recons_cls_prediction[:, target_class])) + (
                                (1 - real_img_cls_prediction[:, target_class]) * safe_log(
                            1 - real_img_recons_cls_prediction[:, target_class]))
    recons_evaluation = -tf.reduce_mean(recons_evaluation)

    # ============= Loss =============
    D_loss_GAN, D_acc, D_precision, D_recall = discriminator_loss('hinge', real_source_logits, fake_target_logits)
    G_loss_GAN = generator_loss('hinge', fake_target_logits)
    G_loss_cyc = l1_loss(x_source, fake_source_img)
    G_loss_rec = l1_loss(x_source, fake_source_recons_img)  #+l2_loss(x_source_img_embedding, fake_source_img_embedding)
    D_loss = (D_loss_GAN * lambda_GAN)
    D_opt = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(D_loss, var_list=D.var_list())

    G_loss = (G_loss_GAN * lambda_GAN) + (G_loss_rec * lambda_cyc) + (G_loss_cyc * lambda_cyc) + (
            fake_evaluation * lambda_cls) + (recons_evaluation * lambda_cls)
    G_opt = tf.train.AdamOptimizer(2e-4, beta1=0., beta2=0.9).minimize(G_loss, var_list=G.var_list())

    # ============= summary =============
    real_img_sum = tf.summary.image('real_img', x_source)
    fake_img_sum = tf.summary.image('fake_target_img', fake_target_img)
    fake_source_img_sum = tf.summary.image('fake_source_img', fake_source_img)
    fake_source_recons_img_sum = tf.summary.image('fake_source_recons_img', fake_source_recons_img)

    acc_d = tf.summary.scalar('discriminator/acc_d', D_acc)
    precision_d = tf.summary.scalar('discriminator/precision_d', D_precision)
    recall_d = tf.summary.scalar('discriminator/recall_d', D_recall)
    loss_d_sum = tf.summary.scalar('discriminator/loss_d', D_loss)
    loss_d_GAN_sum = tf.summary.scalar('discriminator/loss_d_GAN', D_loss_GAN)

    loss_g_sum = tf.summary.scalar('generator/loss_g', G_loss)
    loss_g_GAN_sum = tf.summary.scalar('generator/loss_g_GAN', G_loss_GAN)
    loss_g_cyc_sum = tf.summary.scalar('generator/G_loss_cyc', G_loss_cyc)
    G_loss_rec_sum = tf.summary.scalar('generator/G_loss_rec', G_loss_rec)

    evaluation_fake = tf.summary.scalar('generator/fake_evaluation', fake_evaluation)
    evaluation_recons = tf.summary.scalar('generator/recons_evaluation', recons_evaluation)
    g_sum = tf.summary.merge(
        [loss_g_sum, loss_g_GAN_sum, loss_g_cyc_sum, real_img_sum, G_loss_rec_sum, fake_img_sum,
         fake_source_img_sum, fake_source_recons_img_sum, evaluation_fake, evaluation_recons])
    d_sum = tf.summary.merge([loss_d_sum, loss_d_GAN_sum, acc_d, precision_d, recall_d])

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

            identity_ind = labels == target_labels

            labels = convert_ordinal_to_binary(labels, NUMS_CLASS)
            target_labels = convert_ordinal_to_binary(target_labels, NUMS_CLASS)

            if disentangle:
                target_disentangle_ind = np.random.randint(0, high=k_dim, size=BATCH_SIZE)
                target_disentangle_ind_one_hot = np.eye(k_dim)[target_disentangle_ind]
                target_disentangle_ind_one_hot[identity_ind, :] = 0
                my_feed_dict = {y_t: target_labels, x_source: img, train_phase: True,
                                y_s: labels,
                                y_regularizer: target_disentangle_ind}
            else:
                my_feed_dict = {y_t: target_labels, x_source: img, train_phase: True,
                                y_s: labels}

            if (i + 1) % discriminate_evert_nth == 0:

                _, d_loss, summary_str = sess.run([D_opt, D_loss, d_sum],
                                                  feed_dict=my_feed_dict)
                writer.add_summary(summary_str, counter)

            if (i + 1) % generate_every_nth == 0:
                _, g_loss, g_summary_str = sess.run([G_opt, G_loss, g_sum], feed_dict=my_feed_dict)
                writer.add_summary(g_summary_str, counter)

            counter += 1

            def save_results(sess, step):
                num_seed_imgs = 8
                img, labels = my_data_loader.load_images_and_labels(image_paths[0:num_seed_imgs],
                                                                    image_dir=config['image_dir'], n_class=1,
                                                                    file_names_dict=file_names_dict,
                                                                    num_channel=channels,
                                                                    do_center_crop=True)
                labels = np.repeat(labels, NUMS_CLASS * k_dim, 0)
                labels = labels.ravel()
                labels = convert_ordinal_to_binary(labels, NUMS_CLASS)
                img_repeat = np.repeat(img, NUMS_CLASS * k_dim, 0)

                target_labels = np.asarray([np.asarray(range(NUMS_CLASS)) for j in range(num_seed_imgs * k_dim)])
                target_labels = target_labels.ravel()
                identity_ind = labels == target_labels
                target_labels = convert_ordinal_to_binary(target_labels, NUMS_CLASS)

                if disentangle:
                    target_disentangle_ind = np.asarray(
                        [np.repeat(np.asarray(range(k_dim)), NUMS_CLASS) for j in range(num_seed_imgs)])
                    target_disentangle_ind = target_disentangle_ind.ravel()
                    target_disentangle_ind_one_hot = np.eye(k_dim)[target_disentangle_ind]
                    target_disentangle_ind_one_hot[identity_ind, :] = 0
                    my_feed_dict = {y_t: target_labels, x_source: img_repeat, train_phase: False,
                                    y_s: labels,
                                    y_regularizer: target_disentangle_ind}
                else:
                    my_feed_dict = {y_t: target_labels, x_source: img_repeat, train_phase: False,
                                    y_s: labels}

                FAKE_IMG, fake_logits_ = sess.run([fake_target_img, fake_target_logits],
                                                  feed_dict=my_feed_dict)

                output_fake_img = np.reshape(FAKE_IMG, [-1, k_dim, NUMS_CLASS, input_size, input_size, channels])

                # save samples
                sample_file = os.path.join(sample_dir, '%06d.jpg' % step)
                save_images(output_fake_img, sample_file, num_samples=num_seed_imgs,
                            nums_class=NUMS_CLASS, k_dim=k_dim, image_size=input_size, num_channel=channels)
                np.save(sample_file.split('.jpg')[0] + '_y.npy', labels)

            if counter % save_summary == 0:
                save_results(sess, counter)
                # print(counter, i, e, g_loss, d_loss)

            if counter % 500 == 0:
                saver.save(sess, ckpt_dir + "/model%2d.ckpt" % counter)


if __name__ == "__main__":
    train()
