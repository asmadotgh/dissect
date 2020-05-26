import sys
import os
import math
from classifier.DenseNet import pretrained_classifier as celeba_classifier
from classifier.SimpleNet import pretrained_classifier as shapes_classifier
from data_loader.data_loader import CelebALoader, ShapesLoader

from explainer.networks_128 import Discriminator_Ordinal as celeba_Discriminator_Ordinal
from explainer.networks_128 import Generator_Encoder_Decoder as celeba_Generator_Encoder_Decoder
from explainer.networks_128 import Discriminator_Contrastive as celeba_Discriminator_Contrastive

from explainer.networks_64 import Discriminator_Ordinal as shapes_Discriminator_Ordinal
from explainer.networks_64 import Generator_Encoder_Decoder as shapes_Generator_Encoder_Decoder
from explainer.networks_64 import Discriminator_Contrastive as shapes_Discriminator_Contrastive

import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
from utils import *
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
np.random.seed(0)


def convert_ordinal_to_binary(y, n):
    y = np.asarray(y).astype(int)
    new_y = np.zeros([y.shape[0], n])
    new_y[:, 0] = y
    for i in range(0, y.shape[0]):
        for j in range(1, y[i] + 1):
            new_y[i, j] = 1
    return new_y


def test(config_path, dbg_mode=False, export_output=True, dbg_size=10):
    # ============= Load config =============

    config = yaml.load(open(config_path))
    print(config)

    HAS_MAIN_DIM = 'main_dim' in config_path or 'plus' in config_path
    print('Support main dimension? {}'.format(HAS_MAIN_DIM))

    # ============= Experiment Folder=============
    assets_dir = os.path.join(config['log_dir'], config['name'])
    log_dir = os.path.join(assets_dir, 'log')
    ckpt_dir = os.path.join(assets_dir, 'ckpt_dir')
    sample_dir = os.path.join(assets_dir, 'sample')
    test_dir = os.path.join(assets_dir, 'test')

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
    ckpt_dir_continue = ckpt_dir
    # there is a main knob, at index k_dim, and k_dim disentangled knobs at indices 0..k_dim-1
    k_dim = config['k_dim']
    k_dim_plus = k_dim + 1
    lambda_r = config['lambda_r']
    disentangle = k_dim > 1
    if dbg_mode:
        num_samples = dbg_size
    else:
        num_samples = config['count_to_save']

    discriminate_evert_nth = config['discriminate_every_nth']
    generate_every_nth = config['generate_every_nth']
    dataset = config['dataset']
    if dataset == 'CelebA':
        pretrained_classifier = celeba_classifier
        my_data_loader = CelebALoader()
        Discriminator_Ordinal = celeba_Discriminator_Ordinal
        Generator_Encoder_Decoder = celeba_Generator_Encoder_Decoder
        Discriminator_Contrastive = celeba_Discriminator_Contrastive
    elif dataset == 'shapes':
        pretrained_classifier = shapes_classifier
        if dbg_mode:
            if 'max_samples_per_bin' in config.keys():
                my_data_loader = ShapesLoader(dbg_mode=True, dbg_size=dbg_size,
                                              dbg_n_bins=config['num_bins'],
                                              dbg_max_samples_per_bin=config['max_samples_per_bin'])
            else:
                my_data_loader = ShapesLoader(dbg_mode=True, dbg_size=dbg_size,
                                              dbg_n_bins=config['num_bins'])
        else:
            my_data_loader = ShapesLoader()
        Discriminator_Ordinal = shapes_Discriminator_Ordinal
        Generator_Encoder_Decoder = shapes_Generator_Encoder_Decoder
        Discriminator_Contrastive = shapes_Discriminator_Contrastive

    # ============= Data =============
    try:
        categories, file_names_dict = read_data_file(config['image_label_dict'])
    except:
        print("Problem in reading input data file : ", config['image_label_dict'])
        sys.exit()
    if dbg_mode and dataset == 'shapes':
        data = np.array([str(ind) for ind in my_data_loader.tmp_list])
    else:
        data = np.asarray(list(file_names_dict.keys()))
    print("The classification categories are: ")
    print(categories)
    print('The size of the training set: ', data.shape[0])

    # ============= placeholder =============
    x_source = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x_source')
    y_s = tf.placeholder(tf.int32, [None, NUMS_CLASS], name='y_s')
    y_source = y_s[:, 0]
    train_phase = tf.placeholder(tf.bool, name='train_phase')

    y_t = tf.placeholder(tf.int32, [None, NUMS_CLASS], name='y_t')
    y_target = y_t[:, 0]

    if disentangle:
        y_regularizer = tf.placeholder(tf.int32, [None], name='y_regularizer')
        y_r = tf.placeholder(tf.float32, [None, k_dim], name='y_r')
        y_r_0 = tf.zeros_like(y_r, name='y_r_0')

    if HAS_MAIN_DIM:
        generation_dim = k_dim_plus
    else:
        generation_dim = k_dim

    # ============= G & D =============
    G = Generator_Encoder_Decoder("generator")  # with conditional BN, SAGAN: SN here as well
    D = Discriminator_Ordinal("discriminator")  # with SN and projection

    # TODO AG, currently D only conditions on delta, but not the "knob" index.
    real_source_logits = D(x_source, y_s, NUMS_CLASS, "NO_OPS")
    # TODO AG, currently G conditions on a one-hot vector of size NUMS_CLASS * k_dim. Make it more efficient?
    if disentangle:
        fake_target_img, fake_target_img_embedding = G(x_source, train_phase,
                                                       y_regularizer * NUMS_CLASS + y_target,
                                                       NUMS_CLASS * generation_dim)
        fake_source_img, fake_source_img_embedding = G(fake_target_img, train_phase,
                                                       y_regularizer * NUMS_CLASS + y_source,
                                                       NUMS_CLASS * generation_dim)
        fake_source_recons_img, x_source_img_embedding = G(x_source, train_phase,
                                                           y_regularizer * NUMS_CLASS + y_source,
                                                           NUMS_CLASS * generation_dim)
        if HAS_MAIN_DIM:
            fake_source_main_dim_img, fake_source_main_dim_img_embedding = G(fake_target_img, train_phase,
                                                                             k_dim * NUMS_CLASS + y_source,
                                                                             NUMS_CLASS * k_dim_plus)
    else:
        fake_target_img, fake_target_img_embedding = G(x_source, train_phase, y_target, NUMS_CLASS)
        fake_source_img, fake_source_img_embedding = G(fake_target_img, train_phase, y_source, NUMS_CLASS)
        fake_source_recons_img, x_source_img_embedding = G(x_source, train_phase, y_source, NUMS_CLASS)
    fake_target_logits = D(fake_target_img, y_t, NUMS_CLASS, None)

    # ============= pre-trained classifier =============
    real_img_cls_logit_pretrained, real_img_cls_prediction = pretrained_classifier(x_source, NUMS_CLASS_cls,
                                                                                   reuse=False, name='classifier')
    fake_img_cls_logit_pretrained, fake_img_cls_prediction = pretrained_classifier(fake_target_img, NUMS_CLASS_cls,
                                                                                   reuse=True)
    real_img_recons_cls_logit_pretrained, real_img_recons_cls_prediction = pretrained_classifier(fake_source_img,
                                                                                                 NUMS_CLASS_cls,
                                                                                                 reuse=True)
    # ============= session =============
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    # ============= Checkpoints =============
    print(" [*] Reading checkpoint...")

    ckpt = tf.train.get_checkpoint_state(ckpt_dir_continue)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(ckpt_dir_continue, ckpt_name))
        print(ckpt_dir_continue, ckpt_name)
        print("Successful checkpoint upload")
    else:
        print("Failed checkpoint load")
        sys.exit()

    # ============= load pre-trained classifier checkpoint =============
    class_vars = [var for var in slim.get_variables_to_restore() if 'classifier' in var.name]
    name_to_var_map_local = {var.op.name: var for var in class_vars}
    temp_saver = tf.train.Saver(var_list=name_to_var_map_local)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir_cls)
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    temp_saver.restore(sess, os.path.join(ckpt_dir_cls, ckpt_name))
    print("Classifier checkpoint loaded.................")
    print(ckpt_dir_cls, ckpt_name)

    # ============= Testing =============
    def _safe_append(all_arr, curr_arr, axis=0):
        if np.size(all_arr) == 0:
            res = curr_arr
        else:
            res = np.append(all_arr, curr_arr, axis=axis)
        return res

    def _save_output_arrays(out_dict):
        for fname in out_dict:
            np.save(os.path.join(test_dir, '{}.npy'.format(fname)), out_dict[fname])

    _empty_arr = np.empty([0])
    names = _empty_arr
    real_imgs = _empty_arr
    fake_t_imgs = _empty_arr
    fake_t_embeds = _empty_arr
    fake_s_imgs = _empty_arr
    fake_s_embeds = _empty_arr
    fake_s_recon_imgs = _empty_arr
    s_embeds = _empty_arr
    real_ps = _empty_arr
    fake_ps = _empty_arr
    recon_ps = _empty_arr
    if HAS_MAIN_DIM:
        fake_s_main_dim_imgs = _empty_arr
        fake_s_main_dim_embeds = _empty_arr

    output_dict = {}

    np.random.shuffle(data)

    data = data[0:num_samples]
    for i in range(math.ceil(data.shape[0] / BATCH_SIZE)):
        image_paths = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        # num_seed_imgs is either BATCH_SIZE
        # or if the number of samples is not divisible by BATCH_SIZE a smaller value
        num_seed_imgs = np.shape(image_paths)[0]
        img, labels = my_data_loader.load_images_and_labels(image_paths, config['image_dir'], 1, file_names_dict,
                                                            input_size, channels, do_center_crop=True)
        labels = np.repeat(labels, NUMS_CLASS * generation_dim, 0)
        labels = labels.ravel()
        labels = convert_ordinal_to_binary(labels, NUMS_CLASS)
        img_repeat = np.repeat(img, NUMS_CLASS * generation_dim, 0)

        target_labels = np.asarray([np.asarray(range(NUMS_CLASS)) for j in range(num_seed_imgs * generation_dim)])
        target_labels = target_labels.ravel()
        identity_ind = labels == target_labels
        target_labels = convert_ordinal_to_binary(target_labels, NUMS_CLASS)

        if disentangle:
            target_disentangle_ind = np.asarray(
                [np.repeat(np.asarray(range(generation_dim)), NUMS_CLASS) for j in range(num_seed_imgs)])
            target_disentangle_ind = target_disentangle_ind.ravel()
            target_disentangle_ind_one_hot = np.eye(generation_dim)[target_disentangle_ind][:, 0:k_dim]
            target_disentangle_ind_one_hot[identity_ind, :] = 0
            my_feed_dict = {y_t: target_labels, x_source: img_repeat, train_phase: False,
                            y_s: labels,
                            y_regularizer: target_disentangle_ind, y_r: target_disentangle_ind_one_hot}
        else:
            my_feed_dict = {y_t: target_labels, x_source: img_repeat, train_phase: False,
                            y_s: labels}

        fake_t_img, fake_t_embed, fake_s_img, fake_s_embed, fake_s_recon_img, s_embed, real_p, fake_p, recon_p = sess.run(
            [fake_target_img, fake_target_img_embedding,
             fake_source_img, fake_source_img_embedding,
             fake_source_recons_img, x_source_img_embedding,
             real_img_cls_prediction, fake_img_cls_prediction, real_img_recons_cls_prediction],
            feed_dict=my_feed_dict)
        if HAS_MAIN_DIM:
            fake_s_main_dim_img, fake_s_main_dim_embed = sess.run(
            [fake_source_main_dim_img, fake_source_main_dim_img_embedding],
            feed_dict=my_feed_dict)

        names = _safe_append(names, np.asarray(image_paths), axis=0)
        real_imgs = _safe_append(real_imgs, img, axis=0)
        fake_t_imgs = _safe_append(fake_t_imgs, fake_t_img, axis=0)
        fake_t_embeds = _safe_append(fake_t_embeds, fake_t_embed, axis=0)
        fake_s_imgs = _safe_append(fake_s_imgs, fake_s_img, axis=0)
        fake_s_embeds = _safe_append(fake_s_embeds, fake_s_embed, axis=0)
        fake_s_recon_imgs = _safe_append(fake_s_recon_imgs, fake_s_recon_img, axis=0)
        s_embeds = _safe_append(s_embeds, s_embed, axis=0)
        real_ps = _safe_append(real_ps, real_p, axis=0)
        fake_ps = _safe_append(fake_ps, fake_p, axis=0)
        recon_ps = _safe_append(recon_ps, recon_p, axis=0)

        output_dict.update({'names': names, 'real_imgs': real_imgs,
                            'fake_t_imgs': fake_t_imgs, 'fake_t_embeds': fake_t_embeds,
                            'fake_s_imgs': fake_s_imgs, 'fake_s_embeds': fake_s_embeds,
                            'fake_s_recon_imgs': fake_s_recon_imgs, 's_embeds': s_embeds,
                            'real_ps': real_ps, 'fake_ps': fake_ps, 'recon_ps': recon_ps})

        if HAS_MAIN_DIM:
            fake_s_main_dim_imgs = _safe_append(fake_s_main_dim_imgs, fake_s_main_dim_img, axis=0)
            fake_s_main_dim_embeds = _safe_append(fake_s_main_dim_embeds, fake_s_main_dim_embed, axis=0)
            output_dict.update({'fake_s_main_dim_imgs': fake_s_main_dim_imgs,
                                'fake_s_main_dim_embeds': fake_s_main_dim_embeds})

        print(i)

        if export_output:
            if i % 100 == 0:
                _save_output_arrays(output_dict)

    if export_output:
        _save_output_arrays(output_dict)

    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', '-c', default='configs/celebA_Young_Explainer.yaml')
    args = parser.parse_args()
    test(args.config)
