import math
import sys
import os

from classifier.DenseNet import pretrained_classifier as celeba_classifier
from classifier.SimpleNet import pretrained_classifier as shapes_classifier
from data_loader.data_loader import CelebALoader, ShapesLoader

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
from utils import read_data_file, make3d_tensor, make4d_tensor, convert_ordinal_to_binary
from losses import *
import pdb
import yaml
import warnings
import argparse

warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(0)


def test(config_path, dbg_img_label_dict=None, dbg_mode=False, export_output=True, dbg_size=10, dbg_img_indices=[]):
    # ============= Load config =============

    config = yaml.load(open(config_path))
    print(config)

    # ============= Experiment Folder=============
    assets_dir = os.path.join(config['log_dir'], config['name'])
    log_dir = os.path.join(assets_dir, 'log')
    ckpt_dir = os.path.join(assets_dir, 'ckpt_dir')
    sample_dir = os.path.join(assets_dir, 'sample')
    test_dir = os.path.join(assets_dir, 'test')

    # ============= Experiment Parameters =============
    ckpt_dir_cls = config['cls_experiment']
    if 'evaluation_batch_size' in config.keys():
        BATCH_SIZE = config['evaluation_batch_size']
    else:
        BATCH_SIZE = config['batch_size']
    channels = config['num_channel']
    input_size = config['input_size']
    NUMS_CLASS_cls = config['num_class']
    NUMS_CLASS = config['num_bins']
    MU_CLUSTER = config['mu_cluster']
    VAR_CLUSTER = config['var_cluster']
    TRAVERSAL_N_SIGMA = config['traversal_n_sigma']
    STEP_SIZE = 2*TRAVERSAL_N_SIGMA * VAR_CLUSTER/(NUMS_CLASS - 1)
    OFFSET = MU_CLUSTER - TRAVERSAL_N_SIGMA * VAR_CLUSTER

    metrics_stability_nx = config['metrics_stability_nx']
    metrics_stability_var = config['metrics_stability_var']
    target_class = config['target_class']
    ckpt_dir_continue = ckpt_dir
    if dbg_img_label_dict is not None:
        image_label_dict = dbg_img_label_dict
    else:
        image_label_dict = config['image_label_dict']

    # CSVAE parameters
    beta1 = config['beta1']
    beta2 = config['beta2']
    beta3 = config['beta3']
    beta4 = config['beta4']
    beta5 = config['beta5']
    z_dim = config['z_dim']
    w_dim = config['w_dim']

    if dbg_mode:
        num_samples = dbg_size
    else:
        num_samples = config['count_to_save']

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
        if dbg_mode:
            my_data_loader = ShapesLoader(dbg_mode=True, dbg_size=dbg_size,
                                          dbg_image_label_dict=image_label_dict,
                                          dbg_img_indices=dbg_img_indices)
        else:
            # my_data_loader = ShapesLoader()
            # for efficiency, let's just load as many samples as we need
            my_data_loader = ShapesLoader(dbg_mode=True, dbg_size=num_samples,
                                          dbg_image_label_dict=image_label_dict,
                                          dbg_img_indices=dbg_img_indices)
            dbg_mode = True

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

    # ============= Data =============
    try:
        categories, file_names_dict = read_data_file(image_label_dict)
    except:
        print("Problem in reading input data file : ", image_label_dict)
        sys.exit()
    if dbg_mode and dataset == 'shapes':
        data = np.array([str(ind) for ind in my_data_loader.tmp_list])
    else:
        if len(dbg_img_indices) > 0:
            data = np.asarray(dbg_img_indices)
        else:
            data = np.asarray(list(file_names_dict.keys()))
    print("The classification categories are: ")
    print(categories)
    print('The size of the training set: ', data.shape[0])

    # ============= placeholder =============
    x_source = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x_source')
    y_s = tf.placeholder(tf.int32, [None, NUMS_CLASS_cls], name='y_s')
    y_source = y_s[:, NUMS_CLASS_cls - 1]
    train_phase = tf.placeholder(tf.bool, name='train_phase')

    y_target = tf.placeholder(tf.int32, [None, w_dim], name='y_target')  # between 0 and NUMS_CLASS

    generation_dim = w_dim

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

    # Create a single image based on y_target
    target_w = STEP_SIZE * tf.cast(y_target, dtype=tf.float32) + OFFSET
    fake_target_img = decoder_x(tf.concat([target_w, z], axis=-1))

    # ============= pre-trained classifier =============
    real_img_cls_logit_pretrained, real_img_cls_prediction = pretrained_classifier(x_source, NUMS_CLASS_cls,
                                                                                   reuse=False, name='classifier')
    fake_recon_cls_logit_pretrained, fake_recon_cls_prediction = pretrained_classifier(pred_x, NUMS_CLASS_cls,
                                                                                       reuse=True)
    fake_img_cls_logit_pretrained, fake_img_cls_prediction = pretrained_classifier(fake_target_img, NUMS_CLASS_cls,
                                                                                   reuse=True)

    # ============= predicted probabilities =============
    fake_target_p_tensor = tf.reduce_max(tf.cast(y_target, tf.float32) * 1.0 / float(NUMS_CLASS - 1), axis=1)

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
    def _save_output_array(name, values):
        np.save(os.path.join(test_dir, '{}.npy'.format(name)), values)

    names = np.empty([num_samples], dtype=object)
    real_imgs = np.empty([num_samples, input_size, input_size, channels])
    fake_t_imgs = np.empty([num_samples * generation_dim * NUMS_CLASS, input_size, input_size, channels])
    fake_s_recon_imgs = np.empty([num_samples * generation_dim * NUMS_CLASS, input_size, input_size, channels])
    real_ps = np.empty([num_samples * generation_dim * NUMS_CLASS, NUMS_CLASS_cls])
    recon_ps = np.empty([num_samples * generation_dim * NUMS_CLASS, NUMS_CLASS_cls])
    fake_target_ps = np.empty([num_samples * generation_dim * NUMS_CLASS])
    fake_ps = np.empty([num_samples * generation_dim * NUMS_CLASS, NUMS_CLASS_cls])

    # For stability metric
    stability_fake_t_imgs = np.empty([num_samples * generation_dim * NUMS_CLASS*metrics_stability_nx, input_size, input_size, channels])
    stability_fake_s_recon_imgs = np.empty([num_samples * generation_dim * NUMS_CLASS*metrics_stability_nx, input_size, input_size, channels])
    stability_recon_ps = np.empty([num_samples * generation_dim * NUMS_CLASS*metrics_stability_nx, NUMS_CLASS_cls])
    stability_fake_ps = np.empty([num_samples * generation_dim * NUMS_CLASS*metrics_stability_nx, NUMS_CLASS_cls])

    # TODO: can later save embeddings if needed
    # fake_t_embeds_z = np.empty([num_samples * generation_dim * NUMS_CLASS] + z_dim)
    # fake_t_embeds_w = np.empty([num_samples * generation_dim * NUMS_CLASS] + w_dim)
    # fake_s_embeds_z = np.empty([num_samples * generation_dim * NUMS_CLASS] + z_dim)
    # fake_s_embeds_w = np.empty([num_samples * generation_dim * NUMS_CLASS] + w_dim)
    # s_embeds_z = np.empty([num_samples * generation_dim * NUMS_CLASS] + z_dim)
    # s_embeds_w = np.empty([num_samples * generation_dim * NUMS_CLASS] + w_dim)

    arrs_to_save = [
        'names', 'real_imgs', 'fake_t_imgs', 'fake_s_recon_imgs',
        'real_ps', 'recon_ps', 'fake_target_ps', 'fake_ps',
        'stability_fake_t_imgs', 'stability_fake_s_recon_imgs', 'stability_recon_ps', 'stability_fake_ps'
    ]
    # TODO: can later save embeddings if needed
    # , 's_embeds_z', 's_embeds_w', 'fake_s_embeds_z', 'fake_s_embeds_w', 'fake_t_embeds_z', 'fake_t_embeds_w']

    np.random.shuffle(data)

    data = data[0:num_samples]
    for i in range(math.ceil(data.shape[0] / BATCH_SIZE)):
        image_paths = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        # num_seed_imgs is either BATCH_SIZE
        # or if the number of samples is not divisible by BATCH_SIZE a smaller value
        num_seed_imgs = np.shape(image_paths)[0]
        img, labels = my_data_loader.load_images_and_labels(image_paths, config['image_dir'], 1, file_names_dict,
                                                            channels, do_center_crop=True)
        img_repeat = np.repeat(img, NUMS_CLASS * generation_dim, 0)

        labels = np.repeat(labels, NUMS_CLASS * generation_dim, 0)
        labels = labels.ravel()
        labels = np.eye(NUMS_CLASS_cls)[labels.astype(int)]

        target_labels = np.tile(
            np.repeat(np.expand_dims(np.asarray(range(NUMS_CLASS)), axis=1), generation_dim, axis=1),  # [NUMS_CLASS, w_dim]
            (num_seed_imgs*generation_dim, 1))  # [num_seed_imgs * w_dim * NUMS_CLASS, w_dim]

        my_feed_dict = {y_target: target_labels, x_source: img_repeat, train_phase: False,
                        y_s: labels}

        fake_t_img, fake_s_recon_img, real_p, recon_p, fake_target_p, fake_p = sess.run(
            [fake_target_img, pred_x,
             real_img_cls_prediction, fake_recon_cls_prediction, fake_target_p_tensor, fake_img_cls_prediction],
            feed_dict=my_feed_dict)

        _num_cur_samples = min(data.shape[0] - i * BATCH_SIZE, BATCH_SIZE)
        start_ind = i * BATCH_SIZE
        end_ind = start_ind + _num_cur_samples
        multiplier = generation_dim * NUMS_CLASS
        metric_multiplier = generation_dim * NUMS_CLASS * metrics_stability_nx
        names[start_ind: end_ind] = np.asarray(image_paths)

        stability_fake_t_img = np.empty([_num_cur_samples*generation_dim * NUMS_CLASS*metrics_stability_nx, input_size, input_size, channels])
        stability_fake_s_recon_img = np.empty([_num_cur_samples*generation_dim * NUMS_CLASS*metrics_stability_nx, input_size, input_size, channels])
        stability_recon_p = np.empty([_num_cur_samples*generation_dim * NUMS_CLASS*metrics_stability_nx, NUMS_CLASS_cls])
        stability_fake_p = np.empty([_num_cur_samples*generation_dim * NUMS_CLASS*metrics_stability_nx, NUMS_CLASS_cls])

        for j in range(metrics_stability_nx):
            _start_ind = j * _num_cur_samples * multiplier
            _end_ind = (j+1) * _num_cur_samples * multiplier
            noisy_img = img + np.random.normal(loc=0.0, scale=metrics_stability_var, size=np.shape(img))
            stability_img_repeat = np.repeat(noisy_img, NUMS_CLASS * generation_dim, 0)
            stability_feed_dict = {y_target: target_labels, x_source: stability_img_repeat, train_phase: False,
                                   y_s: labels}
            _stability_fake_t_img, _stability_fake_s_recon_img, _stability_recon_p, _stability_fake_p = sess.run(
                [fake_target_img, pred_x, fake_recon_cls_prediction, fake_img_cls_prediction],
                feed_dict=stability_feed_dict)
            # TODO could improve speed by doing it in one step: problem with sizes
            stability_fake_t_img[_start_ind: _end_ind] = _stability_fake_t_img
            stability_fake_s_recon_img[_start_ind: _end_ind] = _stability_fake_s_recon_img
            stability_recon_p[_start_ind: _end_ind] = _stability_recon_p
            stability_fake_p[_start_ind: _end_ind] = _stability_fake_p

        stability_fake_t_imgs[start_ind * metric_multiplier: end_ind * metric_multiplier] = stability_fake_t_img
        stability_fake_s_recon_imgs[start_ind * metric_multiplier: end_ind * metric_multiplier] = stability_fake_s_recon_img
        stability_recon_ps[start_ind * metric_multiplier: end_ind * metric_multiplier] = stability_recon_p
        stability_fake_ps[start_ind * metric_multiplier: end_ind * metric_multiplier] = stability_fake_p

        real_imgs[start_ind: end_ind] = img
        fake_t_imgs[start_ind * multiplier: end_ind * multiplier] = fake_t_img
        fake_s_recon_imgs[start_ind * multiplier: end_ind * multiplier] = fake_s_recon_img
        real_ps[start_ind * multiplier: end_ind * multiplier] = real_p
        recon_ps[start_ind * multiplier: end_ind * multiplier] = recon_p
        fake_target_ps[start_ind * multiplier: end_ind * multiplier] = fake_target_p
        fake_ps[start_ind * multiplier: end_ind * multiplier] = fake_p

        # TODO: can later save embeddings if needed
        # fake_t_embeds_z[start_ind * multiplier: end_ind * multiplier] = fake_t_embeds_z
        # fake_t_embeds_w[start_ind * multiplier: end_ind * multiplier] = fake_t_embeds_w
        # fake_s_embeds_z[start_ind * multiplier: end_ind * multiplier] = fake_s_embeds_z
        # fake_s_embeds_w[start_ind * multiplier: end_ind * multiplier] = fake_s_embeds_w
        # s_embeds_z[start_ind * multiplier: end_ind * multiplier] = s_embeds_z
        # s_embeds_w[start_ind * multiplier: end_ind * multiplier] = s_embeds_w

        print('{} / {}'.format(i + 1, math.ceil(data.shape[0] / BATCH_SIZE)))

    if export_output:
        for arr_name in arrs_to_save:
            _save_output_array(arr_name, eval(arr_name))

    output_dict = {}
    for arr_name in arrs_to_save:
        output_dict.update({arr_name: eval(arr_name)})

    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    args = parser.parse_args()

    test(args.config)
