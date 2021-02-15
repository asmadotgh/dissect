import sys
import math
from classifier.DenseNet import pretrained_classifier as celeba_classifier
from classifier.SimpleNet import pretrained_classifier as shapes_classifier
from data_loader.data_loader import CelebALoader, ShapesLoader, ArrayLoader

from explainer.networks_128 import Discriminator_Ordinal as Discriminator_Ordinal_128
from explainer.networks_128 import Generator_Encoder_Decoder as Generator_Encoder_Decoder_128
from explainer.networks_128 import get_embedding_size as embedding_size_128

from explainer.networks_64 import Discriminator_Ordinal as Discriminator_Ordinal_64
from explainer.networks_64 import Generator_Encoder_Decoder as Generator_Encoder_Decoder_64
from explainer.networks_64 import get_embedding_size as embedding_size_64

import tensorflow.contrib.slim as slim
from utils import *
from losses import *
import numpy as np
import pdb
import yaml
import warnings
import argparse

warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(0)


def test(config, dbg_img_label_dict=None, dbg_mode=False, export_output=True, dbg_size=10, dbg_img_indices=[],
         overwrite_output_dir=None, overwrite_test_images=None, overwrite_test_labels=None, calc_stability=True):

    # ============= Experiment Folder=============
    assets_dir = os.path.join(config['log_dir'], config['name'])
    log_dir = os.path.join(assets_dir, 'log')
    ckpt_dir = os.path.join(assets_dir, 'ckpt_dir')
    sample_dir = os.path.join(assets_dir, 'sample')

    if overwrite_output_dir is not None:
        test_dir = os.path.join(overwrite_output_dir, 'test')
    else:
        test_dir = os.path.join(assets_dir, 'test')

    # ============= Experiment Parameters =============
    if overwrite_test_labels is not None and overwrite_test_images is not None:
        OVERWRITE_TESTING = True
    else:
        OVERWRITE_TESTING = False

    ckpt_dir_cls = config['cls_experiment']
    if 'evaluation_batch_size' in config.keys():
        BATCH_SIZE = config['evaluation_batch_size']
    else:
        BATCH_SIZE = config['batch_size']
    channels = config['num_channel']
    input_size = config['input_size']
    NUMS_CLASS_cls = config['num_class']
    NUMS_CLASS = config['num_bins']

    metrics_stability_nx = config['metrics_stability_nx']
    metrics_stability_var = config['metrics_stability_var']

    ckpt_dir_continue = ckpt_dir
    if dbg_img_label_dict is not None:
        image_label_dict = dbg_img_label_dict
    else:
        image_label_dict = config['image_label_dict']
    # there are k_dim disentangled knobs at indices 0..k_dim-1
    k_dim = config['k_dim']
    disentangle = k_dim > 1

    if OVERWRITE_TESTING:
        num_samples = len(overwrite_test_labels)
    elif dbg_mode:
        num_samples = dbg_size
    else:
        num_samples = config['count_to_save']

    dataset = config['dataset']
    if dataset == 'CelebA':
        if OVERWRITE_TESTING:
            my_data_loader = ArrayLoader(overwrite_test_images, overwrite_test_labels, input_size=128)
        else:
            my_data_loader = CelebALoader(input_size=128)
        EMBEDDING_SIZE = embedding_size_128()
        pretrained_classifier = celeba_classifier
        Discriminator_Ordinal = Discriminator_Ordinal_128
        Generator_Encoder_Decoder = Generator_Encoder_Decoder_128
    elif dataset == 'shapes':
        if OVERWRITE_TESTING:
            my_data_loader = ArrayLoader(overwrite_test_images, overwrite_test_labels, input_size=64)
        else:
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
        EMBEDDING_SIZE = embedding_size_64()
        pretrained_classifier = shapes_classifier
        Discriminator_Ordinal = Discriminator_Ordinal_64
        Generator_Encoder_Decoder = Generator_Encoder_Decoder_64
    elif dataset == 'CelebA64' or dataset == 'dermatology':
        if OVERWRITE_TESTING:
            my_data_loader = ArrayLoader(overwrite_test_images, overwrite_test_labels, input_size=64)
        else:
            my_data_loader = CelebALoader(input_size=64)
        EMBEDDING_SIZE = embedding_size_64()
        pretrained_classifier = celeba_classifier
        Discriminator_Ordinal = Discriminator_Ordinal_64
        Generator_Encoder_Decoder = Generator_Encoder_Decoder_64

    # ============= Data =============
    try:
        categories, file_names_dict = read_data_file(image_label_dict)
    except:
        print("Problem in reading input data file : ", image_label_dict)
        sys.exit()
    if OVERWRITE_TESTING:
        data = np.arange(len(overwrite_test_labels))
    elif dbg_mode and dataset == 'shapes':
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
    y_s = tf.placeholder(tf.int32, [None, NUMS_CLASS], name='y_s')
    y_source = y_s[:, 0]
    train_phase = tf.placeholder(tf.bool, name='train_phase')

    y_t = tf.placeholder(tf.int32, [None, NUMS_CLASS], name='y_t')
    y_target = y_t[:, 0]

    if disentangle:
        y_regularizer = tf.placeholder(tf.int32, [None], name='y_regularizer')
        y_r = tf.placeholder(tf.float32, [None, k_dim], name='y_r')

    generation_dim = k_dim

    # ============= G & D =============
    G = Generator_Encoder_Decoder("generator")  # with conditional BN, SAGAN: SN here as well
    D = Discriminator_Ordinal("discriminator")  # with SN and projection

    # TODO AG, currently D only conditions on delta, but not the "knob" index.
    real_source_logits = D(x_source, y_s, NUMS_CLASS, "NO_OPS")
    # TODO AG, currently G conditions on a one-hot vector of size NUMS_CLASS * k_dim. Make it more efficient?
    if disentangle:
        fake_target_img, fake_target_img_embedding = G(x_source,
                                                       y_regularizer * NUMS_CLASS + y_target,
                                                       NUMS_CLASS * generation_dim)
        fake_source_img, fake_source_img_embedding = G(fake_target_img,
                                                       y_regularizer * NUMS_CLASS + y_source,
                                                       NUMS_CLASS * generation_dim)
        fake_source_recons_img, x_source_img_embedding = G(x_source,
                                                           y_regularizer * NUMS_CLASS + y_source,
                                                           NUMS_CLASS * generation_dim)
    else:
        fake_target_img, fake_target_img_embedding = G(x_source, y_target, NUMS_CLASS)
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
    fake_img_target_cls_prediction = tf.cast(y_target, tf.float32) * 1.0 / float(NUMS_CLASS - 1)
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
    fake_t_imgs = np.empty([num_samples, generation_dim, NUMS_CLASS, input_size, input_size, channels])
    fake_t_embeds = np.empty([num_samples, generation_dim, NUMS_CLASS] + EMBEDDING_SIZE)
    fake_s_imgs = np.empty([num_samples, generation_dim, NUMS_CLASS, input_size, input_size, channels])
    fake_s_embeds = np.empty([num_samples, generation_dim, NUMS_CLASS] + EMBEDDING_SIZE)
    fake_s_recon_imgs = np.empty([num_samples, generation_dim, NUMS_CLASS, input_size, input_size, channels])
    s_embeds = np.empty([num_samples, generation_dim, NUMS_CLASS] + EMBEDDING_SIZE)
    real_ps = np.empty([num_samples, generation_dim, NUMS_CLASS, NUMS_CLASS_cls])
    recon_ps = np.empty([num_samples, generation_dim, NUMS_CLASS, NUMS_CLASS_cls])
    fake_target_ps = np.empty([num_samples, generation_dim, NUMS_CLASS])
    fake_ps = np.empty([num_samples, generation_dim, NUMS_CLASS, NUMS_CLASS_cls])

    # For stability metric
    stability_fake_t_imgs = np.empty(
        [num_samples, metrics_stability_nx, generation_dim, NUMS_CLASS, input_size, input_size, channels])
    stability_fake_s_recon_imgs = np.empty(
        [num_samples, metrics_stability_nx, generation_dim, NUMS_CLASS, input_size, input_size, channels])
    stability_recon_ps = np.empty(
        [num_samples, metrics_stability_nx, generation_dim, NUMS_CLASS, NUMS_CLASS_cls])
    stability_fake_ps = np.empty([num_samples, metrics_stability_nx, generation_dim, NUMS_CLASS, NUMS_CLASS_cls])


    arrs_to_save = ['names', 'real_imgs', 'fake_t_imgs', 'fake_t_embeds', 'fake_s_imgs', 'fake_s_embeds',
                    'fake_s_recon_imgs', 's_embeds', 'real_ps', 'recon_ps', 'fake_target_ps', 'fake_ps',
                    'stability_fake_t_imgs', 'stability_fake_s_recon_imgs', 'stability_recon_ps', 'stability_fake_ps']

    np.random.shuffle(data)

    data = data[0:num_samples]
    for i in range(math.ceil(data.shape[0] / BATCH_SIZE)):
        image_paths = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        # num_seed_imgs is either BATCH_SIZE
        # or if the number of samples is not divisible by BATCH_SIZE a smaller value
        num_seed_imgs = np.shape(image_paths)[0]
        img, labels = my_data_loader.load_images_and_labels(image_paths, config['image_dir'], 1, file_names_dict,
                                                            channels, do_center_crop=True)
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

        fake_t_img, fake_t_embed, fake_s_img, fake_s_embed, fake_s_recon_img, s_embed, real_p, recon_p, fake_target_p, fake_p = sess.run(
            [fake_target_img, fake_target_img_embedding,
             fake_source_img, fake_source_img_embedding,
             fake_source_recons_img, x_source_img_embedding,
             real_img_cls_prediction, real_img_recons_cls_prediction,
             fake_img_target_cls_prediction, fake_img_cls_prediction],
            feed_dict=my_feed_dict)

        _num_cur_samples = min(data.shape[0] - i * BATCH_SIZE, BATCH_SIZE)
        start_ind = i * BATCH_SIZE
        end_ind = start_ind + _num_cur_samples
        names[start_ind: end_ind] = np.asarray(image_paths)

        if calc_stability:
            for j in range(metrics_stability_nx):
                noisy_img = img + np.random.normal(loc=0.0, scale=metrics_stability_var, size=np.shape(img))
                stability_img_repeat = np.repeat(noisy_img, NUMS_CLASS * generation_dim, 0)
                stability_feed_dict = my_feed_dict.copy()
                stability_feed_dict.update({x_source: stability_img_repeat})
                _stability_fake_t_img, _stability_fake_s_recon_img, _stability_recon_p, _stability_fake_p = sess.run(
                    [fake_target_img, fake_source_recons_img, real_img_recons_cls_prediction, fake_img_cls_prediction],
                    feed_dict=stability_feed_dict)

                stability_fake_t_imgs[start_ind: end_ind, j] = np.reshape(_stability_fake_t_img, (_num_cur_samples, generation_dim, NUMS_CLASS, input_size, input_size, channels))
                stability_fake_s_recon_imgs[start_ind: end_ind, j] = np.reshape(_stability_fake_s_recon_img, (_num_cur_samples, generation_dim, NUMS_CLASS, input_size, input_size, channels))
                stability_recon_ps[start_ind: end_ind, j] = np.reshape(_stability_recon_p, (_num_cur_samples, generation_dim, NUMS_CLASS, NUMS_CLASS_cls))
                stability_fake_ps[start_ind: end_ind, j] = np.reshape(_stability_fake_p, (_num_cur_samples, generation_dim, NUMS_CLASS, NUMS_CLASS_cls))

        real_imgs[start_ind: end_ind] = img
        fake_t_imgs[start_ind: end_ind] = np.reshape(fake_t_img, (_num_cur_samples, generation_dim, NUMS_CLASS, input_size, input_size, channels))
        fake_s_imgs[start_ind: end_ind] = np.reshape(fake_s_img, (_num_cur_samples, generation_dim, NUMS_CLASS, input_size, input_size, channels))
        fake_s_recon_imgs[start_ind: end_ind] = np.reshape(fake_s_recon_img, (_num_cur_samples, generation_dim, NUMS_CLASS, input_size, input_size, channels))
        real_ps[start_ind: end_ind] = np.reshape(real_p, (_num_cur_samples, generation_dim, NUMS_CLASS, NUMS_CLASS_cls))
        recon_ps[start_ind: end_ind] = np.reshape(recon_p, (_num_cur_samples, generation_dim, NUMS_CLASS, NUMS_CLASS_cls))
        fake_target_ps[start_ind: end_ind] = np.reshape(fake_target_p, (_num_cur_samples, generation_dim, NUMS_CLASS))
        fake_ps[start_ind: end_ind] = np.reshape(fake_p, (_num_cur_samples, generation_dim, NUMS_CLASS, NUMS_CLASS_cls))

        _RESHAPE_EMBED_SIZE = [_num_cur_samples, generation_dim, NUMS_CLASS] + EMBEDDING_SIZE
        fake_t_embeds[start_ind: end_ind] = np.reshape(fake_t_embed, _RESHAPE_EMBED_SIZE)
        fake_s_embeds[start_ind: end_ind] = np.reshape(fake_s_embed, _RESHAPE_EMBED_SIZE)
        s_embeds[start_ind: end_ind] = np.reshape(s_embed, _RESHAPE_EMBED_SIZE)

        print('{} / {}'.format(i+1, math.ceil(data.shape[0] / BATCH_SIZE)))

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

    # ============= Load config =============

    config = yaml.load(open(args.config))
    print(config)

    test(config)
