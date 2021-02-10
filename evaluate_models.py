from test_explainer_discoverer import test as test_discoverer
from test_csvae import test as test_csvae
import os
import numpy as np
import pandas as pd
import pdb
import yaml
import warnings
import argparse
from scipy.stats import entropy, pearsonr, spearmanr
from metrics.posthoc_classification import classifier_distinct_64, classifier_realistic_64
import tensorflow as tf
from utils import calc_metrics_arr, calc_accuracy, safe_append
from sklearn.metrics import mean_squared_error
import math

warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(0)


_EMPTY_ARR = np.empty([0])


def _save_csv(out_dir, out_dict):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    metrics_df = pd.DataFrame.from_dict(out_dict)
    metrics_df.to_csv(os.path.join(out_dir, 'metrics.csv'))


def calc_influential(results_dict, target_class):
    print('Calculating metrics for: Influential')
    p = results_dict['fake_target_ps']
    q = results_dict['fake_ps'][:, target_class]
    MSE = mean_squared_error(p, q)
    KL = entropy(p, q)
    pearson_r, pearson_p = pearsonr(p, q)
    spearman_r, spearman_p = spearmanr(p, q)

    print(
        'Influential - MSE: {:.3f}, KL: {:.3f}, pearson_r: {:.3f}, pearson_p: {:.3f}, spearman_r: {:.3f}, '
        'spearman_p:{:.3f}'.format(
            MSE, KL, pearson_r, pearson_p, spearman_r, spearman_p))

    metrics_dict = {}
    for metric in ['MSE', 'KL', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p']:
        metrics_dict.update({'influential_{}'.format(metric): [eval(metric)]})
    print('Metrics successfully calculated: Influential')

    # TODO Faithfulness: Are “relevant” features truly relevant?
    return metrics_dict


def calc_distinct(results_dict):
    tf.reset_default_graph()
    print('Calculating metrics for: Distinct')

    # ============= Metrics Folder - Distinct =============
    output_dir = os.path.join(config['log_dir'], config['name'], 'test', 'metrics', 'distinct')
    logs_dir = os.path.join(output_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # ============= Experiment Parameters =============
    BATCH_SIZE = config['metrics_batch_size']
    EPOCHS = config['metrics_epochs']
    TEST_RATIO = config['metrics_test_ratio']
    NUM_BINS = config['num_bins']
    if 'k_dim' in config.keys():
        N_KNOBS = config['k_dim']
    elif 'w_dim' in config.keys():
        N_KNOBS = config['w_dim']
    else:
        print('Number of knobs not specified. Returning...')
        return {}
    TARGET_CLASS = config['target_class']
    if N_KNOBS <= 1:
        print('This model has only one dimension. Distinctness metrics are not applicable.')
        return {}
    channels = config['num_channel']
    input_size = config['input_size']
    dataset = config['dataset']
    # ============= Data =============
    data = _EMPTY_ARR
    labels = _EMPTY_ARR
    source_len = len(results_dict['real_imgs'])
    for dim in range(N_KNOBS):
        for bin_i in range(NUM_BINS):
            fake_inds = np.array(range(source_len))*N_KNOBS*NUM_BINS + dim*NUM_BINS + bin_i
            data_dim_bin = np.append(results_dict['real_imgs'], results_dict['fake_t_imgs'][fake_inds], axis=-1)
            # dimension dim has been switched
            switched_dim = np.ones(source_len, dtype=int)*dim
            # unless the real probability and fake target probability are the same,
            # in which no dimension has been switched
            fixed_indices = (np.around(results_dict['real_ps'][fake_inds][:, TARGET_CLASS], decimals=2) ==
                             results_dict['fake_target_ps'][fake_inds])
            labels_dim_bin = np.eye(N_KNOBS)[switched_dim]
            labels_dim_bin[fixed_indices] = 0
            data = safe_append(data, data_dim_bin)
            labels = safe_append(labels, labels_dim_bin)

    data_len = len(data)
    data_inds = np.array(range(data_len))
    np.random.shuffle(data_inds)

    train_inds = data_inds[int(data_len * TEST_RATIO):]
    test_inds = data_inds[:int(data_len * TEST_RATIO)]

    print('The size of the training set: ', train_inds.shape[0])
    print('The size of the testing set: ', test_inds.shape[0])
    # ============= placeholder =============
    with tf.name_scope('input'):
        x_ = tf.placeholder(tf.float32, [None, input_size, input_size, channels*2], name='x-input')
        y = tf.placeholder(tf.float32, [None, 2], name='y-input')
        isTrain = tf.placeholder(tf.bool)
    # ============= Model =============
    logit, prediction = classifier_distinct_64(x_, num_dims=N_KNOBS, isTrain=isTrain)
    classif_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logit)
    acc = calc_accuracy(prediction=prediction, labels=y)
    loss = tf.losses.get_total_loss()
    # ============= Optimization functions =============
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
    # ============= summary =============
    cls_loss = tf.summary.scalar('distinct/cls_loss', classif_loss)
    total_loss = tf.summary.scalar('distinct/loss', loss)
    cls_acc = tf.summary.scalar('distinct/acc', acc)
    summary_tf = tf.summary.merge([cls_loss, total_loss, cls_acc])
    # ============= Variables =============
    # Note that this list of variables only include the weights and biases in the model.
    lst_vars = []
    for v in tf.global_variables():
        lst_vars.append(v)
    # ============= Session =============
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=lst_vars)
    writer = tf.summary.FileWriter(output_dir + '/train', sess.graph)
    writer_test = tf.summary.FileWriter(output_dir + '/test', sess.graph)
    # ============= Training =============
    train_loss = []
    itr_train = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        np.random.shuffle(train_inds)
        num_batch = math.ceil(train_inds.shape[0] / BATCH_SIZE)
        for i in range(0, num_batch):
            start = i * BATCH_SIZE
            xs = data[train_inds[start:start + BATCH_SIZE]]
            ys = labels[train_inds[start:start + BATCH_SIZE]]
            [_, _loss, summary_str] = sess.run([train_step, loss, summary_tf],
                                               feed_dict={x_: xs, isTrain: True, y: ys})
            writer.add_summary(summary_str, itr_train)
            itr_train += 1
            total_loss += _loss
        total_loss /= num_batch
        print("Epoch: " + str(epoch) + " loss: " + str(total_loss) + '\n')
        train_loss.append(total_loss)

        checkpoint_name = os.path.join(output_dir, 'cp_epoch_{}.ckpt'.format(epoch))
        saver.save(sess, checkpoint_name)
        np.save(os.path.join(output_dir, 'logs', 'train_loss.npy'), np.asarray(train_loss))

    # ============= Testing =============
    test_preds = _EMPTY_ARR
    test_loss = []
    itr_test = 0

    total_test_loss = 0.0
    num_batch = math.ceil(test_inds.shape[0] / BATCH_SIZE)
    for i in range(0, num_batch):
        start = i * BATCH_SIZE
        xs = data[test_inds[start:start + BATCH_SIZE]]
        ys = labels[test_inds[start:start + BATCH_SIZE]]
        [_loss, summary_str, _pred] = sess.run([loss, summary_tf, prediction],
                                               feed_dict={x_: xs, isTrain: False, y: ys})
        writer_test.add_summary(summary_str, itr_test)
        itr_test += 1
        total_test_loss += _loss
        test_preds = safe_append(test_preds, _pred, axis=0)
    total_test_loss /= num_batch
    print("Epoch: " + str(epoch) + " Test loss: " + str(total_loss) + '\n')
    test_loss.append(total_test_loss)

    np.save(os.path.join(output_dir, 'logs', 'test_loss.npy'), np.asarray(test_loss))
    np.save(os.path.join(output_dir, 'logs', 'test_preds.npy'), np.asarray(test_preds))
    np.save(os.path.join(output_dir, 'logs', 'test_ys.npy'), np.asarray(labels[test_inds]))
    np.save(os.path.join(output_dir, 'logs', 'test_xs.npy'), np.asarray(data[test_inds]))

    accuracy, precision_per_dim, recall_per_dim = calc_metrics_arr(np.round(test_preds), labels[test_inds], average=None)
    _, precision_micro, recall_micro = calc_metrics_arr(np.round(test_preds), labels[test_inds],
                                                                   average='micro')
    _, precision_macro, recall_macro = calc_metrics_arr(np.round(test_preds), labels[test_inds],
                                                        average='macro')

    print('Distinct - accuracy: {:.3f}, '
          'precision: per dim: {}, micro: {:.3f}, macro: {:.3f}, '
          'recall: per dim: {}, micro: {:.3f}, macro: {:.3f}'.format(
        accuracy, precision_per_dim, precision_micro, precision_macro,
        recall_per_dim, recall_micro, recall_macro))
    metrics_dict = {}
    for metric in ['accuracy', 'precision_per_dim', 'precision_micro', 'precision_macro',
                   'recall_per_dim', 'recall_micro', 'recall_macro']:
        metrics_dict.update({'distinct_{}'.format(metric): [eval(metric)]})

    print('Metrics successfully calculated: Distinct')
    return metrics_dict


def calc_realistic(results_dict, config):
    tf.reset_default_graph()
    print('Calculating metrics for: Realistic')

    # ============= Metrics Folder - Realistic =============
    output_dir = os.path.join(config['log_dir'], config['name'], 'test', 'metrics', 'realistic')
    logs_dir = os.path.join(output_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # ============= Experiment Parameters =============
    BATCH_SIZE = config['metrics_batch_size']
    EPOCHS = config['metrics_epochs']
    TEST_RATIO = config['metrics_test_ratio']
    channels = config['num_channel']
    input_size = config['input_size']
    dataset = config['dataset']
    # ============= Data =============
    half_len = len(results_dict['real_imgs'])
    data_real = results_dict['real_imgs']
    fake_inds = np.array(range(len(results_dict['fake_t_imgs'])))
    np.random.shuffle(fake_inds)
    fake_inds = fake_inds[0:half_len]
    data_fake = results_dict['fake_t_imgs'][fake_inds]
    data = np.append(data_real, data_fake, axis=0)
    labels = np.append(np.ones(half_len), np.zeros(half_len), axis=0)
    data_len = len(data)
    data_inds = np.array(range(data_len))
    np.random.shuffle(data_inds)

    train_inds = data_inds[int(data_len * TEST_RATIO):]
    test_inds = data_inds[:int(data_len * TEST_RATIO)]

    print('The size of the training set: ', train_inds.shape[0])
    print('The size of the testing set: ', test_inds.shape[0])
    # ============= placeholder =============
    with tf.name_scope('input'):
        x_ = tf.placeholder(tf.float32, [None, input_size, input_size, channels], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')
        isTrain = tf.placeholder(tf.bool)
    # ============= Model =============
    y = tf.one_hot(y_, 2, on_value=1.0, off_value=0.0, axis=-1)
    logit, prediction = classifier_realistic_64(x_, n_label=2, isTrain=isTrain)
    classif_loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logit)
    acc = calc_accuracy(prediction=prediction, labels=y)
    loss = tf.losses.get_total_loss()
    # ============= Optimization functions =============
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
    # ============= summary =============
    cls_loss = tf.summary.scalar('realistic/cls_loss', classif_loss)
    total_loss = tf.summary.scalar('realistic/loss', loss)
    cls_acc = tf.summary.scalar('realistic/acc', acc)
    summary_tf = tf.summary.merge([cls_loss, total_loss, cls_acc])
    # ============= Variables =============
    # Note that this list of variables only include the weights and biases in the model.
    lst_vars = []
    for v in tf.global_variables():
        lst_vars.append(v)
    # ============= Session =============
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=lst_vars)
    writer = tf.summary.FileWriter(output_dir + '/train', sess.graph)
    writer_test = tf.summary.FileWriter(output_dir + '/test', sess.graph)
    # ============= Training =============
    train_loss = []
    itr_train = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        np.random.shuffle(train_inds)
        num_batch = math.ceil(train_inds.shape[0] / BATCH_SIZE)
        for i in range(0, num_batch):
            start = i * BATCH_SIZE
            xs = data[train_inds[start:start + BATCH_SIZE]]
            ys = labels[train_inds[start:start + BATCH_SIZE]]
            [_, _loss, summary_str] = sess.run([train_step, loss, summary_tf],
                                               feed_dict={x_: xs, isTrain: True, y_: ys})
            writer.add_summary(summary_str, itr_train)
            itr_train += 1
            total_loss += _loss
        total_loss /= num_batch
        print("Epoch: " + str(epoch) + " loss: " + str(total_loss) + '\n')
        train_loss.append(total_loss)

        checkpoint_name = os.path.join(output_dir, 'cp_epoch_{}.ckpt'.format(epoch))
        saver.save(sess, checkpoint_name)
        np.save(os.path.join(output_dir, 'logs', 'train_loss.npy'), np.asarray(train_loss))

    # ============= Testing =============
    test_preds = _EMPTY_ARR
    test_loss = []
    itr_test = 0

    total_test_loss = 0.0
    num_batch = math.ceil(test_inds.shape[0] / BATCH_SIZE)
    for i in range(0, num_batch):
        start = i * BATCH_SIZE
        xs = data[test_inds[start:start + BATCH_SIZE]]
        ys = labels[test_inds[start:start + BATCH_SIZE]]
        [_loss, summary_str, _pred] = sess.run([loss, summary_tf, prediction],
                                              feed_dict={x_: xs, isTrain: False, y_: ys})
        writer_test.add_summary(summary_str, itr_test)
        itr_test += 1
        total_test_loss += _loss
        test_preds = safe_append(test_preds, _pred, axis=0)
    total_test_loss /= num_batch
    print("Epoch: " + str(epoch) + " Test loss: " + str(total_loss) + '\n')
    test_loss.append(total_test_loss)

    np.save(os.path.join(output_dir, 'logs', 'test_loss.npy'), np.asarray(test_loss))
    np.save(os.path.join(output_dir, 'logs', 'test_preds.npy'), np.asarray(test_preds))
    np.save(os.path.join(output_dir, 'logs', 'test_ys.npy'), np.asarray(labels[test_inds]))
    np.save(os.path.join(output_dir, 'logs', 'test_xs.npy'), np.asarray(data[test_inds]))

    accuracy, precision, recall = calc_metrics_arr(np.argmax(test_preds, axis=1), labels[test_inds])

    print('Realistic - accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(accuracy, precision, recall))
    metrics_dict = {}
    for metric in ['accuracy', 'precision', 'recall']:
        metrics_dict.update({'realistic_{}'.format(metric): [eval(metric)]})

    # TODO: Add FID between real images and explanations?

    print('Metrics successfully calculated: Realistic')
    return metrics_dict


# Second, we define the substitutability metric as follows: Let an original train- ing set Dtrain = {(xi,yi|i = 1..N},
# a test set Dtest, and a classifier F(x) → y whose empirical performance on the test set is some score S. Given a
# new set of model-generated boundary-crossing images Dtrans = {(x′i,yi′|i = 1..N} we say that this set is
# R%−substitutable if our classifier can be retrained using Dtrans to achieve performance that is R% of S.
# For example, if our original dataset and classifier yield 90% performance, and we substitute a generated dataset for
# our original dataset and a re-trained classifier yields 45%, we would say the new dataset is 50% substitutable.
def calc_substitutability(results_dict):
    print('Calculating metrics for: Substitutability')

    sub_inds = np.logical_or(results_dict['fake_target_ps'] == 0.0, results_dict['fake_target_ps'] == 1.)
    labels = 1 * (results_dict['fake_target_ps'][sub_inds] > 0.5)
    pred = results_dict['fake_ps'][sub_inds]

    accuracy, precision, recall = calc_metrics_arr(np.argmax(pred, axis=1), labels)

    print('Substitutability - accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(accuracy, precision, recall))
    metrics_dict = {}
    for metric in ['accuracy', 'precision', 'recall']:
        metrics_dict.update({'substitutability_{}'.format(metric): [eval(metric)]})

    print('Metrics successfully calculated: Substitutability')
    return metrics_dict


# TODO add stability: how coherent are explanations for similar inputs
# https://github.com/GDPlumb/ExpO/blob/fdc80bdd09d02c3345a17365105d2fda804eb40b/Code/ExplanationMetrics.py#L151
def calc_stability(results_dict, config):
    print('Calculating metrics for: Stability')
    num_randoms = config['metrics_stability_nx']
    num_imgs = len(results_dict['fake_t_imgs'])
    num_pixes = config['input_size']*config['input_size']*config['num_channel']
    NUMS_CLASS_cls = config['num_class']
    l2_fake_t_img = 0
    l2_fake_s_recon_img = 0
    l2_recon_p = 0
    l2_fake_p = 0

    def _sq_diff(x, y):
        return np.sum((x-y)**2)

    for i in range(num_randoms):
        sub_inds = np.arange(num_imgs)*num_randoms + i
        l2_fake_t_img += _sq_diff(results_dict['fake_t_imgs'], results_dict['stability_fake_t_imgs'][sub_inds])
        l2_fake_s_recon_img += _sq_diff(results_dict['fake_s_recon_imgs'], results_dict['stability_fake_s_recon_imgs'][sub_inds])
        l2_recon_p += _sq_diff(results_dict['recon_ps'], results_dict['stability_recon_ps'][sub_inds])
        l2_fake_p += _sq_diff(results_dict['fake_ps'], results_dict['stability_fake_ps'][sub_inds])

    l2_fake_t_img = l2_fake_t_img/(num_randoms*num_imgs*num_pixes)
    l2_fake_s_recon_img = l2_fake_s_recon_img/(num_randoms*num_imgs*num_pixes)
    l2_recon_p = l2_recon_p/(num_randoms*num_imgs*NUMS_CLASS_cls)
    l2_fake_p = l2_fake_p/(num_randoms*num_imgs*NUMS_CLASS_cls)

    print('Stability - l2_fake_t_img: {:.3f}, l2_fake_s_recon_img: {:.3f}, l2_recon_p: {:.3f}, l2_fake_p:{:.3f}'.format(
        l2_fake_t_img, l2_fake_s_recon_img, l2_recon_p, l2_fake_p))
    metrics_dict = {}
    for metric in ['l2_fake_t_img', 'l2_fake_s_recon_img', 'l2_recon_p', 'l2_fake_p']:
        metrics_dict.update({'stability_{}'.format(metric): [eval(metric)]})

    print('Metrics successfully calculated: Stability')
    return metrics_dict


def evaluate(results_dict, config, output_dir=None, export_output=True):
    target_class = config['target_class']
    metrics_dict = {}

    # Stability
    stability_dict = calc_stability(results_dict, config)
    metrics_dict.update(stability_dict)

    # Substitutability
    substitutability_dict = calc_substitutability(results_dict)
    metrics_dict.update(substitutability_dict)

    # influential
    influential_dict = calc_influential(results_dict, target_class)
    metrics_dict.update(influential_dict)

    # Distinct
    distinct_dict = calc_distinct(results_dict)
    metrics_dict.update(distinct_dict)

    # Realist
    realistic_dict = calc_realistic(results_dict, config)
    metrics_dict.update(realistic_dict)

    if export_output:
        _save_csv(output_dir, metrics_dict)

    return metrics_dict


def get_results_from_file(output_dir):
    print(output_dir)
    if not os.path.exists(output_dir):
        raise ValueError('Directory {} does not exist.'.format(output_dir))
    # Read Explainer/Discoverer Results
    # files_to_load = ['names', 'real_imgs', 'fake_t_imgs', 'fake_t_embeds', 'fake_s_imgs', 'fake_s_embeds',
    #                  'fake_s_recon_imgs', 's_embeds', 'real_ps', 'recon_ps', 'fake_target_ps', 'fake_ps']
    # only loading the subset needed for the analyses:
    files_to_load = ['real_imgs', 'fake_t_imgs', 'real_ps', 'fake_target_ps', 'fake_ps',
                     'fake_s_recon_imgs', 'recon_ps',
                     'stability_fake_t_imgs', 'stability_fake_s_recon_imgs', 'stability_recon_ps','stability_fake_ps'
    ]
    results_dict = {}
    for fname in files_to_load:
        curr_file = os.path.join(output_dir, fname+'.npy')
        if not os.path.exists(curr_file):
            raise ValueError('File {} does not exist.'.format(output_dir))
        values = np.load(curr_file, allow_pickle=True)
        results_dict.update({fname: values})
    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config))
    print(config)

    out_dir = os.path.join(config['log_dir'], config['name'], 'test')

    try:
        results_dict = get_results_from_file(out_dir)
    except:
        print('Results files do not exist. Running test explainer/discoverer/CSVAE to produce results...')
        if 'w_dim' in config.keys():
            results_dict = test_csvae(args.config)
        elif 'k_dim' in config.keys():
            results_dict = test_discoverer(args.config)
        else:
            raise Exception('Config file not supported. Either CSVAE type or explainer/discoverer type...')
        pass

    metrics_dir = os.path.join(out_dir, 'metrics')
    evaluate(results_dict, config, output_dir=metrics_dir)