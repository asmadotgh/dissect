import numpy as np
import pandas as pd
import sys
import os
import pdb
import yaml
import tensorflow as tf
from classifier.DenseNet import pretrained_classifier as celeba_classifier
from classifier.SimpleNet import pretrained_classifier as shapes_classifier
from data_loader.data_loader import ImageLabelLoader, ShapesLoader
from utils import read_data_file
import argparse
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt


def test(config):
    # ============= Experiment Folder=============
    output_dir = os.path.join(config['log_dir'], config['name'])
    classifier_output_path = os.path.join(output_dir, 'classifier_output')
    try:
        os.makedirs(classifier_output_path)
    except:
        pass
    past_checkpoint = output_dir
    # ============= Experiment Parameters =============
    BATCH_SIZE = config['batch_size']
    channels = config['num_channel']
    input_size = config['input_size']
    N_CLASSES = config['num_class']
    dataset = config['dataset']
    # in certain circumstances, for example for when classifier has been trained
    # on re-sampled data, we want to still use the whole dataset for the generative model.
    # That's why we produce classifier's output on the test_image_label_dict
    if ('export_image_label_dict' in config.keys()) and ('export_train' in config.keys()) and (
            'export_test' in config.keys()):
        image_label_dict = config['export_image_label_dict']
        train_ids = config['export_train']
        test_ids = config['export_test']
    else:
        image_label_dict = config['image_label_dict']
        train_ids = config['train']
        test_ids = config['test']
    if dataset == 'CelebA':
        pretrained_classifier = celeba_classifier
        my_data_loader = ImageLabelLoader()
    elif dataset == 'shapes':
        pretrained_classifier = shapes_classifier
        my_data_loader = ShapesLoader()
    elif dataset == 'CelebA64' or dataset == 'dermatology':
        pretrained_classifier = celeba_classifier
        my_data_loader = ImageLabelLoader(input_size=64)
    # ============= Data =============
    try:
        categories, file_names_dict = read_data_file(image_label_dict)
    except:
        print("Problem in reading input data file : ", image_label_dict)
        sys.exit()
    data_train = np.load(train_ids)
    data_test = np.load(test_ids)
    print("The classification categories are: ")
    print(categories)
    print('The size of the training set: ', data_train.shape[0])
    print('The size of the testing set: ', data_test.shape[0])

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
    classif_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logit)
    loss = tf.losses.get_total_loss()
    # ============= Variables =============
    # Note that this list of variables only include the weights and biases in the model.
    lst_vars = []
    for v in tf.global_variables():
        lst_vars.append(v)
    # ============= Session =============
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(var_list=lst_vars)
    tf.global_variables_initializer().run()
    # ============= Load Checkpoint =============
    if past_checkpoint is not None:
        ckpt = tf.train.get_checkpoint_state(past_checkpoint + '/')
        if ckpt and ckpt.model_checkpoint_path:
            print(str(ckpt.model_checkpoint_path))
            saver.restore(sess, tf.train.latest_checkpoint(past_checkpoint + '/'))
        else:
            sys.exit()
    else:
        sys.exit()
    # ============= Testing - Save the Output =============

    def get_predictions(data, subset_name):
        names = np.empty([0])
        prediction_y = np.empty([0])
        true_y = np.empty([0])

        num_batch = int(data.shape[0] / BATCH_SIZE)
        for i in range(0, num_batch):
            start = i * BATCH_SIZE
            ns = data[start:start + BATCH_SIZE]
            xs, ys = my_data_loader.load_images_and_labels(ns, image_dir=config['image_dir'], n_class=N_CLASSES,
                                                           file_names_dict=file_names_dict,
                                                           num_channel=channels, do_center_crop=True)
            [_pred] = sess.run([prediction], feed_dict={x_: xs, isTrain: False, y_: ys})
            if i == 0:
                names = np.asarray(ns)
                prediction_y = np.asarray(_pred)
                true_y = np.asarray(ys)
            else:
                names = np.append(names, np.asarray(ns), axis=0)
                prediction_y = np.append(prediction_y, np.asarray(_pred), axis=0)
                true_y = np.append(true_y, np.asarray(ys), axis=0)
        np.save(classifier_output_path + '/name_{}1.npy'.format(subset_name), names)
        np.save(classifier_output_path + '/prediction_y_{}1.npy'.format(subset_name), prediction_y)
        np.save(classifier_output_path + '/true_y_{}1.npy'.format(subset_name), true_y)
        return names, prediction_y, np.reshape(true_y, [-1, N_CLASSES])

    train_names, train_prediction_y, train_true_y = get_predictions(data_train, 'train')
    test_names, test_prediction_y, test_true_y = get_predictions(data_test, 'test')

    return train_names, train_prediction_y, train_true_y, test_names, test_prediction_y, test_true_y


def process_classifier_output(names, prediction_y, true_y, names_i, prediction_y_i, true_y_i, config, n_bins,
                              max_samples_per_bin, all_samples):
    experiment_dir = os.path.join(config['log_dir'], config['name'], 'explainer_input')
    print('Saving files to: ', experiment_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    view_results(prediction_y, true_y, prediction_y_i, true_y_i)
    df, train_df, test_df = create_dataframe(names, prediction_y, true_y, names_i, prediction_y_i, true_y_i, n_bins)
    if all_samples:
        save_output(df, train_df, test_df, experiment_dir, n_bins, max_samples_per_bin, all_samples)
    else:
        plot_reliability_curve(df, 'Data-before binning', os.path.join(experiment_dir, 'before_rc'), n_bins)
        calibrated_df = calibrated_sampling(df, n_bins, max_samples_per_bin)
        plot_reliability_curve(calibrated_df, 'Data-after binning', os.path.join(experiment_dir, 'after_rc'), n_bins)
        save_output(calibrated_df, train_df, test_df, experiment_dir, n_bins, max_samples_per_bin, all_samples)


def view_results(prediction_y, true_y, prediction_y_i, true_y_i):
    for i in range(prediction_y.shape[1]):
        if prediction_y.shape[1] == 2:
            j = 1
        else:
            j = i
        print("ROC-AUC train: ", roc_auc_score(true_y[:, i], prediction_y[:, j]))
        print("ROC-AUC test: ", roc_auc_score(true_y_i[:, i], prediction_y_i[:, j]))
        print("Accuracy train: ", accuracy_score(true_y[:, i], (prediction_y[:, j] > 0.5).astype(int)))
        print("Accuracy test: ", accuracy_score(true_y_i[:, i], (prediction_y_i[:, j] > 0.5).astype(int)))
        print("Recall train: ", recall_score(true_y[:, i], (prediction_y[:, j] > 0.5).astype(int)))
        print("Recall test: ", recall_score(true_y_i[:, i], (prediction_y_i[:, j] > 0.5).astype(int)))
        tp = np.sum((prediction_y[true_y[:, i] == 1, j] > 0.5).astype(int))
        tp_i = np.sum((prediction_y_i[true_y_i[:, i] == 1, j] > 0.5).astype(int))
        print("Stats train: ", np.unique(true_y[:, i], return_counts=True), tp)
        print("Stats test: ", np.unique(true_y_i[:, i], return_counts=True), tp_i)

        print('Confusion matrix train: ',
              confusion_matrix(true_y[:, i], (prediction_y[:, j] > 0.5).astype(int)))
        print('Confusion matrix test: ',
              confusion_matrix(true_y_i[:, i], (prediction_y_i[:, j] > 0.5).astype(int)))

        if prediction_y.shape[1] == 2:
            break


def create_dataframe(names, prediction_y, true_y,
                     names_i, prediction_y_i, true_y_i, n_bins, current_index=0, current_index_prob=1):
    df_train_results = pd.DataFrame(
        data={'filename': names, 'label': true_y[:, current_index], 'prob': prediction_y[:, current_index_prob]})
    df_train_results['bin'] = np.minimum(
        np.floor(df_train_results["prob"].astype('float') * n_bins).astype('int'), n_bins-1)
    print('Train set size: ', df_train_results.shape)
    print('Number of points in each bin - Train: ', np.unique(df_train_results['bin'], return_counts=True))

    df_test_results = pd.DataFrame(data={
        'filename': names_i, 'label': true_y_i[:, current_index], 'prob': prediction_y_i[:, current_index_prob]})
    df_test_results['bin'] = np.floor(df_test_results["prob"].astype('float') * n_bins).astype('int')
    print('Test set size: ', df_test_results.shape)
    print('Number of points in each bin - Test: ', np.unique(df_test_results['bin'], return_counts=True))

    df = pd.concat([df_train_results, df_test_results])
    print('All data size: ', df.shape)

    return df, df_train_results, df_test_results


def plot_reliability_curve(df, legend_str, fname, n_bins):
    # Reliability Curve
    plt.figure()
    true_label = np.asarray(df['label']).astype(int)
    predicted_prob = np.asarray(df["prob"]).astype(float)
    fraction_of_positives, mean_predicted_value = calibration_curve(true_label, predicted_prob, n_bins=n_bins)
    clf_score = brier_score_loss(true_label, predicted_prob, pos_label=1)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s (%1.3f)" % (legend_str, clf_score))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.ylabel('Fraction of positives')
    plt.ylim([-0.05, 1.05])
    plt.title('Calibration plots  (reliability curve)')
    plt.legend()
    plt.savefig('{}_{}.pdf'.format(fname, n_bins), bbox_inches='tight')


def calibrated_sampling(df, n_bins, n):
    df_bin_all = pd.DataFrame()
    for i in range(n_bins):
        df_bin = df.loc[df['bin'] == i]
        print(df_bin.shape)
        print(np.min(df_bin['prob']), np.max(df_bin['prob']))
        print(np.unique(df_bin['label'], return_counts=True))
        df_bin_0 = df.loc[(df['bin'] == i) & (df['label'] == 0)]
        df_bin_1 = df.loc[(df['bin'] == i) & (df['label'] == 1)]
        n_0 = int((1 - (1.0/float(n_bins) * i)) * n)
        if df_bin_0.shape[0] >= n_0:
            df_bin = df_bin_0.sample(n=n_0)
        else:
            df_bin = df_bin_0
            n_0 = df_bin_0.shape[0]
        n_1 = n - n_0
        if n_1 > 0:
            if df_bin_1.shape[0] >= n_1:
                df_bin = pd.concat([df_bin, df_bin_1.sample(n=n_1)])
            else:
                df_bin = pd.concat([df_bin, df_bin_1])

        df_bin_all = pd.concat([df_bin, df_bin_all])
        print('Binned df shape: ', df_bin_all.shape)
        print('Binned df unique labels', np.unique(df_bin['label'], return_counts=True))

        print('Binned df unique bins', np.unique(df_bin_all['bin'], return_counts=True))

    return df_bin_all


def save_output(df_bin_all, df_train_results, df_test_results, experiment_dir, n_bins,
                max_samples_per_bin, all_samples):
    if all_samples:
        postfix = 'all'
    else:
        postfix = max_samples_per_bin

    output_fname = 'list_attr_{}_{}.txt'.format(n_bins, postfix)
    df_temp = df_bin_all[['filename', 'bin']]
    df_temp.to_csv(os.path.join(experiment_dir, output_fname), sep=' ', index=None, header=None)
    one_line = str(df_temp.shape[0]) + '\n'
    step = 1.0 / float(n_bins)
    second_line = ''
    for i in range(n_bins):
        second_line += '[{:.2f} {:.2f}) '.format(i * step, (i + 1) * step)
    second_line = second_line[:-2]+'] '
    second_line += '\n'
    with open(os.path.join(experiment_dir, output_fname), 'r+') as fp:
        lines = fp.readlines()  # lines is list of line, each element '...\n'
        lines.insert(0, one_line)  # you can use any index if you know the line index
        lines.insert(1, second_line)
        fp.seek(0)  # file pointer locates at the beginning to write the whole file again
        fp.writelines(lines)

    df_bin_all.to_csv(
        os.path.join(experiment_dir, 'Data_Output_Classifier_{}_{}.csv'.format(n_bins, postfix)), sep=' ',
        index=None)
    df_test_results.to_csv(
        os.path.join(experiment_dir, 'Data_Output_Classifier_All_Test_{}_{}.csv'.format(n_bins, postfix)),
        sep=' ', index=None)
    df_train_results.to_csv(
        os.path.join(experiment_dir, 'Data_Output_Classifier_All_Train_{}_{}.csv'.format(n_bins, postfix)),
        sep=' ', index=None)


def get_prediction_from_file(config):
    output_dir = os.path.join(config['log_dir'], config['name'], 'classifier_output')
    print(output_dir)
    # Read classifier output
    train_or_test = 'train1'
    train_names = np.load(os.path.join(output_dir, 'name_' + train_or_test + '.npy'), allow_pickle=True)
    train_prediction_y = np.load(os.path.join(output_dir, 'prediction_y_' + train_or_test + '.npy'))
    train_true_y = np.load(os.path.join(output_dir, 'true_y_' + train_or_test + '.npy'), allow_pickle=True)
    train_or_test = 'test1'
    test_names = np.load(os.path.join(output_dir, 'name_' + train_or_test + '.npy'), allow_pickle=True)
    test_prediction_y = np.load(os.path.join(output_dir, 'prediction_y_' + train_or_test + '.npy'))
    test_true_y = np.load(os.path.join(output_dir, 'true_y_' + train_or_test + '.npy'))
    print(train_names.shape, train_prediction_y.shape, train_true_y.shape)
    print(test_names.shape, test_prediction_y.shape, test_true_y.shape)
    train_true_y = np.reshape(train_true_y, [-1, config['num_class']])
    test_true_y = np.reshape(test_true_y, [-1, config['num_class']])
    return train_names, train_prediction_y, train_true_y, test_names, test_prediction_y, test_true_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--n_bins', '-nb', type=int, default=10)
    parser.add_argument('--all_samples', '-all', action='store_true')
    parser.add_argument('--max_samples_per_bin', '-ms', type=int, default=5000)
    args = parser.parse_args()
    # ============= Load config =============
    config_path = args.config
    config = yaml.load(open(config_path))
    print(config)

    try:
        train_names, train_prediction_y, train_true_y, test_names, test_prediction_y, test_true_y = get_prediction_from_file(config)
    except:
        print('Prediction files do not exist. Loading checkpoint and calculating predictions...')
        train_names, train_prediction_y, train_true_y, test_names, test_prediction_y, test_true_y = test(config)
        pass
    process_classifier_output(train_names, train_prediction_y, train_true_y, test_names, test_prediction_y,
                              test_true_y, config, args.n_bins, args.max_samples_per_bin, args.all_samples)
