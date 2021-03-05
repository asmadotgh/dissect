import os
import pickle
import numpy as np
from tqdm import tqdm
import scipy.misc as scm
import pdb
import os
from glob import glob
from collections import namedtuple
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score


def crop_center(img, cropx, cropy):
    y, x, z = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, :]


def read_data_file(file_path, image_dir=''):
    attr_list = {}
    path = file_path
    file = open(path, 'r')
    n = file.readline()
    n = int(n.split('\n')[0])  # Number of images
    attr_line = file.readline()
    attr_names = attr_line.split('\n')[0].split()  # attribute name
    for line in file:
        row = line.split('\n')[0].split()
        img_name = os.path.join(image_dir, row.pop(0))
        try:
            row = [float(val) for val in row]
        except:
            print(line)
            img_name = img_name + ' ' + row[0]
            row.pop(0)
            row = [float(val) for val in row]
        #    img = img[..., ::-1] # bgr to rgb
        attr_list[img_name] = row
    file.close()
    return attr_names, attr_list


def load_images_and_labels(imgs_names, image_dir, n_class, attr_list, input_size=128, num_channel=3,
                           do_center_crop=False):
    imgs = np.zeros((imgs_names.shape[0], input_size, input_size, num_channel), dtype=np.float32)
    labels = np.zeros((imgs_names.shape[0], n_class), dtype=np.float32)

    for i, img_name in tqdm(enumerate(imgs_names)):
        img = scm.imread(os.path.join(image_dir, img_name))
        if do_center_crop and input_size == 128:
            img = crop_center(img, 150, 150)
        img = scm.imresize(img, [input_size, input_size, num_channel])
        img = np.reshape(img, [input_size, input_size, num_channel])
        img = img / 255.0
        img = img - 0.5
        img = img * 2.0
        imgs[i] = img
        try:
            labels[i] = attr_list[img_name]
        except:
            print(img_name)
    labels[np.where(labels == -1)] = 0
    return imgs, labels


def inverse_image(img):
    img = (img + 1.) * 127.5
    img[img > 255] = 255.
    img[img < 0] = 0.
    return img.astype(np.uint8)


def make4d_tensor(img, num_channel, image_size, row, col, batch_size):
    # img.shape = [row*col*batch_size, h, w, c]
    # final: [batch_size, row*h, col*w, c]
    if num_channel > 1:
        img = tf.reshape(img, [row, col, batch_size, image_size, image_size, num_channel])  # [row, col, batch, h, w, c]
    else:
        img = tf.reshape(img, [row, col, batch_size, image_size, image_size])  # [row, col, batch, h, w]
    img = tf.unstack(img, axis=0)  # row * [col, batch, h, w, c]
    img = tf.concat(img, axis=2)  # [col, batch, row*h, w, c]
    img = tf.unstack(img, axis=0)  # col * [batch, row*h, w, c]
    img = tf.concat(img, axis=2)  # [batch, row*h, col*w, c]
    img = tf.reshape(img, [batch_size, row*image_size, col*image_size, num_channel])  # [batch, row*h, col*w, c]
    return img


def make3d_tensor(img, num_channel, image_size, row, col, batch_size):
    # img.shape = [batch_size*row*col, h, w, c]
    # final: [batch_size, row*h, col*w, c]
    img = make4d_tensor(img, num_channel, image_size, row, col, batch_size)
    img = tf.reshape(img, [batch_size*row*image_size, col*image_size, num_channel])  # [batch*row*h, col*w, c]
    return img


def make3d(img, num_channel, image_size, row, col):
    # img.shape = [row*col, h, w, c]
    # final: [row*h, col*w, c]
    if num_channel > 1:
        img = np.reshape(img, [row, col, image_size, image_size, num_channel])  # [row, col, h, w, c]
    else:
        img = np.reshape(img, [row, col, image_size, image_size])  # [row, col, h, w]
    img = unstack(img, axis=0)  # row * [col, h, w, c]
    img = np.concatenate(img, axis=1)  # [col, row*h, w, c]
    img = unstack(img, axis=0)  # col * [row*h, w, c]
    img = np.concatenate(img, axis=1)  # [row*h, col*w, c]
    return img


def unstack(img, axis):
    d = img.shape[axis]
    arr = [np.squeeze(a, axis=axis) for a in np.split(img, d, axis=axis)]
    return arr


def save_images(img, sample_file, num_samples, nums_class, k_dim=1, image_size=128, num_channel=3):
    n_rows = num_samples * k_dim
    n_cols = nums_class
    img = make3d(img, num_channel=num_channel, image_size=image_size, row=n_rows, col=n_cols)
    img = inverse_image(img)
    scm.imsave(sample_file, img)


def save_image(img, sample_file):
    img = inverse_image(img)
    scm.imsave(sample_file, img)


def save_batch_images(imgs, sample_files, ind_generation_dim, ind_nums_class, label_scaler, output_dir,
                      has_extension=True):
    exported_dict = {}
    for i in range(len(sample_files)):
        dim = ind_generation_dim[i]
        # flip classification outcome
        unscaled_cls = 1 - int(ind_nums_class[i])
        cls = int(unscaled_cls * label_scaler)
        img = inverse_image(imgs[i, dim, cls])
        if has_extension:
            img_name = '{}_dim_{}_cls_{}.{}'.format(sample_files[i][:-4], dim, cls, sample_files[i][-3:])
        else:
            img_name = '{}_dim_{}_cls_{}.jpg'.format(sample_files[i], dim, cls)
        img_dir_name = os.path.join(output_dir, img_name)
        scm.imsave(img_dir_name, img)
        exported_dict[img_name] = unscaled_cls
    return exported_dict


def save_dict(my_dict, filename, attribute):
    first_line = str(len(my_dict.keys())) + '\n'
    second_line = ''.join(attribute) + "\n"
    with open(filename, 'w') as f:
        f.write(first_line)
        f.write(second_line)
        for key in my_dict.keys():
            f.write("{} {}\n".format(key, my_dict[key]))


def save_config_dict(my_dict, filename):
    with open(filename, 'w') as f:
        for key in my_dict.keys():
            val = my_dict[key]
            if isinstance(val, str):
                f.write("{}: '{}'\n".format(key, val))
            else:
                f.write("{}: {}\n".format(key, val))


def calc_metrics_arr(prediction, labels, average='binary'):
    acc = accuracy_score(labels, prediction)
    precision = precision_score(labels, prediction, average=average)
    recall = recall_score(labels, prediction, average=average)
    return acc, precision, recall


def calc_accuracy(prediction, labels):
    # even for a binary classification, we have two classes, hence complexity of this
    acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.reduce_sum(tf.cast(tf.equal(tf.math.round(prediction), labels), dtype=tf.int32), axis=1),
                 tf.shape(labels)[1]), tf.float32)) * 100.0
    return acc


def calc_accuracy_with_logits(logits, labels):
    return calc_accuracy(tf.nn.sigmoid(logits), labels)


# To avoid memory issues: https://github.com/tensorflow/tensorflow/issues/9545
def absolute_variable_scope(name_or_scope, reuse=tf.AUTO_REUSE):
    current_scope = tf.get_default_graph().get_name_scope()
    if not current_scope:
        if name_or_scope.endswith('/'):
            variable_scope = tf.variable_scope(name_or_scope, reuse=reuse)
        else:
            variable_scope = tf.variable_scope('{}/'.format(name_or_scope), reuse=reuse)
    else:
        variable_scope = tf.variable_scope('{}/{}/'.format(current_scope, name_or_scope), reuse=reuse)
    return variable_scope


def absolute_name_scope(scope, reuse=tf.AUTO_REUSE):
    """Builds an absolute tf.name_scope relative to the current_scope.
  This is helpful to reuse nested name scopes.

  E.g. The following will happen when using regular tf.name_scope:

    with tf.name_scope('outer'):
      with tf.name_scope('inner'):
        print(tf.constant(1)) # Will print outer/inner/Const:0
    with tf.name_scope('outer'):
      with tf.name_scope('inner'):
        print(tf.constant(1)) # Will print outer/inner_1/Const:0

  With absolute_name_scope:

    with absolute_name_scope('outer'):
      with absolute_name_scope('inner'):
        print(tf.constant(1)) # Will print outer/inner/Const:0
    with absolute_name_scope('outer'):
      with absolute_name_scope('inner'):
        print(tf.constant(1)) # Will print outer/inner/Const_1:0
  """
    current_scope = tf.get_default_graph().get_name_scope()
    if not current_scope:
        if scope.endswith('/'):
            scope = tf.variable_scope(scope, reuse=reuse)
        else:
            scope = tf.variable_scope('{}/'.format(scope), reuse=reuse)
    else:
        scope = tf.variable_scope('{}/{}/'.format(current_scope, scope), reuse=reuse)
    return scope


def convert_ordinal_to_binary(y, n):
    y = np.asarray(y).astype(int)
    new_y = np.zeros([y.shape[0], n])
    new_y[:, 0] = y
    for i in range(0, y.shape[0]):
        for j in range(1, y[i] + 1):
            new_y[i, j] = 1
    return new_y


def safe_append(all_arr, curr_arr, axis=0):
    if np.size(all_arr) == 0:
        res = curr_arr
    else:
        res = np.append(all_arr, curr_arr, axis=axis)
    return res