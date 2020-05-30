import numpy as np
import os
import sys
import pandas as pd
import zipfile
import argparse
from tqdm import tqdm
import pdb
from utils import *
from sklearn.model_selection import train_test_split
import h5py
np.random.seed(0)


def dataset_split(all_images, output_dir):
    X_train, X_test = train_test_split(all_images, test_size=0.33, random_state=0)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    print(X_train.shape, X_test.shape)
    np.save(os.path.join(output_dir, 'train_ids.npy'), X_train)
    np.save(os.path.join(output_dir, 'test_ids.npy'), X_test)


def find_single_attribute_ind(categories, attribute):
    # attribute: Target attribute for binary classification
    index = np.where(np.asarray(categories) == attribute)
    index = index[0][0]
    return index


def find_attribute_index(categories, attribute):
    # attribute: Target attribute for binary classification
    index_main = []
    for a in attribute:
        print(a)
        index = np.where(np.asarray(categories) == a)
        index = index[0][0]
        index_main.append(index)
    print(index_main)
    return index_main


def save_processed_label_file(df, output_dir, attribute):
    file_name = ''.join(attribute) + '_binary_classification.txt'
    df.to_csv(os.path.join(output_dir, file_name), sep=' ', index=None, header=None)
    print(df.shape)
    one_line = str(df.shape[0]) + '\n'
    second_line = ''.join(attribute) + "\n"
    with open(os.path.join(output_dir, file_name), 'r+') as fp:
        lines = fp.readlines()  # lines is list of line, each element '...\n'
        lines.insert(0, one_line)  # you can use any index if you know the line index
        lines.insert(1, second_line)
        fp.seek(0)  # file pointer locates at the beginning to write the whole file again
        fp.writelines(lines)


# Write the label file for target attribute binary classification
def write_attribute_label_file(df, categories, attribute, output_dir):
    index_main = find_attribute_index(categories, attribute)
    # Train File
    df_temp = df[['Image_Path'] + index_main]
    save_processed_label_file(df_temp, output_dir, attribute)


# Read saved files
def read_saved_files(attribute, output_dir, image_dir):
    file_name = ''.join(attribute) + '_binary_classification.txt'
    categories, file_names_dict = read_data_file(os.path.join(output_dir, file_name), image_dir)
    categories = np.asarray(categories).ravel()
    print(categories)

    print("Number of images: ", len(file_names_dict.keys()))
    print("Few image names:")
    list(file_names_dict.keys())[0:5]

    label = file_names_dict[list(file_names_dict.keys())[0]]
    print(type(label))
    label = np.asarray(label)
    print(label.ravel())


def prep_celeba(attributes=[['Smiling'], ['Young'], ['No_Beard'], ['Heavy_Makeup'], ['Black_Hair'], ['Bangs']]):
    # final paths
    celebA_dir = os.path.join('data', 'CelebA')
    image_dir = os.path.join(celebA_dir, 'images')
    txt_dir = os.path.join(celebA_dir, 'list_attr_celeba.txt')
    print('Image Dir: ', image_dir)
    print('Label File: ', txt_dir)

    # Divide dataset into train and test set
    all_images = os.listdir(image_dir)

    dataset_split(all_images, celebA_dir)

    # Read Label File
    categories, file_names_dict = read_data_file(txt_dir)
    categories = np.asarray(categories).ravel()
    print(categories)

    print("Number of images: ", len(file_names_dict.keys()))

    label = file_names_dict[list(file_names_dict.keys())[0]]
    print(type(label))
    label = np.asarray(label)
    print(label.ravel())

    # Create Binary-Classification Data file
    # Convert the dictionary: attr_list to a dataframe
    df = pd.DataFrame(file_names_dict).T
    df['Image_Path'] = df.index

    for attribute in attributes:
        write_attribute_label_file(df, categories, attribute, celebA_dir)

    for attribute in attributes:
        read_saved_files(attribute, celebA_dir, image_dir)


def prep_celeba_biased():
    attribute = 'Smiling'
    # Attribute is Smiling
    # however, confounded with Young and Blond.
    # Meaning that positive examples are also Young and Blond
    # And negative examples are old and dark haired

    # final paths
    celebA_dir = os.path.join('data', 'CelebA')
    image_dir = os.path.join(celebA_dir, 'images')
    txt_dir = os.path.join(celebA_dir, 'list_attr_celeba.txt')
    biased_celebA_dir = os.path.join(celebA_dir, 'biased')
    if not os.path.exists(biased_celebA_dir):
        os.makedirs(biased_celebA_dir)
    print('Image Dir: ', image_dir)
    print('Label File: ', txt_dir)

    # Read Label File
    categories, all_file_names_dict = read_data_file(txt_dir)
    categories = np.asarray(categories).ravel()

    file_names_dict = {}
    for img in all_file_names_dict.keys():
        smiling = all_file_names_dict[img][find_single_attribute_ind(categories, 'Smiling')]
        young = all_file_names_dict[img][find_single_attribute_ind(categories, 'Young')]
        blond = all_file_names_dict[img][find_single_attribute_ind(categories, 'Blond_Hair')]
        if smiling == young and smiling == blond:
            file_names_dict.update({img: all_file_names_dict[img]})
    print(categories)

    # Divide dataset into train and test set
    all_images = list(file_names_dict.keys())

    dataset_split(all_images, biased_celebA_dir)

    print("Number of images: ", len(file_names_dict.keys()))

    label = file_names_dict[list(file_names_dict.keys())[0]]
    print(type(label))
    label = np.asarray(label)
    print(label.ravel())

    # Create Binary-Classification Data file
    # Convert the dictionary: attr_list to a dataframe
    df = pd.DataFrame(file_names_dict).T
    df['Image_Path'] = df.index

    write_attribute_label_file(df, categories, [attribute], biased_celebA_dir)


def prep_shapes(target_labels=['samecolor', 'redsamecolor', 'redcolor', 'multicolor', 'redcyan']):
    shapes_dir = os.path.join('data', 'shapes')
    dataset = h5py.File(os.path.join(shapes_dir, '3dshapes.h5'), 'r')
    attributes = dataset['labels']  # array shape [480000, 6], float64
    # images = dataset['images']  # array shape [480000, 64, 64, 3], uint8 in range(256)
    # image_shape = images.shape[1:]  # [64, 64, 3]
    # label_shape = labels.shape[1:]  # [6]
    n_samples = attributes.shape[0]  # 10 * 10 * 10 * 8 * 4 * 15 = 480000


    # _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
    #                      'orientation']
    # _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
    #                           'scale': 8, 'shape': 4, 'orientation': 15}

    # Divide dataset into train and test set
    all_images = np.arange(n_samples)
    dataset_split(all_images, shapes_dir)
    for target_label in target_labels:
        if target_label == 'samecolor':
            labels = ((attributes[:, 0] == attributes[:, 1]) & (attributes[:, 0] == attributes[:, 2])).astype(np.int32)
        elif target_label == 'redsamecolor':
            labels = ((attributes[:, 0] == attributes[:, 1]) & (attributes[:, 0] == attributes[:, 2]) & (
                        attributes[:, 0] == 0)).astype(np.int32)
        elif target_label == 'redcolor':
            labels = ((attributes[:, 0] == 0) | (attributes[:, 1] == 0) | (attributes[:, 2] == 0)).astype(np.int32)
        elif target_label == 'multicolor':
            labels = ((attributes[:, 0] == 0) | (attributes[:, 1] == 0.1) | (attributes[:, 2] == 0.2)).astype(np.int32)
        elif target_label == 'redcyan':
            labels = ((attributes[:, 0] == 0.5) | (attributes[:, 2] == 0)).astype(np.int32)
        shape_labels_df = pd.DataFrame(data={'filenames': all_images, target_label: labels})

        save_processed_label_file(shape_labels_df, shapes_dir, target_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeba', '-c', action='store_true')
    parser.add_argument('--shapes', '-s', action='store_true')
    parser.add_argument('--celeba_biased', '-cb', action='store_true')
    args = parser.parse_args()
    if args.shapes:
        prep_shapes()
    if args.celeba:
        prep_celeba()
    if args.celeba_biased:
        prep_celeba_biased()
