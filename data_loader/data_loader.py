import numpy as np
from tqdm import tqdm
import scipy.misc as scm
import os
from utils import crop_center
import h5py
import pdb
from utils import read_data_file


class DataLoader:

    def __init__(self):
        return

    def load_images_and_labels(self, imgs_names, image_dir, n_class, file_names_dict, input_size=128, num_channel=3,
                               do_center_crop=False):
        print("Error! load_images_and_labels should be overwritten in child class")
        raise NotImplementedError


class CelebALoader(DataLoader):
    def __init__(self):
        DataLoader.__init__(self)

    def load_images_and_labels(self, imgs_names, image_dir, n_class, file_names_dict, input_size=128, num_channel=3,
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
                labels[i] = file_names_dict[img_name]
            except:
                print(img_name)
        labels[np.where(labels == -1)] = 0
        return imgs, labels


class ShapesLoader(DataLoader):
    def __init__(self, project_dir='.', dbg_mode=False, dbg_size=32, dbg_n_bins=2,
                 dbg_max_samples_per_bin=5000):
        shapes_dir = os.path.join(project_dir, 'data', 'shapes')
        self.dbg_mode = dbg_mode
        dataset = h5py.File(os.path.join(shapes_dir, '3dshapes.h5'), 'r')
        if self.dbg_mode:
            print('Debug mode activated. Only a few samples from the shapes datasets will be considered.')
            _, file_names_dict = read_data_file(
                os.path.join(project_dir,
                'output/classifier/shapes-redcolor/explainer_input/list_attr_{}_{}.txt'.format(
                    dbg_n_bins, dbg_max_samples_per_bin)))
            _tmp_list = list(file_names_dict.keys())[:dbg_size]
            self.tmp_list = list(np.sort([int(ind) for ind in _tmp_list]))
            self.images = np.array(dataset['images'][self.tmp_list])
        else:
            self.images = np.array(dataset['images'])  # array shape [480000, 64, 64, 3], uint8 in range(256)
        self.images = self.images / 255.0
        self.images = self.images - 0.5
        self.images = self.images * 2.0
        self.attributes = np.array(dataset['labels'])
        self._image_shape = self.images.shape[1:]  # [64, 64, 3]
        self._label_shape = self.attributes.shape[1:]  # [6]
        self._n_samples = self.attributes.shape[0]  # 10 * 10 * 10 * 8 * 4 * 15 = 480000

        self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                                  'orientation']
        self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                                       'scale': 8, 'shape': 4, 'orientation': 15}
        # same color label
        # self.labels = (self.attributes[:, 0] == self.attributes[:, 1]) & (
        #             self.attributes[:, 0] == self.attributes[:, 2])

    def load_images_and_labels(self, imgs_names, image_dir, n_class, file_names_dict, input_size=64, num_channel=3,
                               do_center_crop=False):
        assert n_class == 1
        assert input_size == 64
        assert num_channel == 3
        del image_dir, do_center_crop
        # Currently not handling resizing etc
        # cur_labels = self.labels[imgs_names]
        labels = np.zeros((imgs_names.shape[0], n_class), dtype=np.float32)
        for i, img_name in tqdm(enumerate(imgs_names)):
            labels[i] = file_names_dict[str(img_name)]
        if self.dbg_mode:
            tmp_inds = [self.tmp_list.index(int(ind)) for ind in imgs_names]
            return self.images[tmp_inds], labels
        else:
            return self.images[imgs_names.astype(np.int32)], labels

