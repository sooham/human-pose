from __future__ import print_function, division

import os
import numpy as np
import scipy.io as sio
from matplotlib.pyplot import imread
from scipy.misc import imresize

import csv
import lxml
from ConfigParser import SafeConfigParser

'''
    This file contains utility functions required for tasks
    such as: parsing datasets, saving images,
    creating animations etc.
'''

cwd = os.getcwd()

conf = SafeConfigParser()
conf.read('config.ini')


def rgb2gray(img):
    ''' converts image to color '''
    return (0.299 * img[...,0] + 0.587 * img[...,1] + 0.114 * img[...,2])

class PASCAL_context:
    ''' Class for reading the PASCAL Context 2010 dataset '''
    def __init__(self):
        pass

    def load_dataset(self, batch_size, bw=True):
        ''' Return labels with generator for images and segmented images '''
        seg_dir = conf.get('PASCAL_Context', 'segmented_images')
        train_dir = conf.get('PASCAL_Context', 'trainval_images')
        labels_file = conf.get('PASCAL_Context', 'labels')

        labels = None
        # convert labels to a dictionary
        with open(labels_file, 'r') as flabels:
            reader = csv.reader(flabels, delimiter=':')
            labels_dict = dict(reader)
            labels = np.array([labels_dict[str(k)] for k in range(1, len(labels_dict) + 1)])



        def test_generator():
            ''' a generator for test images'''
            i = 0

            test_images = list(set(
                [file[:-4] for file in os.listdir(seg_dir) if file.endswith('.mat')]
                ).intersection(
                [file[:-4] for file in os.listdir(train_dir) if file.endswith('.jpg')]
            ))[:1000]

            l = len(test_images)

            seg_images = [os.path.join(seg_dir, f + '.mat') for f in test_images]
            test_images = [os.path.join(train_dir, f + '.jpg') for f in test_images]

            while i < l:
                x = [imread(f) for f in test_images[i: i+1]]
                y = [sio.loadmat(f)['LabelMap'] / 459.0 for f in seg_images[i: i+batch_size]]

                if bw:
                    x = [rgb2gray(imresize(f, (96, 117))).ravel() for f in x]
                else:
                    x = [imresize(f, (96, 117)).ravel() for f in x]

                y = [imresize(f, (96, 117), mode='F', interp='nearest').ravel() * 459.0 for f in y]
                i += 1
                yield l, np.float32(x[0]), np.float32(y[0])


        def train_generator():
            ''' a generator for train images'''
            i = 0

            train_images = list(set(
                [file[:-4] for file in os.listdir(seg_dir) if file.endswith('.mat')]
                ).intersection(
                [file[:-4] for file in os.listdir(train_dir) if file.endswith('.jpg')]
            ))[1000:]

            l = len(train_images)

            seg_images = [os.path.join(seg_dir, f + '.mat') for f in train_images]
            train_images = [os.path.join(train_dir, f + '.jpg') for f in train_images]

            while i < l:
                if (i + batch_size) < l:
                    x = [imread(f) for f in train_images[i: i+batch_size]]
                    y = [sio.loadmat(f)['LabelMap'] / 459.0 for f in seg_images[i: i+batch_size]]
                else:
                    x = [imread(f) for f in train_images[i:]]
                    x += [imread(f) for f in train_images[:-l+i+batch_size]]
                    y = [sio.loadmat(f)['LabelMap'] / 459.0 for f in seg_images[i:]]
                    y += [sio.loadmat(f)['LabelMap'] / 459.0 for f in seg_images[:-l+i+batch_size]]

                if bw:
                    x = [rgb2gray(imresize(f, (96, 117))).ravel() for f in x]
                else:
                    x = [imresize(f, (96, 117)).ravel() for f in x]

                y = [imresize(f, (96, 117), mode='F', interp='nearest').ravel() * 459.0 for f in y]

                i = (i + batch_size) % l

                yield l, np.float32(np.vstack(x)), np.float32(np.vstack(y))

        return train_generator, test_generator, labels

if __name__ == '__main__':
    pascal_context = PASCAL_context()