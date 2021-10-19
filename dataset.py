import cv2
import numpy as np
import os

def transform_img(opt, img):
    img = cv2.resize(img, (opt.imgsize, opt.imgsize))
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')
    img = img / 255.
    img = img * 2.0 - 1.0
    return img

def data_loader(opt):
    def reader():
        filenames = os.listdir(opt.train_data_path)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(opt.train_data_path, name)
            img = cv2.imread(filepath)
            img = transform_img(opt, img)
            if name[0] == 'H' or name[0] == 'N':
                label = 0
            elif name[0] == 'P':
                label = 1
            else:
                raise('Not excepted file name')
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == opt.batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader

def valid_data_loader(opt):
    filelists = open(opt.val_label_path).readlines()
    def reader():
        batch_imgs = []
        batch_labels = []
        for line in filelists[1:]:
            line = line.strip().split(',')
            name = line[1]
            label = int(line[2])
            filepath = os.path.join(opt.val_data_path, name)
            img = cv2.imread(filepath)
            img = transform_img(opt, img)
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == opt.batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader