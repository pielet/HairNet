# Author: Shiyang Jia

import os
import cv2
import numpy as np
import h5py


class Dataloader(object):
    """load data with ground truth for HairNet training and testing"""

    def __init__(self, data_path, batch_size):
        """
        Dataloader variables
        param			description				        size				type
        self.train_x 	list of training image 	   		(h, w, channel=2)	uint8 -> float64
        self.train_y    dict of ground truth        	(32, 32, x)         float64
        self.test_x     list of testing image 	   		(h, w, channel=2)	uint8 -> float64
        self.test_y		list of loss weight				(32, 32, x) 		float64
        """
        self.batch_size = batch_size
        self.train_x_dir = os.path.join(data_path, 'train_x_png')
        self.train_y_dir = os.path.join(data_path, 'train_y')
        self.test_x_dir = os.path.join(data_path, 'test_x_png')
        self.test_y_dir = os.path.join(data_path, 'test_y')
        self.pos_mean, self.pos_std, self.curv_mean, self.curv_std = get_mean_std(data_path)
        self.order = list(range(126))       # load test data first

    def load_image(self, image_path, begin, end):
        images = []
        file_list = os.listdir(image_path)
        file_list.sort()
        files = []
        for i in range(begin, end):
            files.append(file_list[self.order[i]])
        for file in files:
            img = cv2.imread(os.path.join(image_path, file), cv2.IMREAD_COLOR)
            img = img.astype(np.float64) / 255
            images.append(img[:, :, (2, 0)])  # (R, B)
        return images

    def load_gt(self, h5_dir, begin, end):
        gt = {'angle': [], 'pos': [], 'curv': [], 'weight': []}
        file_list = os.listdir(h5_dir)
        file_list.sort()
        files = []
        for i in range(begin, end):
            files.append(file_list[self.order[i]])
        for file in files:
            with h5py.File(os.path.join(h5_dir, file), 'r') as h5:
                gt['angle'].append(h5['angle'].value)
                gt['pos'].append(h5['pos'][:].reshape(32, 32, 300))
                gt['curv'].append(h5['curv'][:])
                gt['weight'].append(h5['weight'][:])
        return gt

    def flesh_batch_order(self):
        self.order = np.random.permutation(1800)    # set when training
        # self.order = list(range(1800))

    def get_train_batch(self, i):
        """used in training mode
        Args:
            i: the i-th batch according to self.order
        Return:
            encoder_input: [batch_size, 256, 256, 2]
            decoder_output: [batch_size, 32, 32, 500] -- weight + pos + curv
        """
        # load data
        n = self.batch_size
        begin, end = i * n, (i + 1)*n
        train_x = self.load_image(self.train_x_dir, begin, end)
        train_y = self.load_gt(self.train_y_dir, begin, end)

        # put all the data into big arrays
        encoder_input = np.zeros((n, 256, 256, 2), dtype=np.float64)
        decoder_output = np.zeros((n, 32, 32, 500), dtype=np.float64)    # weight + pos + curv
        for i in range(n):
            encoder_input[i] = train_x[i]
            norm_pos = (train_y['pos'][i] - self.pos_mean) / self.pos_std
            norm_curv = (train_y['curv'][i] - self.curv_mean) / self.curv_std
            decoder_output[i] = np.concatenate((train_y['weight'][i], norm_pos, norm_curv), axis=2)
        return encoder_input, decoder_output

    def get_test_data(self):
        """used in training mode as a test after each epoch"""

        test_x = self.load_image(self.test_x_dir, 0, 126)
        test_y = self.load_gt(self.test_y_dir, 0, 126)

        # put all the data into big arrays
        n = len(test_x)
        encoder_input = np.zeros((n, 256, 256, 2), dtype=np.float64)
        decoder_output = np.zeros((n, 32, 32, 500), dtype=np.float64)
        for i in range(n):
            encoder_input[i] = test_x[i]
            norm_pos = (test_y['pos'][i] - self.pos_mean) / self.pos_std
            norm_curv = (test_y['curv'][i] - self.curv_mean) / self.curv_std
            decoder_output[i] = np.concatenate((test_y['weight'][i], norm_pos, norm_curv), axis=2)

        return encoder_input, decoder_output, np.array(test_y['angle'])


def compute_mean_std(data_dir):
    """
    compute position mean/std and curvature mean/std among training samples
    store in mean_std.h5 file under data_dir and return
    """
    train_y_dir = os.path.join(data_dir, 'train_y')
    file_list = os.listdir(train_y_dir)

    # compute mean
    sum_pos = np.zeros((32, 32, 300), dtype=np.float64)
    sum_curv = np.zeros((32, 32, 100), dtype=np.float64)
    for file in file_list:
        with h5py.File(os.path.join(train_y_dir, file), 'r') as h5:
            sum_pos += h5['pos'][:].reshape(32, 32, 300)
            sum_curv += h5['curv'][:]

    pos_mean = sum_pos / 1800
    curv_mean = sum_curv / 1800

    # compute std
    sum_square_pos = np.zeros((32, 32, 300), dtype=np.float64)
    sum_square_curv = np.zeros((32, 32, 100), dtype=np.float64)

    for file in file_list:
        with h5py.File(os.path.join(train_y_dir, file), 'r') as h5:
            sum_square_pos += np.square(h5['pos'][:].reshape(32, 32, 300) - pos_mean)
            sum_square_curv += np.square(h5['curv'][:] - curv_mean)

    pos_std = np.sqrt(sum_square_pos / 1800)
    curv_std = np.sqrt(sum_square_curv / 1800)
    pos_std[pos_std == 0] = 1   # skip 0
    curv_std[curv_std == 0] = 1

    # write into h5 file under data_dir
    with h5py.File(os.path.join(data_dir, 'mean_std.h5'), 'w') as h5:
        h5.create_dataset('pos_mean', data=pos_mean)
        h5.create_dataset('pos_std', data=pos_std)
        h5.create_dataset('curv_mean', data=curv_mean)
        h5.create_dataset('curv_std', data=curv_std)

    return pos_mean, pos_std, curv_mean, curv_std


def get_mean_std(data_dir):
    """Return position & curvature mean, std (computed / loaded)"""

    file_path = os.path.join(data_dir, 'mean_std.h5')
    if os.path.exists(file_path):
        # read in mean/std
        with h5py.File(file_path, 'r') as h5:
            pos_mean = h5['pos_mean'][:]
            pos_std = h5['pos_std'][:]
            curv_mean = h5['curv_mean'][:]
            curv_std = h5['curv_std'][:]
        return pos_mean, pos_std, curv_mean, curv_std
    else:
        return compute_mean_std(data_dir)


def load_root(data_dir):
    """load hair root position and scalp-coordinate mask
    Args:
        data_dir: data_path/root_param.h5
    Return:
         root: (32, 32, 3) root position
         mask: [bool] (32, 32) true if there grows a strand
    """
    file_path = os.path.join(data_dir, 'root_param.h5')
    with h5py.File(file_path, 'r') as h5:
        root = h5['pos'][:]
        short_mask = h5['mask'][:]
        mask = short_mask > 0

    # put root in 32x32 array
    grid_root = np.zeros((32, 32, 3), dtype=float)
    idx = 0
    for i in range(32):
        for j in range(32):
            if mask[i, j]:
                grid_root[i, j] = root[idx]
                idx += 1

    return grid_root, mask


def load_real_image(data_dir):
    """ load test data
    Return:
        image_arr: (n, 256, 256, 2) numpy array
    """
    images = []
    files = os.listdir(data_dir)
    for file in files:
        img = cv2.imread(os.path.join(data_dir, file), cv2.IMREAD_COLOR)
        img = img.astype(np.float64) / 255
        images.append(img[:, :, (2, 0)])  # (R, B)

    # put all images into big array
    n = len(images)
    image_arr = np.zeros((n, 256, 256, 2), dtype=np.float64)
    for i in range(n):
        image_arr[i] = images[i]

    return image_arr
