'''
REDS dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import json
import torch.utils.data as data
import data.util as util
try:
    import mc  # import memcached
except ImportError:
    pass

logger = logging.getLogger('base')


class KESAIDataset(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(KESAIDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs

        #### directly load image keys
        if self.data_type == 'img':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT']) 
        else:
            raise ValueError(
                'Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')


        # remove the REDS4 for testing
        self.paths_GT = [
            v for v in self.paths_GT if v.split('/')[-2] not in ['23381522', '62600438', '63056025']
        ]
        assert self.paths_GT, 'Error: GT path is empty.'


        logger.info('Use {}'.format(self.opt['color']))
                

    def __getitem__(self, index):

        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]

        name_a = key.split('/')[-2]
        name_b = key.split('/')[-1][:-4]

        center_frame_idx = int(name_b)

        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            N_frames = self.opt['N_frames']
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 99:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 0:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval))
            name_b = '{:05d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders

            while (center_frame_idx + self.half_N_frames * interval > 99) \
                    or (center_frame_idx - self.half_N_frames * interval < 0):  # check notation shenwang
                center_frame_idx = random.randint(0, 99)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:05d}'.format(neighbor_list[self.half_N_frames])
            key = name_a + '_' + name_b #todo
        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))



        #### get the GT image (as the center frame)
        img_GT = util.read_img(None, osp.join(self.GT_root, name_a, name_b + '.png'))

        #### get LQ images
        LQ_size_tuple = (3, 540, 960) if self.LR_input else (3, 2160, 3840)
        img_LQ_l = []
        for v in neighbor_list:
            img_LQ_path = osp.join(self.LQ_root, name_a, '{:05d}.png'.format(v))
            img_LQ = util.read_img(None, img_LQ_path)
            img_LQ_l.append(img_LQ)

        assert key == name_a + '_' + name_b

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale

                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))

                img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            else:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

            #color
            if self.opt['color'] == 'YUV':
                img_LQ_l = [util.bgr2ycbcr(v,only_y=False)  for v in img_LQ_l]
                img_GT = util.bgr2ycbcr(img_GT,only_y=False)


        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB
        img_GT = img_GT[:, :, [2, 1, 0]] # HWC
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]

        # HWC to CHW  numpy to tensor
        # if YUV -> VUY
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,(0, 3, 1, 2)))).float()

        return {'LQs': img_LQs, 'GT': img_GT, 'key': key}

    def __len__(self):
        return len(self.paths_GT)
