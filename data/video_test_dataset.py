import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import json
import numpy as np

class VideoTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        #### Generate data info and cache data
        self.imgs_LQ, self.imgs_GT = {}, {}
        if opt['name'].lower() in ['vid4', 'reds4']:
            subfolders_LQ = util.glob_file_list(self.LQ_root)
            subfolders_GT = util.glob_file_list(self.GT_root)

            # add test
            if opt['use_all_folders'] == False:
                subfolders_LQ = subfolders_LQ[1:2]
                subfolders_GT = subfolders_GT[1:2]

            for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
                subfolder_name = osp.basename(subfolder_GT)
                img_paths_LQ = util.glob_file_list(subfolder_LQ)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                max_idx = len(img_paths_LQ)
                assert max_idx == len(img_paths_GT), 'Different number of images in LQ and GT folders'
                self.data_info['path_LQ'].extend(img_paths_LQ)
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)


                if self.cache_data:
                    print('\nCaching Folder {} Validation Frames, Please wait\n'.format(subfolder_name))
                    self.imgs_LQ[subfolder_name] = util.read_img_seq_display(img_paths_LQ, '{}_LQ'.format(subfolder_name), color=opt['color'])
                    self.imgs_GT[subfolder_name] = util.read_img_seq_display(img_paths_GT, '{}_GT'.format(subfolder_name), color=opt['color'])
                    # self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ)
                    # self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT)
                else:
                    self.imgs_LQ[subfolder_name] = img_paths_LQ
                    self.imgs_GT[subfolder_name] = img_paths_GT


            # load screen notation
            # /DATA7_DB7/data/4khdr/data/Dataset/4khdr_frame_notation.json
            # with open(opt['frame_notation']) as f:
            #     self.frame_notation = json.load(f)

            print("\nVal Dataset Initialized\n")

        elif opt['name'].lower() in ['vimeo90k-test']:
            pass  # TODO
        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

    def check_notation(self, folder_key, center_index):
        """
        check not get dis-continous frames
        """
        notation = self.frame_notation[folder_key][center_index-2:center_index+3]
        if all(item == notation[0] for item in notation):
            return 1
        else:
            return 0

    def __getitem__(self, index):
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        if self.cache_data:
            select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],padding=self.opt['padding'])

            #select_idx = util.index_generation_process_screen_change_withlog_fixbug(folder, self.frame_notation, idx, max_idx, self.opt['N_frames'],
            #                                   padding=self.opt['padding'])

            # select_idx, log1, log2, nota = util.index_generation_process_screen_change_withlog_fixbug(
            #     folder, self.frame_notation, idx, max_idx, self.opt['N_frames'], padding=self.opt['padding'],
            #     enable=1)

            # if not log1 == None:
            #     print('screen change')
            #     print(nota)
            #     print(log1)
            #     print(log2)

            imgs_LQ = self.imgs_LQ[folder].index_select(0, torch.LongTensor(select_idx))
            img_GT = self.imgs_GT[folder][idx]


        else:
            select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],padding=self.opt['padding'])

            imgs_LQ_path = []
            for v in select_idx:
                imgs_LQ_path.append(self.imgs_LQ[folder][v])
            img_GT_path = self.imgs_GT[folder][idx]

            # read gt
            img_GT = util.read_img(None, img_GT_path)
            img_LQ_l = []
            for v in imgs_LQ_path:
                img_LQ = util.read_img(None, v)
                img_LQ_l.append(img_LQ)

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
            imgs_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,(0, 3, 1, 2)))).float()

        return {
            'LQs': imgs_LQ,
            'GT': img_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border
        }

    def __len__(self):
        return len(self.data_info['path_GT'])
