'''
Eval 4kHDR validation set.
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import time
import torch
from AverageMeter import *

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch
import itertools
import options.options as option
import argparse

import matplotlib.pyplot as plt


def main():
    #################
    # configurations
    #################
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, required=True)
    #parser.add_argument("--screen_notation", type=str, required=True)
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)

    PAD = 32

    total_run_time = AverageMeter()
    print("GPU ", torch.cuda.device_count())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda')
    
    data_mode = 'sharp_bicubic' 
    flip_test = False

    Input_folder = args.input_path
    GT_folder = args.gt_path
    Result_folder = args.output_path
    Model_path = args.model_path

    # create results folder
    if not os.path.exists(Result_folder):
        os.makedirs(Result_folder, exist_ok=True)

    model_path = Model_path

    N_in = 5

    model = EDVR_arch.EDVR(nf=opt['network_G']['nf'], nframes=opt['network_G']['nframes'],
                           groups=opt['network_G']['groups'], front_RBs=opt['network_G']['front_RBs'],
                           back_RBs=opt['network_G']['back_RBs'],
                           predeblur=opt['network_G']['predeblur'], HR_in=opt['network_G']['HR_in'],
                           w_TSA=opt['network_G']['w_TSA'])


    #### dataset
    test_dataset_folder = Input_folder
    GT_dataset_folder = GT_folder

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    padding = 'new_info'
    save_imgs = True

    save_folder = os.path.join(Result_folder, data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    avg_rgb_psnr_l, avg_rgb_psnr_center_l, avg_rgb_psnr_border_l = [], [], []

    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))

    end = time.time()

    for subfolder in subfolder_l:

        input_subfolder = os.path.split(subfolder)[1]

        subfolder_GT = os.path.join(GT_dataset_folder, input_subfolder)

        if not os.path.exists(subfolder_GT):
            continue

        print("Evaluate Folders: ", input_subfolder)

        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images, notice we load yuv img here
        imgs_LQ = data_util.read_img_seq_yuv(subfolder)  # Num x 3 x H x W
        img_GT_l = []
        for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
            img_GT_l.append(data_util.read_img_yuv(None, img_GT_path))

        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0
        avg_rgb_psnr, avg_rgb_psnr_border, avg_rgb_psnr_center = 0, 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)  # 960 x 540

            # here we split the input images 960x540 into 9 320x180 patch
            gtWidth = 3840
            gtHeight = 2160
            intWidth_ori = imgs_in.shape[4]  # 960
            intHeight_ori = imgs_in.shape[3]  # 540
            scale = 4
 
            intPaddingRight = PAD   # 32# 64# 128# 256
            intPaddingLeft = PAD    # 32#64 #128# 256
            intPaddingTop = PAD     # 32#64 #128#256
            intPaddingBottom = PAD  # 32#64 # 128# 256

            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

            imgs_in = torch.squeeze(imgs_in, 0)  # N C H W

            imgs_in = pader(imgs_in)  # N C 604 1024

            # todo: output 4k

            X0 = imgs_in

            X0 = torch.unsqueeze(X0, 0)
            if flip_test:
                output = util.flipx4_forward(model, X0)
            else:
                output = util.single_forward(model, X0)

            # todo remove padding
            output = output[0, :, intPaddingTop * scale:(intPaddingTop + intHeight_ori) * scale,
                                  intPaddingLeft * scale: (intPaddingLeft + intWidth_ori) * scale]

            output = util.tensor2img(output.squeeze(0))

            print("*****************current image process time \t " + str(
                time.time() - end) + "s ******************")

            total_run_time.update(time.time() - end, 1)

            # calculate PSNR on YUV
            y_all = output / 255.
            GT = np.copy(img_GT_l[img_idx])

            y_all, GT = util.crop_border([y_all, GT], crop_border)
            crt_psnr = util.calculate_psnr(y_all * 255, GT * 255)
            logger.info('{:3d} - {:25} \tYUV_PSNR: {:.6f} dB'.format(img_idx + 1, img_name, crt_psnr))


            # here we also calculate PSNR on RGB
            y_all_rgb = data_util.ycbcr2rgb(output / 255.)
            GT_rgb = data_util.ycbcr2rgb(np.copy(img_GT_l[img_idx])) 
            y_all_rgb, GT_rgb = util.crop_border([y_all_rgb, GT_rgb], crop_border)
            crt_rgb_psnr = util.calculate_psnr(y_all_rgb * 255, GT_rgb * 255)
            logger.info('{:3d} - {:25} \tRGB_PSNR: {:.6f} dB'.format(img_idx + 1, img_name, crt_rgb_psnr))


            if save_imgs:
                im_out = np.round(y_all_rgb*255.).astype(numpy.uint8)
                # todo, notice here we got rgb img, but cv2 need bgr when saving a img
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), cv2.cv2Color(im_out, cv2.COLOR_RGB2BGR))

            # for YUV and RGB, respectively
            if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                avg_rgb_psnr_center += crt_rgb_psnr
                N_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                avg_rgb_psnr_border += crt_rgb_psnr
                N_border += 1

        # for YUV
        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

        logger.info('Folder {} - Average YUV PSNR: {:.6f} dB for {} frames; '
                    'Center YUV PSNR: {:.6f} dB for {} frames; '
                    'Border YUV PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
                                                                   (N_center + N_border),
                                                                   avg_psnr_center, N_center,
                                                                   avg_psnr_border, N_border))

        # for RGB
        avg_rgb_psnr = (avg_rgb_psnr_center + avg_rgb_psnr_border) / (N_center + N_border)
        avg_rgb_psnr_center = avg_rgb_psnr_center / N_center
        avg_rgb_psnr_border = 0 if N_border == 0 else avg_rgb_psnr_border / N_border
        avg_rgb_psnr_l.append(avg_rgb_psnr)
        avg_rgb_psnr_center_l.append(avg_rgb_psnr_center)
        avg_rgb_psnr_border_l.append(avg_rgb_psnr_border)

        logger.info('Folder {} - Average RGB PSNR: {:.6f} dB for {} frames; '
                    'Center RGB PSNR: {:.6f} dB for {} frames; '
                    'Border RGB PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_rgb_psnr,
                                                                   (N_center + N_border),
                                                                   avg_rgb_psnr_center, N_center,
                                                                   avg_rgb_psnr_border, N_border))

        
    logger.info('################ Tidy Outputs ################')
    # for YUV
    for subfolder_name, psnr, psnr_center, psnr_border in zip(subfolder_name_l, avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l):
        logger.info('Folder {} - Average YUV PSNR: {:.6f} dB. '
                    'Center YUV PSNR: {:.6f} dB. '
                    'Border YUV PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center, psnr_border))

    # for RGB
    for subfolder_name, psnr, psnr_center, psnr_border in zip(subfolder_name_l, avg_rgb_psnr_l, avg_rgb_psnr_center_l, avg_rgb_psnr_border_l):
        logger.info('Folder {} - Average RGB PSNR: {:.6f} dB. '
                    'Center RGB PSNR: {:.6f} dB. '
                    'Border RGB PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center, psnr_border))


    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Total Average YUV PSNR: {:.6f} dB for {} clips. '
                'Center YUV PSNR: {:.6f} dB. Border YUV PSNR: {:.6f} dB.'.format(
        sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
        sum(avg_psnr_center_l) / len(avg_psnr_center_l),
        sum(avg_psnr_border_l) / len(avg_psnr_border_l)))
    logger.info('Total Average RGB PSNR: {:.6f} dB for {} clips. '
                'Center RGB PSNR: {:.6f} dB. Border RGB PSNR: {:.6f} dB.'.format(
        sum(avg_rgb_psnr_l) / len(avg_rgb_psnr_l), len(subfolder_l),
        sum(avg_rgb_psnr_center_l) / len(avg_rgb_psnr_center_l),
        sum(avg_rgb_psnr_border_l) / len(avg_rgb_psnr_border_l)))


if __name__ == '__main__':
    main()
