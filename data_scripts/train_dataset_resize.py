import os
import glob
import cv2
import numpy as np
import sys

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    mod_scale = 4
    up_scale = 4
    folder4k = '/lustre/home/acct-eezy/eezy/4khdr/data/Dataset/train_4k'
    output = '/lustre/home/acct-eezy/eezy/4khdr/data/Dataset/train_4k_resize_to_540p_gt'

    folders = sorted(os.listdir(folder4k))

    print(folders)

    len_folders = len(folders)

    mkdir(output)

    for i,folder in enumerate(folders):

        print(folder)
        path4k = os.path.join(folder4k, folder)
        pathoutput = os.path.join(output, folder)
        images = sorted(os.listdir(path4k))
        
        mkdir(pathoutput)

        

        for img in images:
                img_path_4k = os.path.join(path4k, img)
                img_path_out = os.path.join(pathoutput, img)

                print('No.{} -- Processing {}'.format(i, img))
                # read image
                image = cv2.imread(os.path.join(img_path_4k), cv2.IMREAD_UNCHANGED)

                width = int(np.floor(image.shape[1] / mod_scale))
                height = int(np.floor(image.shape[0] / mod_scale))

                dim = (width, height)

                # resize image
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)

                # modcrop
                # if len(image.shape) == 3:
                #         image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
                # else:
                #         image_HR = image[0:mod_scale * height, 0:mod_scale * width]
                # # LR
                # image_LR = imresize_np(image_HR, 1 / up_scale, True)
                # # bic
                # image_Bic = imresize_np(image_LR, up_scale, True)


                # cv2.imwrite(os.path.join(img_path_out), image_HR)
                cv2.imwrite(os.path.join(img_path_out), resized)


    print('Finished.')

if __name__ == "__main__":
    main()