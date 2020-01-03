import os
import glob

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass


def main():
    folder4k = '/DATA7_DB7/data/4khdr/data/Dataset/train_4k'
    folder540p = '/DATA7_DB7/data/4khdr/data/Dataset/train_540p'

    folders = sorted(os.listdir(folder4k))
    folders_540p = sorted(os.listdir(folder540p))
    len_folders = len(folders)

    for i,folder in enumerate(folders):

        print(folder)
        path4k = os.path.join(folder4k, folder)
        path540p = os.path.join(folder540p, folder)

        images = sorted(os.listdir(path540p))

        for idx, img in enumerate(images):

                img_path_4k = os.path.join(path4k, img)
                print('No.{} -- Processing {}'.format(idx, img_path_4k.split[-1]))
                # read image
                image = cv2.imread(os.path.join(img_path_4k))
                width = int(np.floor(image.shape[1] / mod_scale))
                height = int(np.floor(image.shape[0] / mod_scale))
                # modcrop
                if len(image.shape) == 3:
                image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
                else:
                image_HR = image[0:mod_scale * height, 0:mod_scale * width]
                # LR
                image_LR = imresize_np(image_HR, 1 / up_scale, True)
                # bic
                image_Bic = imresize_np(image_LR, up_scale, True)

                cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
                cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
                cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)



    print('Finished.')


def DIV2K(path):
    img_path_l = glob.glob(os.path.join(path, '*'))
    for img_path in img_path_l:
        new_path = img_path.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')
        os.rename(img_path, new_path)


if __name__ == "__main__":
    main()