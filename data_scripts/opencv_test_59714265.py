import numpy 
import os
import cv2


video_path = '/lustre/home/acct-eezy/eezy/4khdr/data/train_LDR_540p'
save_path = '/lustre/home/acct-eezy/eezy/4khdr/data/Dataset/train_LDR_540p'

def create_png(video_path1, save_path1):

    if not os.path.isdir(save_path1):
        os.makedirs(save_path1, exist_ok=True)

    cap = cv2.VideoCapture(video_path1)
    c = 0
    retval = 1
    while retval:
        retval, image = cap.read()
        if (retval):
            cv2.imwrite(os.path.join(save_path1, '{:05d}.png'.format(c)),image)
        c += 1
        cv2.waitKey(1)
    cap.release()

if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

# videos = os.listdir(video_path)

# for video in videos:
video = '59714265.mp4'
temp_video_path = os.path.join(video_path, video)
temp_save_path = os.path.join(save_path, video[:-4])
create_png(temp_video_path, temp_save_path)
