import argparse
import cv2
import os
import sys
import torch
import warnings

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

dir_name = os.path.dirname(__file__)
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(dir_name, '../PoseEstimation/'))
sys.path.append(os.path.join(dir_name, '../utils'))
from prepare_source import extract_frames, load_openpose_model
# Import from openpose
from network.rtpose_vgg import get_model
from evaluate.coco_eval import get_multiplier, get_outputs
# Import from utils
from openpose_utils import remove_noise, get_pose
from torch_utils import get_torch_device

device = get_torch_device()


def prepare_target(save_dir):

    video_files = [f for f in os.listdir(save_dir) if f.endswith('.mp4')]
    assert (not len(video_files) < 1), "No mp4 file found!"
    assert (not len(video_files) > 1), "More than one video file found, make sure to have only one!"

    video_file = os.path.join(save_dir, video_files[0])

    img_dir = os.path.join(save_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    extract_frames(video_file, img_dir, max_frames=np.inf)

    model = load_openpose_model()

    extract_poses(model, save_dir)


def extract_poses(model, save_dir):
    '''make label images for pix2pix'''

    train_dir = os.path.join(save_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    train_img_dir = os.path.join(train_dir, 'train_img')
    os.makedirs(train_img_dir, exist_ok=True)
    train_label_dir = os.path.join(train_dir, 'train_label')
    os.makedirs(train_label_dir, exist_ok=True)
    train_head_dir = os.path.join(train_dir, 'head_img')
    os.makedirs(train_head_dir, exist_ok=True)

    img_dir = os.path.join(save_dir, 'images')

    pose_cords = []
    for idx in tqdm(range(len(os.listdir(img_dir)))):
        img_path = os.path.join(img_dir, '{:05}.png'.format(idx))
        img = cv2.imread(img_path)

        shape_dst = np.min(img.shape[:2])
        oh = (img.shape[0] - shape_dst) // 2
        ow = (img.shape[1] - shape_dst) // 2

        img = img[oh:oh + shape_dst, ow:ow + shape_dst]
        img = cv2.resize(img, (512, 512))
        multiplier = get_multiplier(img)
        with torch.no_grad():
            paf, heatmap = get_outputs(multiplier, img, model, 'rtpose', device)
        r_heatmap = np.array([remove_noise(ht)
                              for ht in heatmap.transpose(2, 0, 1)[:-1]]).transpose(1, 2, 0)
        heatmap[:, :, :-1] = r_heatmap
        param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
        label, cord = get_pose(param, heatmap, paf)
        index = 13
        crop_size = 25
        try:
            head_cord = cord[index]
        except IndexError:
            try:
                head_cord = pose_cords[-1]  # if there is not head point in picture, use last frame
            except IndexError:
                print("skipping 1st frame as pose detection failed")
                continue

        head = img[int(head_cord[1] - crop_size): int(head_cord[1] + crop_size),
                   int(head_cord[0] - crop_size): int(head_cord[0] + crop_size), :]
        try:           
            plt.imshow(head)
            plt.savefig(os.path.join(train_head_dir, 'pose_{}.jpg'.format(idx)))
            plt.clf()
            cv2.imwrite(os.path.join(train_img_dir, '{:05}.png'.format(idx)), img)
            cv2.imwrite(os.path.join(train_label_dir, '{:05}.png'.format(idx)), label)
        except Exception as e:
            print(e)
            continue

        pose_cords.append(head_cord)

    pose_cords_arr = np.array(pose_cords, dtype=np.int)
    np.save(os.path.join(save_dir, 'pose_source.npy'), pose_cords_arr)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare target Video')
    parser.add_argument('-s', '--save-dir', type=str,
                        default=os.path.join(dir_name, '../../data/targets/example_target'),
                        help='Path to the folder where the video is saved. One video per folder!')
    args = parser.parse_args()
    prepare_target(args.save_dir)
