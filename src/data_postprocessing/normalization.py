import argparse
import os
import cv2

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

dir_name = os.path.dirname(__file__)


def normalize(source_dir, target_dir):

    target_img = cv2.imread(os.path.join(target_dir, 'train/train_label/00010.png'))[:, :, 0]
    target_img_rgb = cv2.imread(os.path.join(target_dir, 'train/train_img/00010.png'))

    # Original code seems wrong:
    # source_img = cv2.imread(os.path.join(target_dir, 'train/train_label/00001.png'))[:, :, 0]
    # source_img_rgb = cv2.imread(os.path.join(target_dir, 'train/train_img/00001.png'))

    source_img = cv2.imread(os.path.join(source_dir, 'test_label_ori/00010.png'))[:, :, 0]
    source_img_rgb = cv2.imread(os.path.join(source_dir, 'test_img/00010.png'))

    path = os.path.join(source_dir, 'test_label_ori')
    output = os.path.join(source_dir, 'test_label')
    os.makedirs(output, exist_ok=True)
    head_dir = os.path.join(source_dir, 'test_head')
    os.makedirs(head_dir, exist_ok=True)
    pose_dir = os.path.join(source_dir, 'pose_source.npy')
    pose_cord = np.load(pose_dir)

    plt.subplot(222)
    plt.imshow(target_img)
    plt.subplot(221)
    plt.imshow(target_img_rgb)
    plt.subplot(224)
    plt.imshow(source_img)
    plt.subplot(223)
    plt.imshow(source_img_rgb)
    plt.savefig('norm.png')

    target_head, target_height = get_scale(target_img)
    target_head_x = target_head[0]
    target_head_y = target_head[1]

    source_head, source_height = get_scale(source_img)

    new_head_pose = []

    for img_idx in tqdm(range(len(os.listdir(path)))):
        img = cv2.imread(os.path.join(path, '{:05}.png'.format(img_idx)))
        source_rsize = cv2.resize(img,
                                  (int(img.shape[0] * target_height / source_height),
                                   int(img.shape[1] * target_height / source_height)))

        source_pad = np.pad(source_rsize, ((1000, 1000), (1000, 1000), (0, 0)),
                            mode='edge')

        source_head_rs, source_height_rs = get_scale(source_pad[:, :, 0])
        source_head_rs_x = source_head_rs[0]
        source_head_rs_y = source_head_rs[1]

        new_source = source_pad[
                    (source_head_rs_x - target_head_x):(source_head_rs_x + (target_img.shape[0] - target_head_x)),
                    int((source_pad.shape[1] - target_img.shape[1])/2):int((source_pad.shape[1] - (source_pad.shape[1] - target_img.shape[1])/2))
                    ]
        new_source_head, _ = get_scale(new_source[:, :, 0])

        source_head_x, source_head_y = source_head
        source_cord_y, source_cord_x = pose_cord[0]

        new_head_y = int(new_source_head[1] - (source_head_y - source_cord_y))
        new_head_x = int(new_source_head[0] - (source_head_x - source_cord_x) * (target_height / source_height))

        crop_size = 50
        new_head_pose.append([new_head_y, new_head_x])
        head = img[int(new_head_x - crop_size): int(new_head_x + crop_size),
                   int(new_head_y - crop_size): int(new_head_y + crop_size), :]
        plt.imshow(head)
        plt.savefig(os.path.join(head_dir, 'pose_{}.jpg'.format(img_idx)))
        plt.clf()
        cv2.imwrite(os.path.join(output, '{:05}.png'.format(img_idx)), new_source)

    pose_cords_arr = np.array(new_head_pose, dtype=np.int)
    np.save(os.path.join(source_dir, 'pose_source_norm.npy'), pose_cords_arr)


def get_scale(label_img):
    any1 = label_img.any(axis=1)
    linspace1 = np.arange(len(any1))
    head_x, height = linspace1[list(any1)][0], len(linspace1[list(any1)])
    any0 = label_img[head_x, :] != 0
    linspace2 = np.arange(len(any0))
    head_y = int(np.mean(linspace2[list(any0)]))
    return (head_x, head_y), height


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare target Video')
    parser.add_argument('-t', '--target-dir', type=str,
                        default=os.path.join(dir_name, '../../data/targets/example_target'),
                        help='Path to the folder where the target video is saved. One video per folder!')
    parser.add_argument('-s', '--source-dir', type=str,
                        default=os.path.join(dir_name, '../../data/sources/bruno_mars'),
                        help='Path to the folder where the target video is saved. One video per folder!')
    args = parser.parse_args()
    normalize(args.source_dir, args.target_dir)
