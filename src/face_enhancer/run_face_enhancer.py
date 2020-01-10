import argparse
import cv2
import os
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage.io import imsave
from imageio import get_writer

dir_name = os.path.dirname(__file__)
sys.path.append(dir_name)
sys.path.append(os.path.join(dir_name, '../utils'))

import model
import dataset
from trainer import Trainer
from torch_utils import get_torch_device

device = get_torch_device()
if device != torch.device('cpu'):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

image_transforms = transforms.Compose([
        Image.fromarray,
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
    ])


def load_models(directory):
    generator = model.GlobalGenerator(n_downsampling=2, n_blocks=6)
    gen_name = os.path.join(directory, '16000_generator.pth')

    if os.path.isfile(gen_name):
        gen_dict = torch.load(gen_name)
        generator.load_state_dict(gen_dict)

    return generator.to(device)


def torch2numpy(tensor):
    generated = tensor.detach().cpu().permute(1, 2, 0).numpy()
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype(np.uint8)


def test_face_enhancer(target_dir, source_dir, run_name):
    torch.backends.cudnn.benchmark = True
    checkpoints_dir = os.path.join(dir_name, '../../checkpoints')
    dataset_dir = os.path.join(target_dir, 'face_transfer')   # save test_sync in this folder
    pose_name = os.path.join(source_dir, 'pose_source_norm.npy') # coordinate save every heads
    ckpt_dir = os.path.join(checkpoints_dir, run_name, 'face')
    result_dir = os.path.join(dir_name, '../../results')
    save_dir = os.path.join(result_dir, run_name + '_enhanced', 'test_latest/images')
    os.makedirs(save_dir, exist_ok=True)

    image_folder = dataset.ImageFolderDataset(dataset_dir, cache=os.path.join(dataset_dir, 'local.db'), is_test=True)
    face_dataset = dataset.FaceCropDataset(image_folder, pose_name, image_transforms, crop_size=48)
    length = len(face_dataset)
    print('Picture number', length)

    generator = load_models(os.path.join(ckpt_dir))

    for i in tqdm(range(length)):
        _, fake_head, top, bottom, left, right, real_full, fake_full \
            = face_dataset.get_full_sample(i)

        with torch.no_grad():
            fake_head.unsqueeze_(0)
            fake_head = fake_head.to(device)
            residual = generator(fake_head)
            enhanced = fake_head + residual

        enhanced.squeeze_()
        enhanced = torch2numpy(enhanced)
        fake_full_old = fake_full.copy()
        fake_full[top: bottom, left: right, :] = enhanced

        b, g, r = cv2.split(fake_full)
        fake_full = cv2.merge([r, g, b])
        cv2.imwrite(os.path.join(save_dir, '{:05}_synthesized_image.png'.format(i)), fake_full)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test face enhancer')
    parser.add_argument('-t', '--target-dir', type=str,
                        default=os.path.join(dir_name, '../../data/targets/example_target'),
                        help='Path to the folder where the target video is saved. One video per folder!')
    parser.add_argument('-s', '--source-dir', type=str,
                        default=os.path.join(dir_name, '../../data/sources/bruno_mars'),
                        help='Path to the folder where the source video is saved. One video per folder!')
    parser.add_argument('-r', '--run-name', type=str,
                        default='bruno_mars_example',
                        help='Name of the run')
    args = parser.parse_args()
    test_face_enhancer(args.target_dir, args.source_dir, args.run_name)
