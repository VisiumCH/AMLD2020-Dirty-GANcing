import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.backends import cudnn
from PIL import Image
image_transforms = transforms.Compose([
        Image.fromarray,
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
    ])

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


def load_models(directory, batch_num):
    # 20180924: smaller network.
    generator = model.GlobalGenerator(n_downsampling=2, n_blocks=6)
    discriminator = model.NLayerDiscriminator(input_nc=3, n_layers=3)  # 48 input
    gen_name = os.path.join(directory, '%05d_generator.pth' % batch_num)
    dis_name = os.path.join(directory, '%05d_discriminator.pth' % batch_num)

    if os.path.isfile(gen_name) and os.path.isfile(dis_name):
        gen_dict = torch.load(gen_name)
        dis_dict = torch.load(dis_name)
        generator.load_state_dict(gen_dict)
        discriminator.load_state_dict(dis_dict)
        print('Models loaded, resume training from batch %05d...' % batch_num)
    else:
        print('Cannot find saved models, start training from scratch...')
        batch_num = 0

    return generator, discriminator, batch_num


def train_face_enhancer(target_dir, run_name, is_debug):
    checkpoints_dir = os.path.join(dir_name, '../../checkpoints')
    dataset_dir = os.path.join(target_dir, 'face')
    pose_name = os.path.join(target_dir, 'pose_source.npy')
    ckpt_dir = os.path.join(checkpoints_dir, run_name, 'face')
    log_dir = os.path.join(ckpt_dir, 'logs')
    batch_num = 10
    batch_size = 10

    image_folder = dataset.ImageFolderDataset(dataset_dir, cache=os.path.join(dataset_dir, 'local.db'))
    face_dataset = dataset.FaceCropDataset(image_folder, pose_name, image_transforms, crop_size=48)  # 48 for 512-frame, 96 for HD frame
    data_loader = DataLoader(face_dataset, batch_size=batch_size,
                             drop_last=True, num_workers=4, shuffle=True)

    generator, discriminator, batch_num = load_models(ckpt_dir, batch_num)

    if is_debug:
        trainer = Trainer(ckpt_dir, log_dir, face_dataset, data_loader, log_every=1, save_every=1)
    else:
        trainer = Trainer(ckpt_dir, log_dir, face_dataset, data_loader)
    trainer.train(generator, discriminator, batch_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test pose transfer')
    parser.add_argument('-t', '--target-dir', type=str,
                        default=os.path.join(dir_name, '../../data/sources/example_target'),
                        help='Path to the folder where the target video is saved. One video per folder!')
    parser.add_argument('-r', '--run-name', type=str,
                        default='bruno_mars_example',
                        help='Path to the folder where the target video is saved. One video per folder!')
    parser.add_argument('-d', '--is-debug', default=False, action='store_true',
                        help='Debug mode')
    args = parser.parse_args()
    cudnn.enabled = True
    train_face_enhancer(args.target_dir, args.run_name, args.is_debug)
