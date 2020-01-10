import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

dir_name = os.path.dirname(__file__)

pix2pixhd_dir = os.path.join(dir_name, '../pix2pixHD/')
sys.path.append(pix2pixhd_dir)
sys.path.append(os.path.join(dir_name, '../..'))
sys.path.append(os.path.join(dir_name, '../utils'))

import util.util as util
from data.data_loader import CreateDataLoader
from models.models import create_model
from util import html
from util.visualizer import Visualizer
from torch_utils import get_torch_device

device = get_torch_device()


def prepare_face_enhancer_data(target_dir, run_name):

    face_sync_dir = os.path.join(target_dir, 'face')
    os.makedirs(face_sync_dir, exist_ok=True)
    test_sync_dir = os.path.join(face_sync_dir, 'test_sync')
    os.makedirs(test_sync_dir, exist_ok=True)
    test_real_dir = os.path.join(face_sync_dir, 'test_real')
    os.makedirs(test_real_dir, exist_ok=True)
    test_img = os.path.join(target_dir, 'test_img')
    os.makedirs(test_img, exist_ok=True)
    test_label = os.path.join(target_dir, 'test_label')
    os.makedirs(test_label, exist_ok=True)

    transfer_face_sync_dir = os.path.join(target_dir, 'face_transfer')
    os.makedirs(transfer_face_sync_dir, exist_ok=True)
    transfer_test_sync_dir = os.path.join(transfer_face_sync_dir, 'test_sync')
    os.makedirs(transfer_test_sync_dir, exist_ok=True)
    transfer_test_real_dir = os.path.join(transfer_face_sync_dir, 'test_real')
    os.makedirs(transfer_test_real_dir, exist_ok=True)

    train_dir = os.path.join(target_dir, 'train', 'train_img')
    label_dir = os.path.join(target_dir, 'train', 'train_label')

    print('Prepare test_real....')
    for img_file in tqdm(sorted(os.listdir(train_dir))):
        img_idx = int(img_file.split('.')[0])
        img = cv2.imread(os.path.join(train_dir, '{:05}.png'.format(img_idx)))
        label = cv2.imread(os.path.join(label_dir, '{:05}.png'.format(img_idx)))
        cv2.imwrite(os.path.join(test_real_dir, '{:05}.png'.format(img_idx)), img)
        cv2.imwrite(os.path.join(transfer_test_real_dir, '{:05}.png'.format(img_idx)), img)
        cv2.imwrite(os.path.join(test_img, '{:05}.png'.format(img_idx)), img)
        cv2.imwrite(os.path.join(test_label, '{:05}.png'.format(img_idx)), label)

    print('Prepare test_sync....')

    import src.config.test_opt as opt
    if device == torch.device('cpu'):
        opt.gpu_ids = []
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    opt.checkpoints_dir = os.path.join(dir_name, '../../checkpoints/')
    opt.dataroot = target_dir
    opt.name = run_name
    opt.nThreads = 0
    opt.results_dir = os.path.join(dir_name, '../../face_enhancer_results/')

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)

    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    model = create_model(opt)

    for data in tqdm(dataset):
        minibatch = 1
        generated = model.inference(data['label'], data['inst'])

        visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.data[0]))])
        img_path = data['path']
        visualizer.save_images(webpage, visuals, img_path)
    webpage.save()
    torch.cuda.empty_cache()

    print(f'Copy the synthesized images in {test_sync_dir}...')
    synthesized_image_dir = os.path.join(dir_name, '../../face_enhancer_results', run_name, 'test_latest/images/')
    img_list = [f for f in os.listdir(synthesized_image_dir) if f.endswith('synthesized_image.jpg')]
    for img_file in tqdm(sorted(img_list)):
        img_idx = int(img_file.split('_')[0])
        img = cv2.imread(os.path.join(synthesized_image_dir, '{:05}_synthesized_image.jpg'.format(img_idx)))
        cv2.imwrite(os.path.join(test_sync_dir, '{:05}.png'.format(img_idx)), img)

    print('Copy transfer_test_sync')
    previous_run_img_dir = os.path.join(dir_name, '../../results', run_name, 'test_latest/images/')
    img_list = [f for f in os.listdir(previous_run_img_dir) if f.endswith('synthesized_image.jpg')]
    for img_file in tqdm(sorted(img_list)):
        img_idx = int(img_file.split('_')[0])
        img = cv2.imread(os.path.join(previous_run_img_dir, '{:05}_synthesized_image.jpg'.format(img_idx)))
        cv2.imwrite(os.path.join(transfer_test_sync_dir, '{:05}.png'.format(img_idx)), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test pose transfer')
    parser.add_argument('-t', '--target-dir', type=str,
                        default=os.path.join(dir_name, '../../data/sources/example_target'),
                        help='Path to the folder where the target video is saved. One video per folder!')
    parser.add_argument('-r', '--run-name', type=str,
                        default='bruno_mars_example',
                        help='Path to the folder where the target video is saved. One video per folder!')
    args = parser.parse_args()
    prepare_face_enhancer_data(args.target_dir, args.run_name)
