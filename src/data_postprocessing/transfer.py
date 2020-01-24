import argparse
import os
import sys
from collections import OrderedDict

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


def test_transfer(source_dir, run_name, temporal_smoothing=False, live_run_name=None):
    import src.config.test_opt as opt

    opt.name = run_name
    opt.dataroot = source_dir
    opt.temporal_smoothing = temporal_smoothing
    if device == torch.device('cpu'):
        opt.gpu_ids = []
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    opt.checkpoints_dir = os.path.join(dir_name, '../../checkpoints')
    opt.results_dir = os.path.join(dir_name, '../../results')

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    #print(opt.load_pretrain)
    model = create_model(opt)

    if live_run_name is not None:
        opt.name = live_run_name
    visualizer = Visualizer(opt)

    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    generated = None
    for data in tqdm(dataset):
        if temporal_smoothing:
            if generated is None:
                previous_frame = torch.zeros((1, 3, opt.loadSize, opt.loadSize))
                generated = model.inference(data['label'], data['inst'], previous_frame)
            else:
                generated = model.inference(data['label'], data['inst'], generated)

        else:
            generated = model.inference(data['label'], data['inst'])

        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                               ('synthesized_image', util.tensor2im(generated.data[0]))])
        img_path = data['path']
        visualizer.save_images(webpage, visuals, img_path)
    webpage.save()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test pose transfer')
    parser.add_argument('-s', '--source-dir', type=str,
                        default=os.path.join(dir_name, '../../data/sources/bruno_mars'),
                        help='Path to the folder where the target video is saved. One video per folder!')
    parser.add_argument('-r', '--run-name', type=str,
                        default='bruno_mars_example',
                        help='Path to the folder where the target video is saved. One video per folder!')
    parser.add_argument('-ts', '--temporal-smoothing', action='store_true',
                        help='Whether to use temporal smoothing or not')            
    args = parser.parse_args()
    test_transfer(args.source_dir, args.run_name, args.temporal_smoothing)
