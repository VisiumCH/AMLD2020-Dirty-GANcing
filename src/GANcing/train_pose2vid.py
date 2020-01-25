import argparse
import json
import os
import numpy as np
import torch
import time
import sys
from collections import OrderedDict
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')

dir_name = os.path.dirname(__file__)
pix2pixhd_dir = os.path.join(dir_name, '../pix2pixHD/')
sys.path.append(pix2pixhd_dir)
sys.path.append(os.path.join(dir_name, '../..'))
sys.path.append(os.path.join(dir_name, '../utils'))


from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from torch_utils import get_torch_device

device = get_torch_device()


def train_pose2vid(target_dir, run_name, temporal_smoothing=False):
    import src.config.train_opt as opt

    opt = update_opt(opt, target_dir, run_name, temporal_smoothing)

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.json')
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    if opt.load_pretrain != '':
        with open(iter_path, 'r') as f:
            iter_json = json.load(f)
    else:
        iter_json = {'start_epoch': 1, 'epoch_iter': 0}

    start_epoch = iter_json['start_epoch']
    epoch_iter = iter_json['epoch_iter']
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    model = create_model(opt)
    model = model.to(device)
    visualizer = Visualizer(opt)

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            ############## Forward Pass ######################
            if temporal_smoothing:
                losses, generated = model(Variable(data['label']), Variable(data['inst']),
                                          Variable(data['image']), Variable(data['feat']),
                                          Variable(data['previous_label']), Variable(data['previous_image']), infer=save_fake)
            else:
                losses, generated = model(Variable(data['label']), Variable(data['inst']),
                                        Variable(data['image']), Variable(data['feat']), infer=save_fake)

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

            ############### Backward Pass ####################
            # update generator weights
            model.optimizer_G.zero_grad()
            loss_G.backward()
            model.optimizer_G.step()

            # update discriminator weights
            model.optimizer_D.zero_grad()
            loss_D.backward()
            model.optimizer_D.step()


            ############## Display results and errors ##########

            print(f"Epoch {epoch} batch {i}:")
            print(f"loss_D: {loss_D}, loss_G: {loss_G}")
            print(f"loss_D_fake: {loss_dict['D_fake']}, loss_D_real: {loss_dict['D_real']}")
            print(f"loss_G_GAN {loss_dict['G_GAN']}, loss_G_GAN_Feat: {loss_dict.get('G_GAN_Feat', 0)}, loss_G_VGG: {loss_dict.get('G_VGG', 0)}\n")

            ### print out errors
            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                # errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')
                iter_json['start_epoch'] = epoch
                iter_json['epoch_iter'] = epoch_iter
                with open(iter_path, 'w') as f:
                    json.dump(iter_json, f)

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
            iter_json['start_epoch'] = epoch + 1
            iter_json['epoch_iter'] = 0
            with open(iter_path, 'w') as f:
                json.dump(iter_json, f)

        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.update_learning_rate()

    torch.cuda.empty_cache()


def update_opt(opt, target_dir, run_name, temporal_smoothing):
    opt.dataroot = os.path.join(target_dir, 'train')
    opt.name = run_name
    if os.path.isdir(os.path.join(dir_name, "../../checkpoints", run_name)):
        print("Run already exists, will try to resume training")
        opt.load_pretrain = os.path.join(dir_name, "../../checkpoints", run_name)

    if device == torch.device('cpu'):
        opt.gpu_ids = []

    opt.temporal_smoothing = temporal_smoothing

    return opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare target Video')
    parser.add_argument('-t', '--target-dir', type=str,
                        default=os.path.join(dir_name, '../../data/targets/example_target'),
                        help='Path to the folder where the target video is saved. One video per folder!')
    parser.add_argument('-r', '--run-name', type=str,
                        default='bruno_mars_example',
                        help='Name of the current run')
    parser.add_argument('-ts', '--temporal-smoothing', action='store_true',
                        help='Whether to use temporal smoothing or not')            
    args = parser.parse_args()
    train_pose2vid(args.target_dir, args.run_name, args.temporal_smoothing)
