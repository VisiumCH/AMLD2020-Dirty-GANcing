### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch


def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel, GANcing, InferenceModelGANcing
        if opt.isTrain:
            if opt.temporal_smoothing:
                model = GANcing()
            else:
                model = Pix2PixHDModel()
        else:
            if opt.temporal_smoothing:
                model = InferenceModelGANcing()
            else:
                model = InferenceModel()
    else:
        from .ui_model import UIModel
        model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    # if opt.isTrain and len(opt.gpu_ids):
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
