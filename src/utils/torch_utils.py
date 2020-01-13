import torch
import os


def get_torch_device():
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)
        return torch.device('cuda')
    else:
        return torch.device('cpu')
