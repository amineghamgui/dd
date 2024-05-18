import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import os
import random
import time
import argparse
import numpy as np
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.cuda.amp as amp

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from config import build_dataset_config, build_model_config
from models.detector import build_model


GLOBAL_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description='YOWO')
    # CUDA
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')

    # Visualization
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='./weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='use tensorboard')

    # Mix precision training
    parser.add_argument('--fp16', dest="fp16", action="store_true", default=False,
                        help="Adopting mix precision training.")

    # Evaluation
    parser.add_argument('--eval', action='store_true', default=False, 
                        help='do evaluation during training.')
    parser.add_argument('--eval_epoch', default=1, type=int, 
                        help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--save_dir', default='inference_results/',
                        type=str, help='save inference results.')

    # Model
    parser.add_argument('-v', '--version', default='yowo', type=str, choices=['yowo', 'yowo_nano'],
                        help='build YOWO')
    parser.add_argument('--topk', default=40, type=int,
                        help='topk candidates for evaluation')
    parser.add_argument('-p', '--coco_pretrained', default=None, type=str,
                        help='coco pretrained weight')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--ema', dest="ema", action="store_true", default=False,
                        help="use model EMA.")

    # Dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, jhmdb')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    
    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)


import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import to_pil_image


import torchvision.transforms as T
import matplotlib.pyplot as plt

def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist
    print('World size: {}'.format(distributed_utils.get_world_size()))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # amp
    scaler = amp.GradScaler(enabled=args.fp16)

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(d_cfg, args, is_train=True)

    # dataloader
    batch_size = d_cfg['batch_size'] * distributed_utils.get_world_size()
    dataloader = build_dataloader(args, dataset, 1, CollateFunc(), is_train=True)


    # Assurez-vous que 'dataloader' est votre DataLoader
    pil_frames = []

    # Variable pour vérifier si le premier batch est traité
    first_batch_processed = False

    for iter_i, (frame_ids, video_clips, targets) in enumerate(dataloader):
        # Vérifiez si le premier batch est déjà traité
        if first_batch_processed:
            break

        # Permuter les dimensions pour obtenir les frames individuelles
        v = video_clips.permute(0, 2, 1, 3, 4)
        v = v[0]  # Sélectionner le premier batch
        
        # Transformer les frames en images PIL
        transform = T.ToPILImage()
        for frame_tensor in v:
            pil_frame = transform(frame_tensor)
            pil_frame = pil_frame.convert('RGB')
            pil_frames.append(pil_frame)

        # Marquer que le premier batch est traité
        first_batch_processed = True

    # Afficher les images du premier batch
    for i, pil_frame in enumerate(pil_frames):
        plt.figure()
        plt.imshow(pil_frame)
        plt.title(f'Frame {i}')
        plt.show()