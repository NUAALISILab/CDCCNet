from argparse import ArgumentParser
from tkinter import N
import torch
import numpy as np
import random
from models.trainer import *
import os

"""
the main function for training the CD networks
"""


def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu


def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models_dp(args.pretrain)


def multi_train(args):
    dataloader = utils.get_multi_loader(args)
    model = CDTrainer(args=args, dataloaders={'train': dataloader})
    model.train_models_dp(args.pretrain)


def test(args):
    from models.evaluator import CDEvaluator
    test_data_names = args.test_data_names.split(',')
    for name in test_data_names:
        name = name.strip()
        dataloader = utils.get_loader(name, img_size=args.img_size,
                                      batch_size=1, is_train=False,
                                      split='test')
        print(f"Evaluating on dataset: {name}")
        model = CDEvaluator(args=args, dataloader=dataloader, dataset_name=name)

        model.eval_models_dp()


if __name__ == '__main__':
    seed_torch(42)
    # ------------
    # args
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name',
                        default='...', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # data
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='', type=str)
    parser.add_argument('--train_data_names', type=str, default='', help='Syntheworld,GZ,WHU,EGY_BCD,LEVIR,LEVIR+')
    parser.add_argument('--test_data_names', type=str, default='', help='Syntheworld,GZ,WHU,EGY_BCD,LEVIR,LEVIR+')

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--pixel_edge', default=0, type=int)
    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--pretrain', default='', type=str, help='resnet50-19c8e357.pth')
    parser.add_argument('--vis_act', action='store_true')
    parser.add_argument('--net_G', default='CDCCNet', type=str, help='CDCCNet' )
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)
    parser.add_argument('--min_lr', default=1e-6, type=int)
    

    args = parser.parse_args()
    utils.get_device(args)
    print("Current GPU:", args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)
    # multi_train(args)
    test(args)

