import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import utils

import data_config
from datasets.CD_dataset import CDDataset


def get_loader(data_name, img_size=256, batch_size=4, split='test',
               is_train=False, dataset='CDDataset'):
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=img_size, is_train=is_train,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)

    return dataloader


def get_loaders(args):

    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    image_type = dataConfig.image_type
    split = args.split


    if args.dataset == 'CDDataset':
        training_set = CDDataset(root_dir=root_dir, img_size=args.img_size,
                                 split=split,is_train=True,
                                 label_transform=label_transform,image_type=image_type
                                 )

    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train']}

    return dataloaders


class CombinedDataset(Dataset):
    def __init__(self, datasets, names):
        self.datasets = datasets
        self.names = names
        self.cum_lengths = np.cumsum([len(d) for d in datasets])

    def __len__(self):
        return int(self.cum_lengths[-1])

    def __getitem__(self, idx):
        ds_idx = np.searchsorted(self.cum_lengths, idx, side='right')
        prev_cum = 0 if ds_idx == 0 else self.cum_lengths[ds_idx - 1]
        sample = self.datasets[ds_idx][idx - prev_cum]
        sample['dataset'] = self.names[ds_idx]
        return sample

def get_multi_loader(args):
    names = [n.strip() for n in args.train_data_names.split(',')]
    datasets_list = []
    for name in names:
        cfg = data_config.DataConfig().get_data_config(name)
        ds = CDDataset(
            root_dir=cfg.root_dir,
            img_size=args.img_size,
            split=args.split,
            is_train=True,
            label_transform=cfg.label_transform,
            image_type=getattr(cfg, 'image_type', 'png')
        )
        datasets_list.append(ds)
    combined = CombinedDataset(datasets_list, names)
    loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    return loader


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
