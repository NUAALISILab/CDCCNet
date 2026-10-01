import numpy as np
import os
import random
import glob
'''
    定义了一些常用的工具函数，用于设置随机种子、创建目录、以及从文件夹中获取文件路径
'''

def seed_random(seed=2020):
    # 加入以下随机种子，数据输入，随机扩充等保持一致
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_paths(image_folder_path, suffix='*.png'):
    """从文件夹中返回指定格式的文件
    :param image_folder_path: str
    :param suffix: str
    :return: list
    """
    paths = sorted(glob.glob(os.path.join(image_folder_path, suffix)))
    return paths


def get_paths_from_list(image_folder_path, list):
    """从image folder中找到list中的文件，返回path list"""
    out = []
    for item in list:
        path = os.path.join(image_folder_path,item)
        out.append(path)
    return sorted(out)


