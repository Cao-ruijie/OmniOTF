import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
import numpy as np

class ReconsDataset2D(torch.utils.data.Dataset):
    def __init__(self, train_in_path, train_up_path, train_gt_path, img_type):
        self.train_in_path = train_in_path
        self.train_gt_path = train_gt_path
        self.train_up_path = train_up_path
        self.img_type = img_type
        self.dirs_gt = os.listdir(self.train_gt_path)

    def __len__(self):
        dirs = os.listdir(self.train_gt_path)  # open the files
        return len(dirs)  # because one of the file is for groundtruth

    def __getitem__(self, idx):
        image_name = os.path.join(self.train_gt_path, self.dirs_gt[idx])
        data_gt = io.imread(image_name)
        data_gt = data_gt / data_gt.max()
        data_gt = np.expand_dims(data_gt, 0)

        image_name = os.path.join(self.train_in_path, self.dirs_gt[idx])
        data_in = io.imread(image_name)
        data_in = data_in / data_in.max()
        data_in = np.expand_dims(data_in, 0)

        image_name = os.path.join(self.train_up_path, self.dirs_gt[idx])
        data_up = io.imread(image_name)
        data_up = data_up / data_in.max()
        data_up = np.expand_dims(data_up, 0)

        sample = {'image_in': data_in, 'image_gt': data_gt, 'image_up': data_up}
        return sample


class ReconsDataset2D_single(torch.utils.data.Dataset):
    def __init__(self, train_gt_path, train_ref_path, img_type):
        self.train_gt_path = train_gt_path
        self.train_ref_path = train_ref_path
        self.img_type = img_type
        self.dirs_gt = os.listdir(self.train_gt_path)

    def __len__(self):
        dirs = os.listdir(self.train_gt_path)  # open the files
        return len(dirs)  # because one of the file is for groundtruth

    def __getitem__(self, idx):
        image_name = os.path.join(self.train_gt_path, self.dirs_gt[idx])
        data_gt = io.imread(image_name)
        data_gt = data_gt / data_gt.max()
        data_gt = np.expand_dims(data_gt, 0)

        image_name = os.path.join(self.train_ref_path, self.dirs_gt[idx])
        data_ref = io.imread(image_name)
        data_ref = data_ref / data_ref.max()
        data_ref = np.expand_dims(data_ref, 0)
        sample = {'image_gt': data_gt, 'image_ref': data_ref}
        return sample

# ===== 改进版非配对数据集加载 =====
class UnpairedSRDataset(Dataset):
    def __init__(self, path_A, path_B, img_size=128):
        """
        Args:
            path_A (str): 低分辨率图像目录
            path_B (str): 高分辨率图像目录
            img_size (int): 统一输出尺寸
        """
        # 获取所有文件路径（网页7方法）
        self.paths_A = [os.path.join(path_A, f)
                        for f in os.listdir(path_A)
                        if f.endswith(('.tif'))]
        self.paths_B = [os.path.join(path_B, f)
                        for f in os.listdir(path_B)
                        if f.endswith(('.tif'))]

    def __len__(self):
        return max(len(self.paths_A), len(self.paths_B))  # 取最大长度

    def __getitem__(self, idx):
        # 随机索引生成（网页3的采样思路）
        idx_A = torch.randint(0, len(self.paths_A), (1,)).item()
        idx_B = torch.randint(0, len(self.paths_B), (1,)).item()

        # 加载并预处理图像（网页7的异常处理）
        try:
            img_A = io.imread(self.paths_A[idx_A])
            img_A = img_A / img_A.max()
            img_A = np.expand_dims(img_A, 0)
            img_B = io.imread(self.paths_B[idx_B])
            img_B = img_B / img_B.max()
            img_B = np.expand_dims(img_B, 0)
        except:
            return self[(idx + 1) % len(self)]  # 遇到损坏文件跳过

        return {'A': img_A,
                'B': img_B}


# ===== 创建DataLoader =====
def create_paired_loaders(path_A, path_B, batch_size=8):
    dataset = UnpairedSRDataset(path_A, path_B)

    # 多线程加速加载（网页2建议）
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,  # 启用随机打乱（网页3）
                        num_workers=0,
                        pin_memory=True,
                        drop_last=True)  # 保证批次完整
    return loader