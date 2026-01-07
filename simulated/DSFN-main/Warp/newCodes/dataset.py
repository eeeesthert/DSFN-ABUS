from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
import os
import glob
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path):
        self.width = 512
        self.height = 512
        self.train_path = data_path
        self.datas = OrderedDict()

        # 期望目录结构：
        #   data_path/
        #       input1/
        #       input2/
        #       depthInput1/
        #       depthInput2/
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = os.path.basename(data)
            if data_name in ("input1", "input2", "depthInput1", "depthInput2"):
                img_list = glob.glob(os.path.join(data, '*.jpg'))
                img_list.sort()
                self.datas[data_name] = {
                    "path": data,
                    "image": img_list,
                }

        print("TrainDataset folders:", self.datas.keys())

    def __len__(self):
        return len(self.datas["input1"]["image"])

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))  # [C, H, W]
        return img

    def _load_depth(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :]  # [1, H, W]
        return img

    def __getitem__(self, index):
        # 彩色输入
        input1 = self._load_rgb(self.datas["input1"]["image"][index])
        input2 = self._load_rgb(self.datas["input2"]["image"][index])

        # 深度 / mask 先验
        depth1 = self._load_depth(self.datas["depthInput1"]["image"][index])
        depth2 = self._load_depth(self.datas["depthInput2"]["image"][index])

        # 转 tensor
        input1_tensor = torch.from_numpy(input1)
        input2_tensor = torch.from_numpy(input2)
        depth1_tensor = torch.from_numpy(depth1)
        depth2_tensor = torch.from_numpy(depth2)

        # 随机交换左右
        if random.randint(0, 1) == 0:
            return input1_tensor, input2_tensor, depth1_tensor, depth2_tensor
        else:
            return input2_tensor, input1_tensor, depth2_tensor, depth1_tensor


class TestDataset(Dataset):
    def __init__(self, data_path):
        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = os.path.basename(data)
            if data_name in ("input1", "input2"):
                img_list = glob.glob(os.path.join(data, '*.jpg'))
                img_list.sort()
                self.datas[data_name] = {
                    "path": data,
                    "image": img_list,
                }

        print("TestDataset folders:", self.datas.keys())

    def __len__(self):
        return len(self.datas["input1"]["image"])

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def __getitem__(self, index):
        input1 = self._load_rgb(self.datas["input1"]["image"][index])
        input2 = self._load_rgb(self.datas["input2"]["image"][index])

        input1_tensor = torch.from_numpy(input1)
        input2_tensor = torch.from_numpy(input2)

        return input1_tensor, input2_tensor


class TestDepthOutDataset(Dataset):
    def __init__(self, data_path):
        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = os.path.basename(data)
            if data_name in ("input1", "input2", "depthInput1", "depthInput2"):
                img_list = glob.glob(os.path.join(data, '*.jpg'))
                img_list.sort()
                self.datas[data_name] = {
                    "path": data,
                    "image": img_list,
                }

        print("TestDepthOutDataset folders:", self.datas.keys())

    def __len__(self):
        return len(self.datas["input1"]["image"])

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def _load_depth(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :]
        return img

    def __getitem__(self, index):
        input1 = self._load_rgb(self.datas["input1"]["image"][index])
        input2 = self._load_rgb(self.datas["input2"]["image"][index])
        depth1 = self._load_depth(self.datas["depthInput1"]["image"][index])
        depth2 = self._load_depth(self.datas["depthInput2"]["image"][index])

        input1_tensor = torch.from_numpy(input1)
        input2_tensor = torch.from_numpy(input2)
        depth1_tensor = torch.from_numpy(depth1)
        depth2_tensor = torch.from_numpy(depth2)

        return input1_tensor, input2_tensor, depth1_tensor, depth2_tensor
