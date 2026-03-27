from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
import os
import glob
from collections import OrderedDict


def _read_coord_file(path):
    """Return (x, y, valid). txt/csv: 'x y [valid]' or 'x,y[,valid]'"""
    if not os.path.exists(path):
        return 0.0, 0.0, 0.0
    raw = open(path, 'r', encoding='utf-8').read().strip()
    if not raw:
        return 0.0, 0.0, 0.0
    parts = raw.replace(',', ' ').split()
    if len(parts) < 2:
        return 0.0, 0.0, 0.0
    x = float(parts[0])
    y = float(parts[1])
    valid = float(parts[2]) if len(parts) >= 3 else float(x >= 0 and y >= 0)
    return x, y, valid


def _gaussian_heatmap_from_coord(x, y, h=512, w=512, sigma=6.0):
    if x < 0 or y < 0:
        return np.zeros((1, h, w), dtype=np.float32), 0.0
    xx = np.arange(w, dtype=np.float32)[None, :]
    yy = np.arange(h, dtype=np.float32)[:, None]
    heat = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)).astype(np.float32)
    return heat[None, ...], 1.0


class TrainDataset(Dataset):
    def __init__(self, data_path):
        self.width = 512
        self.height = 512
        self.train_path = data_path
        self.datas = OrderedDict()

        valid_folders = (
            "input1", "input2", "input3",
            "nippleHeatmap1", "nippleHeatmap2", "nippleHeatmap3",
            "nippleCoord1", "nippleCoord2", "nippleCoord3",
        )

        for data in sorted(glob.glob(os.path.join(self.train_path, '*'))):
            data_name = os.path.basename(data)
            if data_name in valid_folders:
                items = sorted(glob.glob(os.path.join(data, '*.jpg')))
                if len(items) == 0:
                    items = sorted(glob.glob(os.path.join(data, '*.png')))
                if len(items) == 0:
                    items = sorted(glob.glob(os.path.join(data, '*.txt')))
                if len(items) == 0:
                    items = sorted(glob.glob(os.path.join(data, '*.csv')))
                self.datas[data_name] = {"path": data, "image": items}

        print("TrainDataset folders:", self.datas.keys())

    def __len__(self):
        return len(self.datas["input1"]["image"])

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        return np.transpose(img, (2, 0, 1))

    def _load_heatmap(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.width, self.height))
        return (img.astype(np.float32) / 255.0)[np.newaxis, :, :], 1.0

    def _load_prior(self, index, heat_folder, coord_folder):
        if heat_folder in self.datas:
            return self._load_heatmap(self.datas[heat_folder]["image"][index])

        if coord_folder in self.datas:
            coord_path = self.datas[coord_folder]["image"][index]
            x, y, valid = _read_coord_file(coord_path)
            heat, auto_valid = _gaussian_heatmap_from_coord(x, y, self.height, self.width)
            return heat, float(valid) * auto_valid

        return np.zeros((1, self.height, self.width), dtype=np.float32), 0.0

    def __getitem__(self, index):
        input1 = self._load_rgb(self.datas["input1"]["image"][index])
        input2 = self._load_rgb(self.datas["input2"]["image"][index])
        input3 = self._load_rgb(self.datas["input3"]["image"][index])

        heat1, valid1 = self._load_prior(index, "nippleHeatmap1", "nippleCoord1")
        heat2, valid2 = self._load_prior(index, "nippleHeatmap2", "nippleCoord2")
        heat3, valid3 = self._load_prior(index, "nippleHeatmap3", "nippleCoord3")

        return (
            torch.from_numpy(input1),
            torch.from_numpy(input2),
            torch.from_numpy(input3),
            torch.from_numpy(heat1),
            torch.from_numpy(heat2),
            torch.from_numpy(heat3),
            torch.tensor([valid1], dtype=torch.float32),
            torch.tensor([valid2], dtype=torch.float32),
            torch.tensor([valid3], dtype=torch.float32),
        )


class TestDataset(Dataset):
    def __init__(self, data_path):
        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.datas = OrderedDict()

        for data in sorted(glob.glob(os.path.join(self.test_path, '*'))):
            data_name = os.path.basename(data)
            if data_name in ("input1", "input2", "input3"):
                img_list = sorted(glob.glob(os.path.join(data, '*.jpg')))
                if len(img_list) == 0:
                    img_list = sorted(glob.glob(os.path.join(data, '*.png')))
                self.datas[data_name] = {"path": data, "image": img_list}

        print("TestDataset folders:", self.datas.keys())

    def __len__(self):
        return len(self.datas["input1"]["image"])

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, index):
        input1 = self._load_rgb(self.datas["input1"]["image"][index])
        input2 = self._load_rgb(self.datas["input2"]["image"][index])
        input3 = self._load_rgb(self.datas["input3"]["image"][index])
        return torch.from_numpy(input1), torch.from_numpy(input2), torch.from_numpy(input3)


class TestDepthOutDataset(Dataset):
    def __init__(self, data_path):
        self.width = 512
        self.height = 512
        self.test_path = data_path
        self.datas = OrderedDict()

        valid = ("input1", "input2", "nippleHeatmap1", "nippleHeatmap2", "nippleCoord1", "nippleCoord2")
        for data in sorted(glob.glob(os.path.join(self.test_path, '*'))):
            data_name = os.path.basename(data)
            if data_name in valid:
                items = sorted(glob.glob(os.path.join(data, '*.jpg')))
                if len(items) == 0:
                    items = sorted(glob.glob(os.path.join(data, '*.png')))
                if len(items) == 0:
                    items = sorted(glob.glob(os.path.join(data, '*.txt')))
                if len(items) == 0:
                    items = sorted(glob.glob(os.path.join(data, '*.csv')))
                self.datas[data_name] = {"path": data, "image": items}

    def __len__(self):
        return len(self.datas["input1"]["image"])

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, index):
        input1 = self._load_rgb(self.datas["input1"]["image"][index])
        input2 = self._load_rgb(self.datas["input2"]["image"][index])

        if "nippleHeatmap1" in self.datas:
            heat1 = cv2.imread(self.datas["nippleHeatmap1"]["image"][index], cv2.IMREAD_GRAYSCALE)
            heat1 = cv2.resize(heat1, (self.width, self.height)).astype(np.float32) / 255.0
            heat1 = heat1[np.newaxis, :, :]
        else:
            x1, y1, v1 = _read_coord_file(self.datas["nippleCoord1"]["image"][index])
            heat1, _ = _gaussian_heatmap_from_coord(x1, y1, self.height, self.width)

        if "nippleHeatmap2" in self.datas:
            heat2 = cv2.imread(self.datas["nippleHeatmap2"]["image"][index], cv2.IMREAD_GRAYSCALE)
            heat2 = cv2.resize(heat2, (self.width, self.height)).astype(np.float32) / 255.0
            heat2 = heat2[np.newaxis, :, :]
        else:
            x2, y2, v2 = _read_coord_file(self.datas["nippleCoord2"]["image"][index])
            heat2, _ = _gaussian_heatmap_from_coord(x2, y2, self.height, self.width)

        return torch.from_numpy(input1), torch.from_numpy(input2), torch.from_numpy(heat1), torch.from_numpy(heat2)
