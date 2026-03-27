# from torch.utils.data import Dataset
# import numpy as np
# import cv2
# import torch
# import os
# import glob
# from collections import OrderedDict
# import random


# class TrainDataset(Dataset):
#     def __init__(self, data_path):

        
#         self.width = 512
#         self.height = 512

#         self.train_path = data_path
#         self.datas = OrderedDict()

#         #   train_path/
#         #       warp1/
#         #       warp2/
#         #       mask1/
#         #       mask2/
#         #       warp_depth1/
#         #       warp_depth2/
#         datas = glob.glob(os.path.join(self.train_path, '*'))
#         for data in sorted(datas):
#             data_name = os.path.basename(data)  # 
#             if data_name in ['warp1', 'warp2', 'mask1', 'mask2', 'warp_depth1', 'warp_depth2']:
#                 img_list = glob.glob(os.path.join(data, '*.jpg'))
#                 img_list.sort()
#                 self.datas[data_name] = {
#                     'path': data,
#                     'image': img_list,
#                 }
#         print("TrainDataset folders:", self.datas.keys())
#         print("num images:", {k: len(v['image']) for k, v in self.datas.items()})

#     def __len__(self):
#         return len(self.datas['warp1']['image'])

#     # --------- resize ---------

#     def _load_rgb(self, path):
#         img = cv2.imread(path, cv2.IMREAD_COLOR)  # H, W, 3
        
#         img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
#         img = img.astype(np.float32)
#         img = img / 127.5 - 1.0                 # [-1,1]，和 Warp 阶段一致
#         img = np.transpose(img, (2, 0, 1))      # C, H, W
#         return torch.from_numpy(img)

#     def _load_mask_or_depth(self, path):
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # H, W
#         img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
#         img = img.astype(np.float32)
#         img = img / 255.0                        # [0,1]
#         img = img[np.newaxis, :, :]              # 1, H, W
#         return torch.from_numpy(img)

#     # --------- 取一个样本 ---------

#     def __getitem__(self, index):

#         # load warp1/warp2
#         warp1 = self._load_rgb(self.datas['warp1']['image'][index])
#         warp2 = self._load_rgb(self.datas['warp2']['image'][index])

#         # load mask1/mask2
#         mask1 = self._load_mask_or_depth(self.datas['mask1']['image'][index])
#         mask2 = self._load_mask_or_depth(self.datas['mask2']['image'][index])

#         # depthinput1/2：这里其实是你 warp 后的结节 mask 先验
#         depthinput1 = self._load_mask_or_depth(self.datas['warp_depth1']['image'][index])
#         depthinput2 = self._load_mask_or_depth(self.datas['warp_depth2']['image'][index])

#         # 随机交换左右，保持和 Warp 阶段风格一致
#         if_exchange = random.randint(0, 1)
#         if if_exchange == 0:
#             return warp1, warp2, mask1, mask2, depthinput1, depthinput2
#         else:
#             return warp2, warp1, mask2, mask1, depthinput2, depthinput1


# class TestDataset(Dataset):
#     def __init__(self, data_path):

#         self.width = 512
#         self.height = 512

#         self.test_path = data_path
#         self.datas = OrderedDict()

#         datas = glob.glob(os.path.join(self.test_path, '*'))
#         for data in sorted(datas):
#             data_name = os.path.basename(data)
#             if data_name in ['warp1', 'warp2', 'mask1', 'mask2']:
#                 img_list = glob.glob(os.path.join(data, '*.jpg'))
#                 img_list.sort()
#                 self.datas[data_name] = {
#                     'path': data,
#                     'image': img_list,
#                 }

#         print("TestDataset folders:", self.datas.keys())
#         print("num images:", {k: len(v['image']) for k, v in self.datas.items()})

#     def __len__(self):
#         return len(self.datas['warp1']['image'])

#     def _load_rgb(self, path):
#         img = cv2.imread(path, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
#         img = img.astype(np.float32)
#         img = img / 127.5 - 1.0
#         img = np.transpose(img, (2, 0, 1))
#         return torch.from_numpy(img)

#     def _load_mask(self, path):
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
#         img = img.astype(np.float32)
#         img = img / 255.0
#         img = img[np.newaxis, :, :]
#         return torch.from_numpy(img)

#     def __getitem__(self, index):

#         warp1 = self._load_rgb(self.datas['warp1']['image'][index])
#         warp2 = self._load_rgb(self.datas['warp2']['image'][index])

#         mask1 = self._load_mask(self.datas['mask1']['image'][index])
#         mask2 = self._load_mask(self.datas['mask2']['image'][index])

#         return warp1, warp2, mask1, mask2
        
# class TestDepthDataset(Dataset):
#     def __init__(self, data_path):
#         self.test_path = data_path
#         self.datas = OrderedDict()

#         datas = glob.glob(os.path.join(self.test_path, '*'))
#         for data in sorted(datas):
#             data_name = os.path.basename(data)
#             if data_name in ('warp1', 'warp2', 'mask1', 'mask2', 'warp_depth1', 'warp_depth2'):
#                 img_list = sorted(glob.glob(os.path.join(data, '*.jpg')))
#                 self.datas[data_name] = {
#                     'path': data,
#                     'image': img_list,
#                 }

#         print('TestDepthDataset folders:', self.datas.keys())


#         if 'warp1' not in self.datas:
#             raise RuntimeError("warp1 folder not found under test_path; please check directory structure")

#         n = len(self.datas['warp1']['image'])
#         print('num images:', {k: len(v['image']) for k, v in self.datas.items()})


#         assert all(len(v['image']) == n for v in self.datas.values()), \
#             'Test the number of slices in floder is not the same'

#     def __len__(self):
#         return len(self.datas['warp1']['image'])

#     def _load_img(self, path):
#         img = cv2.imread(path, cv2.IMREAD_COLOR)
#         if img is None:
#             raise RuntimeError(f'cv2.imread failed: {path}')
#         img = img.astype(np.float32)
#         img = img / 127.5 - 1.0
#         img = np.transpose(img, (2, 0, 1))  # [C,H,W]
#         return img

#     def _load_mask01(self, path):
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise RuntimeError(f'cv2.imread failed: {path}')
#         img = img.astype(np.float32) / 255.0
#         img = img[np.newaxis, :, :]  # [1,H,W]
#         return img

#     def __getitem__(self, index):
#         warp1 = self._load_img(self.datas['warp1']['image'][index])
#         warp2 = self._load_img(self.datas['warp2']['image'][index])
#         mask1 = self._load_mask01(self.datas['mask1']['image'][index])
#         mask2 = self._load_mask01(self.datas['mask2']['image'][index])
#         depth1 = self._load_mask01(self.datas['warp_depth1']['image'][index])
#         depth2 = self._load_mask01(self.datas['warp_depth2']['image'][index])

#         warp1_tensor = torch.from_numpy(warp1)
#         warp2_tensor = torch.from_numpy(warp2)
#         mask1_tensor = torch.from_numpy(mask1)
#         mask2_tensor = torch.from_numpy(mask2)
#         depth1_tensor = torch.from_numpy(depth1)
#         depth2_tensor = torch.from_numpy(depth2)

#         return warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, depth1_tensor, depth2_tensor
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

        #   train_path/
        #       warp1/
        #       warp2/
        #       mask1/
        #       mask2/
        #       warp_depth1/
        #       warp_depth2/
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = os.path.basename(data)  # 
            if data_name in ['warp1', 'warp2', 'mask1', 'mask2', 'warp_depth1', 'warp_depth2']:
                img_list = glob.glob(os.path.join(data, '*.jpg'))
                img_list.sort()
                self.datas[data_name] = {
                    'path': data,
                    'image': img_list,
                }
        print("TrainDataset folders:", self.datas.keys())
        print("num images:", {k: len(v['image']) for k, v in self.datas.items()})

    def __len__(self):
        return len(self.datas['warp1']['image'])

    # --------- resize ---------

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # H, W, 3
        
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0                 # [-1,1] Warp ׶һ
        img = np.transpose(img, (2, 0, 1))      # C, H, W
        return torch.from_numpy(img)

    def _load_mask_or_depth(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # H, W
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32)
        img = img / 255.0                        # [0,1]
        img = img[np.newaxis, :, :]              # 1, H, W
        return torch.from_numpy(img)

    # --------- ȡһ ---------

    def __getitem__(self, index):

        # load warp1/warp2
        warp1 = self._load_rgb(self.datas['warp1']['image'][index])
        warp2 = self._load_rgb(self.datas['warp2']['image'][index])

        # load mask1/mask2
        mask1 = self._load_mask_or_depth(self.datas['mask1']['image'][index])
        mask2 = self._load_mask_or_depth(self.datas['mask2']['image'][index])

        # depthinput1/2ʵ warp Ľ mask 
        depthinput1 = self._load_mask_or_depth(self.datas['warp_depth1']['image'][index])
        depthinput2 = self._load_mask_or_depth(self.datas['warp_depth2']['image'][index])

        # ңֺ Warp ׶ηһ
        if_exchange = random.randint(0, 1)
        if if_exchange == 0:
            return warp1, warp2, mask1, mask2, depthinput1, depthinput2
        else:
            return warp2, warp1, mask2, mask1, depthinput2, depthinput1


class TestDataset(Dataset):
    def __init__(self, data_path):

        self.width = 512
        self.height = 512

        self.test_path = data_path
        self.datas = OrderedDict()

        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = os.path.basename(data)
            if data_name in ['warp1', 'warp2', 'mask1', 'mask2']:
                img_list = glob.glob(os.path.join(data, '*.jpg'))
                img_list.sort()
                self.datas[data_name] = {
                    'path': data,
                    'image': img_list,
                }

        print("TestDataset folders:", self.datas.keys())
        print("num images:", {k: len(v['image']) for k, v in self.datas.items()})

    def __len__(self):
        return len(self.datas['warp1']['image'])

    def _load_rgb(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

    def _load_mask(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32)
        img = img / 255.0
        img = img[np.newaxis, :, :]
        return torch.from_numpy(img)

    def __getitem__(self, index):

        warp1 = self._load_rgb(self.datas['warp1']['image'][index])
        warp2 = self._load_rgb(self.datas['warp2']['image'][index])

        mask1 = self._load_mask(self.datas['mask1']['image'][index])
        mask2 = self._load_mask(self.datas['mask2']['image'][index])

        return warp1, warp2, mask1, mask2
        
class TestDepthDataset(Dataset):
    def __init__(self, data_path):
        self.test_path = data_path
        self.datas = OrderedDict()

        valid = (
            'warp1', 'warp2', 'mask1', 'mask2', 'warp_depth1', 'warp_depth2',
            'warp_fixed', 'warp3', 'mask_fixed', 'mask3', 'warp_depth_fixed', 'warp_depth3'
        )
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = os.path.basename(data)
            if data_name in valid:
                img_list = sorted(glob.glob(os.path.join(data, '*.jpg')))
                if len(img_list) == 0:
                    img_list = sorted(glob.glob(os.path.join(data, '*.png')))
                self.datas[data_name] = {'path': data, 'image': img_list}

        self.has_three = all(k in self.datas for k in ('warp_fixed', 'warp1', 'warp3', 'mask_fixed', 'mask1', 'mask3'))
        if not self.has_three and 'warp1' not in self.datas:
            raise RuntimeError('warp1 folder not found under test_path; please check directory structure')

        n_key = 'warp_fixed' if self.has_three else 'warp1'
        n = len(self.datas[n_key]['image'])
        assert all(len(v['image']) == n for v in self.datas.values()), 'Test the number of slices in floder is not the same'
        print('TestDepthDataset mode:', 'three-input' if self.has_three else 'two-input')
        print('TestDepthDataset folders:', self.datas.keys())

    def __len__(self):
        n_key = 'warp_fixed' if self.has_three else 'warp1'
        return len(self.datas[n_key]['image'])

    def _load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'cv2.imread failed: {path}')
        img = img.astype(np.float32)
        img = img / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def _load_mask01(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f'cv2.imread failed: {path}')
        img = img.astype(np.float32) / 255.0
        img = img[np.newaxis, :, :]
        return img

    def __getitem__(self, index):
        if self.has_three:
            wf = self._load_img(self.datas['warp_fixed']['image'][index])
            w1 = self._load_img(self.datas['warp1']['image'][index])
            w3 = self._load_img(self.datas['warp3']['image'][index])
            mf = self._load_mask01(self.datas['mask_fixed']['image'][index])
            m1 = self._load_mask01(self.datas['mask1']['image'][index])
            m3 = self._load_mask01(self.datas['mask3']['image'][index])

            df = self._load_mask01(self.datas.get('warp_depth_fixed', self.datas['mask_fixed'])['image'][index])
            d1 = self._load_mask01(self.datas.get('warp_depth1', self.datas['mask1'])['image'][index])
            d3 = self._load_mask01(self.datas.get('warp_depth3', self.datas['mask3'])['image'][index])

            return (torch.from_numpy(wf), torch.from_numpy(w1), torch.from_numpy(w3),
                    torch.from_numpy(mf), torch.from_numpy(m1), torch.from_numpy(m3),
                    torch.from_numpy(df), torch.from_numpy(d1), torch.from_numpy(d3))

        warp1 = self._load_img(self.datas['warp1']['image'][index])
        warp2 = self._load_img(self.datas['warp2']['image'][index])
        mask1 = self._load_mask01(self.datas['mask1']['image'][index])
        mask2 = self._load_mask01(self.datas['mask2']['image'][index])
        depth1 = self._load_mask01(self.datas['warp_depth1']['image'][index])
        depth2 = self._load_mask01(self.datas['warp_depth2']['image'][index])

        return (torch.from_numpy(warp1), torch.from_numpy(warp2),
                torch.from_numpy(mask1), torch.from_numpy(mask2),
                torch.from_numpy(depth1), torch.from_numpy(depth2))
