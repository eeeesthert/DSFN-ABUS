# coding: utf-8
import argparse
import os
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from network import build_model, Network, RepConvN
from dataset import TestDepthDataset


last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')


def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_data = TestDepthDataset(data_path=args.test_path)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=0,      
        shuffle=False,
        drop_last=False
    )

    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    for m in net.modules():
        if isinstance(m, RepConvN):
            m.fuse_convs()
            m.forward = m.forward_fuse  # update forward

    # load checkpoint
    ckpt_list = glob.glob(os.path.join(MODEL_DIR, "*.pth"))
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')
        return

    ave_fusion_dir = os.path.join(args.test_path, "ave_fusion")
    os.makedirs(ave_fusion_dir, exist_ok=True)

    out_mask_dir = os.path.join(args.test_path, "fusion_nodule_mask")
    os.makedirs(out_mask_dir, exist_ok=True)

    path_learn_mask1 = '../learn_mask1/'
    os.makedirs(path_learn_mask1, exist_ok=True)
    path_learn_mask2 = '../learn_mask2/'
    os.makedirs(path_learn_mask2, exist_ok=True)
    path_final_composition = '../composition/'
    os.makedirs(path_final_composition, exist_ok=True)

    print("##################start testing#######################")
    net.eval()

    for i, batch_value in enumerate(test_loader):

        warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, depth1_tensor, depth2_tensor = batch_value

        warp1_tensor = warp1_tensor.float()
        warp2_tensor = warp2_tensor.float()
        mask1_tensor = mask1_tensor.float()
        mask2_tensor = mask2_tensor.float()
        depth1_tensor = depth1_tensor.float()
        depth2_tensor = depth2_tensor.float()

        if torch.cuda.is_available():
            warp1_tensor = warp1_tensor.cuda()
            warp2_tensor = warp2_tensor.cuda()
            mask1_tensor = mask1_tensor.cuda()
            mask2_tensor = mask2_tensor.cuda()
            depth1_tensor = depth1_tensor.cuda()
            depth2_tensor = depth2_tensor.cuda()

        with torch.no_grad():
          
            batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

        if i == 0:
            print("batch_out keys:", batch_out.keys())

      
        stitched_image = batch_out['stitched_image']   
        learned_mask1 = batch_out['learned_mask1']    
        learned_mask2 = batch_out['learned_mask2']

   
        if learned_mask1.shape[1] > 1:
            learned_mask1 = learned_mask1[:, :1, ...]
            learned_mask2 = learned_mask2[:, :1, ...]


        fused_nodule = depth1_tensor * learned_mask1 + depth2_tensor * learned_mask2
        fused_nodule_bin = (fused_nodule > 0.5).float()

        stitched_np = ((stitched_image[0] + 1) * 127.5).clamp(0, 255).cpu().numpy().transpose(1, 2, 0)
        stitched_np = stitched_np.astype(np.uint8)
        save_name = f"{i+1:06d}.jpg"
        cv2.imwrite(os.path.join(ave_fusion_dir, save_name), stitched_np)


        mask_np = fused_nodule_bin[0, 0].cpu().numpy()
        mask_np = (mask_np * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_mask_dir, f"{i+1:06d}.png"), mask_np)


        lm1_np = (learned_mask1[0] * 255).cpu().numpy().transpose(1, 2, 0)
        lm2_np = (learned_mask2[0] * 255).cpu().numpy().transpose(1, 2, 0)
        cv2.imwrite(os.path.join(path_learn_mask1, save_name), lm1_np)
        cv2.imwrite(os.path.join(path_learn_mask2, save_name), lm2_np)
        cv2.imwrite(os.path.join(path_final_composition, save_name), stitched_np)

        if (i + 1) % 200 == 0:
            print(f"processed {i+1} slices")

        print('i = {}'.format(i + 1))


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument(
        '--test_path',
        type=str,
        default='~/wwmt_tdsc/simulated/dsfn_warp_cv/fold1/train'
    )

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    args.test_path = os.path.expanduser(args.test_path)
    print(args)

    test(args)
