# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import imageio
from network import build_output_model_three, Network
from dataset import TestDataset
import os
import cv2
import time
import glob

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')


def create_gif(image_list, gif_name, duration=0.35):
    imageio.mimsave(gif_name, image_list, 'GIF', duration=duration)


def test(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    ckpt_list = sorted(glob.glob(MODEL_DIR + "/*.pth"))
    if ckpt_list:
        checkpoint = torch.load(ckpt_list[-1])
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(ckpt_list[-1]))
    else:
        print('No checkpoint found!')

    output_dir = os.path.join(args.test_path, 'stitch3_out')
    os.makedirs(output_dir, exist_ok=True)

    net.eval()
    start_time = time.perf_counter()

    for i, batch_value in enumerate(test_loader):
        input1, input2, input3 = [x.float() for x in batch_value]
        if torch.cuda.is_available():
            input1, input2, input3 = input1.cuda(), input2.cuda(), input3.cuda()

        with torch.no_grad():
            out = build_output_model_three(net, input1, input2, input3, alpha=args.alpha)

        fixed = ((out['final_warp_fixed'][0] + 1) * 127.5).cpu().numpy().transpose(1, 2, 0)
        moved1 = ((out['final_warp1'][0] + 1) * 127.5).cpu().numpy().transpose(1, 2, 0)
        moved3 = ((out['final_warp3'][0] + 1) * 127.5).cpu().numpy().transpose(1, 2, 0)

        fuse = fixed * (fixed / (fixed + moved1 + moved3 + 1e-6)) + \
               moved1 * (moved1 / (fixed + moved1 + moved3 + 1e-6)) + \
               moved3 * (moved3 / (fixed + moved1 + moved3 + 1e-6))

        cv2.imwrite(os.path.join(output_dir, f"{i+1:06d}_fixed.jpg"), fixed)
        cv2.imwrite(os.path.join(output_dir, f"{i+1:06d}_moved1.jpg"), moved1)
        cv2.imwrite(os.path.join(output_dir, f"{i+1:06d}_moved3.jpg"), moved3)
        cv2.imwrite(os.path.join(output_dir, f"{i+1:06d}_fused.jpg"), fuse)

        print(f"i={i+1}, left_nonoverlap={out['left_nonoverlap'].item():.2f}, right_nonoverlap={out['right_nonoverlap'].item():.2f}, right_limit={out['right_limit'].item():.2f}")

    print(f"elapsed: {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='/root/UDIS2/dataSet/testing/')
    parser.add_argument('--alpha', type=float, default=0.8)
    args = parser.parse_args()
    print(args)
    test(args)
