# coding: utf-8
import argparse
import os
import glob
import cv2
import torch
from torch.utils.data import DataLoader, Subset
import imageio

from network import build_depth_output_model, Network
from dataset import TestDepthOutDataset 
import grid_res

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')


def draw_mesh_on_warp(warp, f_local):
    point_color = (0, 255, 0)  # BGR
    thickness = 2
    lineType = 8

    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i + 1, j, 0]), int(f_local[i + 1, j, 1])),
                    point_color,
                    thickness,
                    lineType,
                )
            elif i == grid_h:
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i, j + 1, 0]), int(f_local[i, j + 1, 1])),
                    point_color,
                    thickness,
                    lineType,
                )
            else:
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i + 1, j, 0]), int(f_local[i + 1, j, 1])),
                    point_color,
                    thickness,
                    lineType,
                )
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i, j + 1, 0]), int(f_local[i, j + 1, 1])),
                    point_color,
                    thickness,
                    lineType,
                )

    return warp


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def test(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    test_data = TestDepthOutDataset(data_path=args.test_path)

    if args.limit_cases > 0:
        input1_paths = test_data.datas["input1"]["image"]  

        selected_indices = []   
        seen_case_ids = []   

        for idx, p in enumerate(input1_paths):
            basename = os.path.basename(p) 
            case_id = basename.split('_')[0]

            if case_id not in seen_case_ids:
                if len(seen_case_ids) >= args.limit_cases:
                    continue
                seen_case_ids.append(case_id)

            if case_id in seen_case_ids:
                selected_indices.append(idx)

        print(f"[DEBUG] limit_cases={args.limit_cases}, "
              f"chosen={len(seen_case_ids)}, 2D slices={len(selected_indices)}")

        test_data = Subset(test_data, selected_indices)

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        drop_last=False,
    )

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    ckpt_list = glob.glob(os.path.join(MODEL_DIR, "*.pth"))
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')

    print("##################start testing#######################")

    path_ave_fusion = os.path.join(args.test_path, 'ave_fusion')
    os.makedirs(path_ave_fusion, exist_ok=True)

    path_warp1 = os.path.join(args.test_path, 'warp1')
    os.makedirs(path_warp1, exist_ok=True)

    path_warp2 = os.path.join(args.test_path, 'warp2')
    os.makedirs(path_warp2, exist_ok=True)

    path_mask1 = os.path.join(args.test_path, 'mask1')
    os.makedirs(path_mask1, exist_ok=True)

    path_mask2 = os.path.join(args.test_path, 'mask2')
    os.makedirs(path_mask2, exist_ok=True)

    path_depth1 = os.path.join(args.test_path, 'warp_depth1')
    os.makedirs(path_depth1, exist_ok=True)

    path_depth2 = os.path.join(args.test_path, 'warp_depth2')
    os.makedirs(path_depth2, exist_ok=True)

    net.eval()
    for i, batch_value in enumerate(test_loader):

        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()
        depthInput1_tensor = batch_value[2].float()
        depthInput2_tensor = batch_value[3].float()
        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()
            depthInput1_tensor = depthInput1_tensor.cuda()
            depthInput2_tensor = depthInput2_tensor.cuda()

        with torch.no_grad():
            batch_out = build_depth_output_model(
                net,
                inpu1_tesnor,
                inpu2_tesnor,
                depthInput1_tensor,
                depthInput2_tensor,
            )

        final_warp1 = batch_out['final_warp1']
        final_warp1_mask = batch_out['final_warp1_mask']
        final_warp2 = batch_out['final_warp2']
        final_warp2_mask = batch_out['final_warp2_mask']
        final_mesh1 = batch_out['mesh1']
        final_mesh2 = batch_out['mesh2']
        depth_warp1 = batch_out['depth_warp1']
        depth_warp2 = batch_out['depth_warp2']

        final_warp1_np = ((final_warp1[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
        final_warp2_np = ((final_warp2[0] + 1) * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
        final_warp1_mask_np = final_warp1_mask[0].cpu().detach().numpy().transpose(1, 2, 0)
        final_warp2_mask_np = final_warp2_mask[0].cpu().detach().numpy().transpose(1, 2, 0)
        final_mesh1_np = final_mesh1[0].cpu().detach().numpy()
        final_mesh2_np = final_mesh2[0].cpu().detach().numpy()
        depth_warp1_np = depth_warp1.cpu().detach().numpy().transpose(1, 2, 0)
        depth_warp2_np = depth_warp2.cpu().detach().numpy().transpose(1, 2, 0)

        idx_str = str(i + 1).zfill(6)

        cv2.imwrite(os.path.join(path_warp1, idx_str + ".jpg"), final_warp1_np)
        cv2.imwrite(os.path.join(path_warp2, idx_str + ".jpg"), final_warp2_np)
        cv2.imwrite(os.path.join(path_mask1, idx_str + ".jpg"), final_warp1_mask_np * 255)
        cv2.imwrite(os.path.join(path_mask2, idx_str + ".jpg"), final_warp2_mask_np * 255)
        cv2.imwrite(os.path.join(path_depth1, idx_str + ".jpg"), depth_warp1_np)
        cv2.imwrite(os.path.join(path_depth2, idx_str + ".jpg"), depth_warp2_np)

        ave_fusion = final_warp1_np * (final_warp1_np / (final_warp1_np + final_warp2_np + 1e-6)) + \
                     final_warp2_np * (final_warp2_np / (final_warp1_np + final_warp2_np + 1e-6))
        cv2.imwrite(os.path.join(path_ave_fusion, idx_str + ".jpg"), ave_fusion)

        print('i = {}'.format(i + 1))

        torch.cuda.empty_cache()

    print("##################end testing#######################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument(
        '--test_path',
        type=str,
        default='~/wwmt_tdsc/simulated/dsfn_warp_cv/fold1/train',
        help=' input1/input2/depthInput1/depthInput2'
    )
    parser.add_argument(
        '--limit_cases',
        type=int,
        default=30,
        help=-1
    )

    print('<==================== Loading data ===================>\n')
    args = parser.parse_args()
    args.test_path = os.path.expanduser(args.test_path)
    print(args)

    test(args)
