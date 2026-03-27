# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
# import torch.nn as nn
import imageio
# from network import build_output_model, Network
# from dataset import *
import os
import cv2
import time
from network import build_output_model_three, Network
from dataset import TestDataset

# import grid_res
# grid_h = grid_res.GRID_H
# grid_w = grid_res.GRID_W
import glob
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')


# def draw_mesh_on_warp(warp, f_local):


#     point_color = (0, 255, 0) # BGR
#     thickness = 2
#     lineType = 8

#     num = 1
#     for i in range(grid_h+1):
#         for j in range(grid_w+1):

#             num = num + 1
#             if j == grid_w and i == grid_h:
#                 continue
#             elif j == grid_w:
#                 cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
#             elif i == grid_h:
#                 cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
#             else :
#                 cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
#                 cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)

#     return warp


def create_gif(image_list, gif_name, duration=0.35):
    # frames = []
    # for image_name in image_list:
    #     frames.append(image_name)
    # imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    # return
    imageio.mimsave(gif_name, image_list, 'GIF', duration=duration)




def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # dataset
    test_data = TestDataset(data_path=args.test_path)
   
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    #load the existing models if it exists
    # ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    # ckpt_list.sort()
    # if len(ckpt_list) != 0:
    #     model_path = ckpt_list[-1]
    #     #model_path = '/opt/data/private/nl/Repository/Unsupervised_Mesh_Stitching/UDISv2-88/UDISv2-Homo_TPS88-10grid_NO-res50-new3/model/epoch150_model.pth'
    #     checkpoint = torch.load(model_path)
    ckpt_list = sorted(glob.glob(MODEL_DIR + "/*.pth"))
    if ckpt_list:
        checkpoint = torch.load(ckpt_list[-1])
        net.load_state_dict(checkpoint['model'])

        # print('load model from {}!'.format(model_path))
        print('load model from {}!'.format(ckpt_list[-1]))
    else:
        print('No checkpoint found!')
    output_dir = os.path.join(args.test_path, 'stitch3_out')
    os.makedirs(output_dir, exist_ok=True)



    # print("##################start testing#######################")
    # # create folders if it dose not exist

    # path_ave_fusion = '../ave_fusion/'
    # if not os.path.exists(path_ave_fusion):
    #     os.makedirs(path_ave_fusion)
    # path_warp1 = args.test_path + 'warp1/'
    # if not os.path.exists(path_warp1):
    #     os.makedirs(path_warp1)
    # path_warp2 = args.test_path + 'warp2/'
    # if not os.path.exists(path_warp2):
    #     os.makedirs(path_warp2)
    # path_mask1 = args.test_path + 'mask1/'
    # if not os.path.exists(path_mask1):
    #     os.makedirs(path_mask1)
    # path_mask2 = args.test_path + 'mask2/'
    # if not os.path.exists(path_mask2):
    #     os.makedirs(path_mask2)

    # total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print("Number of trainable parameters:", total_params)
    net.eval()
    start_time = time.perf_counter()

    # net.eval()
    for i, batch_value in enumerate(test_loader):

        #if i != 975:
        #    continue

        # inpu1_tesnor = batch_value[0].float()
        # inpu2_tesnor = batch_value[1].float()
        input1, input2, input3 = [x.float() for x in batch_value]

        if torch.cuda.is_available():
            # inpu1_tesnor = inpu1_tesnor.cuda()
            # inpu2_tesnor = inpu2_tesnor.cuda()
            input1, input2, input3 = input1.cuda(), input2.cuda(), input3.cuda()

        with torch.no_grad():
            # batch_out = build_output_model(net, inpu1_tesnor, inpu2_tesnor)

        # final_warp1 = batch_out['final_warp1']
        # final_warp1_mask = batch_out['final_warp1_mask']
        # final_warp2 = batch_out['final_warp2']
        # final_warp2_mask = batch_out['final_warp2_mask']
        # final_mesh1 = batch_out['mesh1']
        # final_mesh2 = batch_out['mesh2']


        # final_warp1 = ((final_warp1[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        # final_warp2 = ((final_warp2[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        # final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1,2,0)
        # final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1,2,0)
        # final_mesh1 = final_mesh1[0].cpu().detach().numpy()
        # final_mesh2 = final_mesh2[0].cpu().detach().numpy()
            out = build_output_model_three(net, input1, input2, input3, alpha=args.alpha)



        # path = path_warp1 + str(i+1).zfill(6) + ".jpg"
        # cv2.imwrite(path, final_warp1)
        # path = path_warp2 + str(i+1).zfill(6) + ".jpg"
        # cv2.imwrite(path, final_warp2)
        # path = path_mask1 + str(i+1).zfill(6) + ".jpg"
        # cv2.imwrite(path, final_warp1_mask*255)
        # path = path_mask2 + str(i+1).zfill(6) + ".jpg"
        # cv2.imwrite(path, final_warp2_mask*255)
        fixed = ((out['final_warp_fixed'][0] + 1) * 127.5).cpu().numpy().transpose(1, 2, 0)
        moved1 = ((out['final_warp1'][0] + 1) * 127.5).cpu().numpy().transpose(1, 2, 0)
        moved3 = ((out['final_warp3'][0] + 1) * 127.5).cpu().numpy().transpose(1, 2, 0)

        # ave_fusion = final_warp1 * (final_warp1/ (final_warp1+final_warp2+1e-6)) + final_warp2 * (final_warp2/ (final_warp1+final_warp2+1e-6))
        # path = path_ave_fusion + str(i+1).zfill(6) + ".jpg"
        # cv2.imwrite(path, ave_fusion)
        fuse = fixed * (fixed / (fixed + moved1 + moved3 + 1e-6)) + \
               moved1 * (moved1 / (fixed + moved1 + moved3 + 1e-6)) + \
               moved3 * (moved3 / (fixed + moved1 + moved3 + 1e-6))

        # print('i = {}'.format( i+1))
        cv2.imwrite(os.path.join(output_dir, f"{i+1:06d}_fixed.jpg"), fixed)
        cv2.imwrite(os.path.join(output_dir, f"{i+1:06d}_moved1.jpg"), moved1)
        cv2.imwrite(os.path.join(output_dir, f"{i+1:06d}_moved3.jpg"), moved3)
        cv2.imwrite(os.path.join(output_dir, f"{i+1:06d}_fused.jpg"), fuse)

        # torch.cuda.empty_cache()
        print(f"i={i+1}, left_nonoverlap={out['left_nonoverlap'].item():.2f}, right_nonoverlap={out['right_nonoverlap'].item():.2f}, right_limit={out['right_limit'].item():.2f}")


    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time  # 计算经过的时间
    # print(f"执行时间：{elapsed_time}秒")
    print(f"elapsed: {time.perf_counter() - start_time:.2f}s")


    # print("##################end testing#######################")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    
    # # /opt/data/private/nl/Data/UDIS-D/testing/  or  /opt/data/private/nl/Data/UDIS-D/training/
    # parser.add_argument('--test_path', type=str, default='/root/UDIS2/dataSet/testing/')  # IVSD_vis_testing      challenge      testing 


    # print('<==================== Loading data ===================>\n')

    parser.add_argument('--test_path', type=str, default='/root/UDIS2/dataSet/testing/')
    parser.add_argument('--alpha', type=float, default=0.8)
    args = parser.parse_args()
    print(args)
    test(args)
