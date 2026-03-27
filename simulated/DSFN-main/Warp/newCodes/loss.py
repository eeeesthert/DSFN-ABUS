import torch
import torch.nn as nn
import torch.nn.functional as F

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W


def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))


def cal_lp_loss(input1, input2, output_H, output_H_inv, warp_mesh, warp_mesh_mask):
    batch_size, _, img_h, img_w = input1.size()

    # part one: sym homo loss with color balance
    delta1 = ( torch.sum(output_H[:,0:3,:,:], [2,3])  -   torch.sum(input1*output_H[:,3:6,:,:], [2,3]) ) /  torch.sum(output_H[:,3:6,:,:], [2,3])

    input1_balance = input1 + delta1.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta2 = ( torch.sum(output_H_inv[:,0:3,:,:], [2,3])  -   torch.sum(input2*output_H_inv[:,3:6,:,:], [2,3]) ) /  torch.sum(output_H_inv[:,3:6,:,:], [2,3])
    input2_balance = input2 + delta2.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_1 = l_num_loss(input1_balance*output_H[:,3:6,:,:], output_H[:,0:3,:,:], 1) + l_num_loss(input2_balance*output_H_inv[:,3:6,:,:], output_H_inv[:,0:3,:,:], 1)

    # part two: tps loss with color balance
    delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(input1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
    input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = l_num_loss(input1_newbalance*warp_mesh_mask, warp_mesh, 1)


    lp_loss = 3. * lp_loss_1 + 1. * lp_loss_2

    return lp_loss

def cal_lp_loss2(input1, warp_mesh, warp_mesh_mask):
    batch_size, _, img_h, img_w = input1.size()

    delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(input1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
    input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = l_num_loss(input1_newbalance*warp_mesh_mask, warp_mesh, 1)
    lp_loss =  1. * lp_loss_2

    return lp_loss

def inter_grid_loss(overlap, mesh):

    ##############################
    # compute horizontal edges
    w_edges = mesh[:,:,0:grid_w,:] - mesh[:,:,1:grid_w+1,:]
    # compute angles of two successive horizontal edges
    cos_w = torch.sum(w_edges[:,:,0:grid_w-1,:] * w_edges[:,:,1:grid_w,:],3) / (torch.sqrt(torch.sum(w_edges[:,:,0:grid_w-1,:]*w_edges[:,:,0:grid_w-1,:],3))*torch.sqrt(torch.sum(w_edges[:,:,1:grid_w,:]*w_edges[:,:,1:grid_w,:],3)))
    # horizontal angle-preserving error for two successive horizontal edges
    delta_w_angle = 1 - cos_w
    # horizontal angle-preserving error for two successive horizontal grids
    delta_w_angle = delta_w_angle[:,0:grid_h,:] + delta_w_angle[:,1:grid_h+1,:]
    ##############################

    ##############################
    # compute vertical edges
    h_edges = mesh[:,0:grid_h,:,:] - mesh[:,1:grid_h+1,:,:]
    # compute angles of two successive vertical edges
    cos_h = torch.sum(h_edges[:,0:grid_h-1,:,:] * h_edges[:,1:grid_h,:,:],3) / (torch.sqrt(torch.sum(h_edges[:,0:grid_h-1,:,:]*h_edges[:,0:grid_h-1,:,:],3))*torch.sqrt(torch.sum(h_edges[:,1:grid_h,:,:]*h_edges[:,1:grid_h,:,:],3)))
    # vertical angle-preserving error for two successive vertical edges
    delta_h_angle = 1 - cos_h
    # vertical angle-preserving error for two successive vertical grids
    delta_h_angle = delta_h_angle[:,:,0:grid_w] + delta_h_angle[:,:,1:grid_w+1]
    ##############################

    # on overlapping regions
    depth_diff_w = (1-torch.abs(overlap[:,:,0:grid_w-1] - overlap[:,:,1:grid_w])) * overlap[:,:,0:grid_w-1]
    error_w = depth_diff_w * delta_w_angle
    # on overlapping regions
    depth_diff_h = (1-torch.abs(overlap[:,0:grid_h-1,:] - overlap[:,1:grid_h,:])) * overlap[:,0:grid_h-1,:]
    error_h = depth_diff_h * delta_h_angle

    return torch.mean(error_w) + torch.mean(error_h)



# intra-grid constraint
def intra_grid_loss(pts):

    max_w = 512/grid_w * 2
    max_h = 512/grid_h * 2

    delta_x = pts[:,:,1:grid_w+1,0] - pts[:,:,0:grid_w,0]
    delta_y = pts[:,1:grid_h+1,:,1] - pts[:,0:grid_h,:,1]

    loss_x = F.relu(delta_x - max_w)
    loss_y = F.relu(delta_y - max_h)
    loss = torch.mean(loss_x) + torch.mean(loss_y)


    return loss



# ----- add -----

# def depth_loss(depth_warp1, depth_warp2, depth_mask1, depth_mask2):
#     mask_overlap = depth_warp1 * depth_warp2
#     depth_mask_overlap = torch.where(mask_overlap > 0, torch.ones_like(mask_overlap), torch.zeros_like(mask_overlap))
# #     print("depth_mask1.shape:")
# #     print(depth_mask1.shape)
# #     print("depth_mask2.shape:")
# #     print(depth_mask2.shape)
# #     print("depth_mask_overlap.shape:")
# #     print(depth_mask_overlap.shape)
# #     print("depth_warp1.shape:")
# #     print(depth_warp1.shape)
#     masked_depth_warp1 = depth_warp1 * depth_mask_overlap
#     masked_depth_warp2 = depth_warp2 * depth_mask_overlap
#     ## Z分数归一化
#     # min_val1 = torch.min(masked_depth_warp1[depth_mask_overlap == 1])
#     # max_val1 = torch.max(masked_depth_warp1[depth_mask_overlap == 1])
#     # min_val2 = torch.min(masked_depth_warp2[depth_mask_overlap == 1])
#     # max_val2 = torch.max(masked_depth_warp1[depth_mask_overlap == 1])
#     # min_depth = torch.min(min_val1, min_val2)
#     depth_mean = (torch.mean(masked_depth_warp1[depth_mask_overlap == 1]) + torch.mean(masked_depth_warp2[depth_mask_overlap == 1])) / 2
#     depth_std = (torch.std(masked_depth_warp1[depth_mask_overlap == 1]) + torch.std(masked_depth_warp2[depth_mask_overlap == 1])) / 2
#     z_score_depth_warp1 = (masked_depth_warp1 - depth_mean) / depth_std
#     z_score_depth_warp2 = (masked_depth_warp2 - depth_mean) / depth_std
#     final_z_depth_1 = z_score_depth_warp1 * depth_mask_overlap
#     final_z_depth_2 = z_score_depth_warp2 * depth_mask_overlap

#     depth_loss = torch.abs(final_z_depth_1 - final_z_depth_2) ** 2
    
#     return torch.mean(depth_loss)

# #def depth_num_loss(depth_img1, depth_img2, l_num=1):
#     # Z-score standardization 
# #     depth_mean = (torch.mean(depth_img1) + torch.mean(depth_img2)) / 2
# #     depth_std = (torch.std(depth_img1) + torch.std(depth_img2)) / 2
# #     z_score_depth1 = (depth_img1 - depth_mean) / depth_std
# #     z_score_depth2 = (depth_img2 - depth_mean) / depth_std
# #     return torch.mean(torch.abs((z_score_depth1 - z_score_depth2)**l_num))
#   #  combined_depth_maps = torch.cat((depth_img1,depth_img2))
#   #  min_score = torch.min(combined_depth_maps)
#   #  max_score = torch.max(combined_depth_maps)

#    # new_depth1 = (depth_img1 - min_score) / (max_score - min_score)
#    # new_depth2 = (depth_img2 - min_score) / (max_score - min_score)
#    # return torch.mean(torch.abs((new_depth1 - new_depth2)**l_num))
   

# def depth_num_loss(depth_img1, depth_img2, l_num=1):
#     combined_depth_maps = torch.cat((depth_img1, depth_img2))
#     min_score = torch.min(combined_depth_maps)
#     max_score = torch.max(combined_depth_maps)
#     # ⭐ 防止 max_score == min_score（全 0 / 常数图）导致除 0
#     denom = (max_score - min_score).clamp_min(1e-6)
#     new_depth1 = (depth_img1 - min_score) / denom
#     new_depth2 = (depth_img2 - min_score) / denom
#     return torch.mean(torch.abs((new_depth1 - new_depth2)**l_num))


# def cal_mask_loss(mask1, mask2, warp_mask):
#     """
#     mask1, mask2: [B, 1, H, W], 0/1
#     warp_mask:    [B, 1, H, W], 由网络预测出来的 “融合域 / 变形后域”
#     这里只做一个很简单的 L2 匹配作为示例：
#     """
#     # 把 mask1 warp 到对方视角（这里用 warp_mesh_mask 近似当权重）
#     # 简化版：只在 warp 区域内约束 mask1 和 mask2 相似
#     overlap_region = warp_mask > 0.5
#     if overlap_region.sum() == 0:
#         return torch.tensor(0.0, device=mask1.device, dtype=mask1.dtype)
#     diff = (mask1 - mask2) ** 2
#     diff = diff * warp_mask
#     return diff.sum() / overlap_region.sum()

# '''

# def cal_depth_loss(depthInput1, depthInput2, output_H, output_H_inv, warp_mesh, warp_mesh_mask):
#     batch_size, _, img_h, img_w = depthInput1.size()

#     # part one: sym homo loss with color balance
#     delta1 = ( torch.sum(output_H[:,0:3,:,:], [2,3])  -   torch.sum(depthInput1*output_H[:,3:6,:,:], [2,3]) ) /  torch.sum(output_H[:,3:6,:,:], [2,3])
#     depth_input1_balance = depthInput1 + delta1.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

#     delta2 = ( torch.sum(output_H_inv[:,0:3,:,:], [2,3])  -   torch.sum(depthInput2*output_H_inv[:,3:6,:,:], [2,3]) ) /  torch.sum(output_H_inv[:,3:6,:,:], [2,3])
#     depth_input2_balance = depthInput2 + delta2.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

#     depth_lp_loss_1 = depth_num_loss(depth_input1_balance*output_H[:,3:6,:,:], output_H[:,0:3,:,:], 1) + depth_num_loss(depth_input2_balance*output_H_inv[:,3:6,:,:], output_H_inv[:,0:3,:,:], 1)

#     # part two: tps loss with color balance
#     delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(depthInput1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
#     depth_input1_newbalance = depthInput1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

#     depth_lp_loss_2 = depth_num_loss(depth_input1_newbalance*warp_mesh_mask, warp_mesh, 1)


#     depth_lp_loss = 3. * depth_lp_loss_1 + 1. * depth_lp_loss_2

#     return depth_lp_loss
# '''
# def cal_depth_loss(depth1, depth2, output_H, output_H_inv,
#                    warp_mesh, warp_mesh_mask):
#     """
#     现在用作结节 mask 的对齐约束：
#     depth1, depth2: [B, 1, H, W]，值在 [0, 1]
#     warp_mesh_mask: [B, 1, H, W]，表示 warp/overlap 区域（越大越重要）

#     这里简单地在 warp 区域内做一个对称 BCE：
#         BCE(depth1, depth2) + BCE(depth2, depth1) / 2
#     """

#     depth1 = depth1.float()
#     depth2 = depth2.float()

#     eps = 1e-6
#     depth1 = depth1.clamp(eps, 1.0 - eps)
#     depth2 = depth2.clamp(eps, 1.0 - eps)

#     if warp_mesh_mask is not None:
#         mask = (warp_mesh_mask > 0.5).float()

#         # 如果大小不一致，插值到 depth 的大小
#         if mask.shape != depth1.shape:
#             mask = F.interpolate(mask, size=depth1.shape[-2:], mode="nearest")

#         valid = mask.sum()
#         if valid.item() == 0:
#             # 没有重叠区域就直接返回 0，不给梯度
#             return torch.tensor(0.0, device=depth1.device, dtype=depth1.dtype)

#         bce1 = F.binary_cross_entropy(depth1, depth2, reduction='none')
#         bce2 = F.binary_cross_entropy(depth2, depth1, reduction='none')
#         bce = 0.5 * (bce1 + bce2)  # 对称 BCE

#         bce = bce * mask
#         loss = bce.sum() / valid
#     else:
#         loss1 = F.binary_cross_entropy(depth1, depth2)
#         loss2 = F.binary_cross_entropy(depth2, depth1)
#         loss = 0.5 * (loss1 + loss2)

#     return loss
# ----- nipple heatmap branch -----
def soft_argmax_2d(heatmap):
    """heatmap: [B,1,H,W], value in [0,1], return normalized xy in [-1,1]."""
    b, _, h, w = heatmap.shape
    flat = heatmap.view(b, -1)
    prob = F.softmax(flat, dim=1).view(b, 1, h, w)
    ys = torch.linspace(-1.0, 1.0, h, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, h, 1)
    xs = torch.linspace(-1.0, 1.0, w, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, 1, w)
    exp_x = (prob * xs).sum(dim=(2, 3))
    exp_y = (prob * ys).sum(dim=(2, 3))
    return torch.cat([exp_x, exp_y], dim=1)
    
def cal_nipple_heatmap_loss(target_heatmap, warped_heatmap, overlap_mask=None, coord_weight=0.2):
    target_heatmap: [B,1,H,W], fixed-view nipple heatmap
    warped_heatmap: [B,1,H,W], moved-view heatmap warped to fixed view
    overlap_mask: optional [B,1,H,W]
    target_heatmap = target_heatmap.float().clamp(0.0, 1.0)
    warped_heatmap = warped_heatmap.float().clamp(0.0, 1.0)

    if overlap_mask is not None:
        if overlap_mask.shape[-2:] != target_heatmap.shape[-2:]:
            overlap_mask = F.interpolate(overlap_mask.float(), size=target_heatmap.shape[-2:], mode='nearest')
        valid = (overlap_mask > 0.5).float()
        denom = valid.sum().clamp_min(1.0)
        mse = ((target_heatmap - warped_heatmap) ** 2 * valid).sum() / denom
    else:
        mse = F.mse_loss(target_heatmap, warped_heatmap)
    target_coord = soft_argmax_2d(target_heatmap)
    warped_coord = soft_argmax_2d(warped_heatmap)
    coord_loss = F.smooth_l1_loss(warped_coord, target_coord)
    return mse + coord_weight * coord_loss
def cal_depth_loss(depth1, depth2, output_H, output_H_inv, warp_mesh, warp_mesh_mask):
    """Backward-compatible alias: now used as nipple heatmap supervision."""
    return cal_nipple_heatmap_loss(depth1, depth2, overlap_mask=warp_mesh_mask[:, 0:1, ...])
