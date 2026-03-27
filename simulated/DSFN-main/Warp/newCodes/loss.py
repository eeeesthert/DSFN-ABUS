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


def cal_nipple_heatmap_loss(target_heatmap, warped_heatmap, overlap_mask=None, coord_weight=0.2, pair_valid=None):
    """
    target_heatmap: [B,1,H,W], fixed-view nipple heatmap
    warped_heatmap: [B,1,H,W], moved-view heatmap warped to fixed view
    overlap_mask: optional [B,1,H,W]
    """
    target_heatmap = target_heatmap.float().clamp(0.0, 1.0)
    warped_heatmap = warped_heatmap.float().clamp(0.0, 1.0)

    if overlap_mask is not None:
        if overlap_mask.shape[-2:] != target_heatmap.shape[-2:]:
            overlap_mask = F.interpolate(overlap_mask.float(), size=target_heatmap.shape[-2:], mode='nearest')
        valid = (overlap_mask > 0.5).float()
    else:
        valid = torch.ones_like(target_heatmap)

    if pair_valid is not None:
        pair_valid = pair_valid.view(-1, 1, 1, 1).to(valid.device).float()
        valid = valid * pair_valid

    denom = valid.sum().clamp_min(1.0)
    mse = ((target_heatmap - warped_heatmap) ** 2 * valid).sum() / denom

    target_coord = soft_argmax_2d(target_heatmap)
    warped_coord = soft_argmax_2d(warped_heatmap)
    coord_each = F.smooth_l1_loss(warped_coord, target_coord, reduction='none').mean(dim=1)
    if pair_valid is not None:
        pv = pair_valid.view(-1).to(coord_each.device).float()
        coord_loss = (coord_each * pv).sum() / pv.sum().clamp_min(1.0)
    else:
        coord_loss = coord_each.mean()

    return mse + coord_weight * coord_loss


def cal_depth_loss(depth1, depth2, output_H, output_H_inv, warp_mesh, warp_mesh_mask):
    """Backward-compatible alias: now used as nipple heatmap supervision."""
    return cal_nipple_heatmap_loss(depth1, depth2, overlap_mask=warp_mesh_mask[:, 0:1, ...])
