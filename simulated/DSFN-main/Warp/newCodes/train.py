import argparse
import torch
from torch.utils.data import DataLoader, Subset
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, Network, build_train_model
from dataset import TrainDataset
import glob
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss, cal_depth_loss, cal_mask_loss
import pandas as pd


def overlay_mask_for_tb(img, mask, color=(1.0, 0.0, 0.0), alpha=0.5):

    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    img01 = (img + 1.0) / 2.0
    img01 = img01.clamp(0.0, 1.0)

    mask01 = mask.clamp(0.0, 1.0)
    if mask01.shape[0] == 1:
        mask3 = mask01.repeat(3, 1, 1)
    else:
        mask3 = mask01

    c = torch.tensor(color, dtype=img01.dtype, device=img01.device).view(3, 1, 1)
    overlay = img01 * (1 - alpha * mask3) + c * (alpha * mask3)
    overlay = overlay.clamp(0.0, 1.0)
    return overlay

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
# path to save the summary files
SUMMARY_DIR = os.path.join(last_path, 'summary')
writer = SummaryWriter(log_dir=SUMMARY_DIR)
# path to save the model files
MODEL_DIR = os.path.join(last_path, 'model')
# create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # define dataset
    train_data = TrainDataset(data_path=args.train_path)
        # if you only want to use first N ABUS cases (volumes) for quick debug
    if args.limit_cases > 0:
        input1_paths = train_data.datas["input1"]["image"] 

        selected_indices = []   
        seen_case_ids = []         
        for idx, p in enumerate(input1_paths):
            basename = os.path.basename(p)       
            case_id = basename.split('_')[0]  

            if case_id not in seen_case_ids:
                #new case
                if len(seen_case_ids) >= args.limit_cases:
                    # skip
                    continue
                seen_case_ids.append(case_id)

            
            if case_id in seen_case_ids:
                selected_indices.append(idx)

        print(f"[DEBUG] limit_cases={args.limit_cases}, number of chosen ={len(seen_case_ids)}, number of 2D slices={len(selected_indices)}")

       
        train_data = Subset(train_data, selected_indices)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from stratch!')

    print("##################start training#######################")
    score_print_fre = 300

    for epoch in range(start_epoch, args.max_epoch):

        print("start epoch {}".format(epoch))
        net.train()
        loss_sigma = 0.0
        overlap_loss_sigma = 0.
        nonoverlap_loss_sigma = 0.
        overlap_depth_loss_sigma = 0.

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader):

            inpu1_tesnor = batch_value[0].float()
            inpu2_tesnor = batch_value[1].float()
            depthInput1_tensor = batch_value[2].float()
            depthInput2_tensor = batch_value[3].float()
            if torch.cuda.is_available():
                inpu1_tesnor = inpu1_tesnor.cuda()
                inpu2_tesnor = inpu2_tesnor.cuda()
                depthInput1_tensor = depthInput1_tensor.cuda()
                depthInput2_tensor = depthInput2_tensor.cuda()

            # forward, backward, update weights
            optimizer.zero_grad()

            #batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor)# build_train_model(net, inpu1_tesnor, inpu2_tesnor)#, depthInput1_tensor, depthInput2_tensor)
            batch_out = build_train_model(net,
                              inpu1_tesnor,
                              inpu2_tesnor,
                              depthInput1_tensor,
                              depthInput2_tensor)

            # result
            output_H = batch_out['output_H']
            output_H_inv = batch_out['output_H_inv']
            warp_mesh = batch_out['warp_mesh']
            warp_mesh_mask = batch_out['warp_mesh_mask']
            mesh1 = batch_out['mesh1']
            mesh2 = batch_out['mesh2']
            overlap = batch_out['overlap']
            
            depth_warp1 = batch_out['depth_warp1']   # [B,1,H,W] warp_H 
            depth_warp2 = batch_out['depth_warp2']   # [B,1,H,W] warp_mesh
#             depth_warp1 = batch_out['depth_warp1']
#             depth_warp2 = batch_out['depth_warp2']
#             depth_mask1 = batch_out['depth_mask1']
#             depth_mask2 = batch_out['depth_mask2']
            
            
            # calculate loss for overlapping regions
            overlap_loss = cal_lp_loss(inpu1_tesnor, inpu2_tesnor, output_H, output_H_inv, warp_mesh, warp_mesh_mask)
            # calculate loss for non-overlapping regions
            nonoverlap_loss = 10*inter_grid_loss(overlap, mesh2) + 10*intra_grid_loss(mesh2)
            
            # ----- add ----- calculate overlap depth loss
#             overlap_depth_loss = depth_loss(depth_warp1, depth_warp2, depth_mask1, depth_mask2)
            overlap_depth_loss = cal_depth_loss(depthInput1_tensor, depthInput2_tensor, output_H, output_H_inv, warp_mesh, warp_mesh_mask)
            #overlap_depth_loss = cal_mask_loss(depthInput1_tensor, depthInput2_tensor, warp_mesh_mask)
            # conf
            #overlap_depth_loss = 0.3 * overlap_depth_loss
            overlap_depth_loss = 0.1 * overlap_depth_loss

            total_loss = overlap_loss +  nonoverlap_loss +  overlap_depth_loss
            if not torch.isfinite(total_loss):
                print(f"[WARN] non-finite loss at epoch {epoch}, iter {i}:",
                      "overlap_loss=", overlap_loss.item(),
                      "nonoverlap_loss=", nonoverlap_loss.item(),
                      "overlap_depth_loss=", overlap_depth_loss.item())
                continue
            total_loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            overlap_loss_sigma += overlap_loss.item()
            nonoverlap_loss_sigma += nonoverlap_loss.item()
            overlap_depth_loss_sigma += overlap_depth_loss.item()
            loss_sigma += total_loss.item()

#             print(glob_iter)

            # record loss and images in tensorboard
            if i % score_print_fre == 0 and i != 0:
                average_loss = loss_sigma / score_print_fre
                average_overlap_loss = overlap_loss_sigma/ score_print_fre
                average_nonoverlap_loss = nonoverlap_loss_sigma/ score_print_fre
                average_overlap_depth_loss = overlap_depth_loss_sigma/ score_print_fre
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                nonoverlap_loss_sigma = 0.
                overlap_depth_loss_sigma = 0.
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Overlap Loss: {:.4f}  Non-overlap Loss: {:.4f} overlap_depth_loss:{:.4f} lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader),
                                          average_loss, average_overlap_loss, average_nonoverlap_loss, average_overlap_depth_loss, optimizer.state_dict()['param_groups'][0]['lr']))
                # visualization
                writer.add_image("inpu1", (inpu1_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("inpu2", (inpu2_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("warp_H", (output_H[0,0:3,:,:]+1.)/2., glob_iter)
                writer.add_image("warp_mesh", (warp_mesh[0]+1.)/2., glob_iter)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('overlap loss', average_overlap_loss, glob_iter)
                writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, glob_iter)
                writer.add_scalar('depth_loss', average_overlap_depth_loss, glob_iter)
                in1_overlay = overlay_mask_for_tb(
                inpu1_tesnor[0].detach().cpu(),
                depthInput1_tensor[0].detach().cpu())
                in2_overlay = overlay_mask_for_tb(
                inpu2_tesnor[0].detach().cpu(),
                depthInput2_tensor[0].detach().cpu())
            # warp_H + warp  mask(depth_warp1)
                warpH_overlay = overlay_mask_for_tb(
                    output_H[0, 0:3].detach().cpu(),
                    depth_warp1[0].detach().cpu())
            # warp_mesh + warp  mask(depth_warp2)
                warpMesh_overlay = overlay_mask_for_tb(
                    warp_mesh[0].detach().cpu(),
                    depth_warp2[0].detach().cpu()
                )
        
                writer.add_image("inpu1_mask", in1_overlay, glob_iter)
                writer.add_image("inpu2_mask", in2_overlay, glob_iter)
                writer.add_image("warp_H_mask", warpH_overlay, glob_iter)
                writer.add_image("warp_mesh_mask", warpMesh_overlay, glob_iter)
                writer.add_image("raw_mask1", depthInput1_tensor[0], glob_iter)
                writer.add_image("raw_mask2", depthInput2_tensor[0], glob_iter)
                writer.add_image("raw_warpH_mask", depth_warp1[0], glob_iter)
                writer.add_image("raw_warpMesh_mask", depth_warp2[0], glob_iter)

            glob_iter += 1


        scheduler.step()
        # save model
        # for Ablation Study , only save epoch 50 and eopch 100
        if ((epoch+1) % 100 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model_.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)
    print("##################end training#######################")


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default=r'~/wwmt_tdsc/simulated/dsfn_warp_cv/fold1/train')
    parser.add_argument('--limit_cases', type=int, default=30,
                        help='-1 all')

    # parse the arguments
    args = parser.parse_args()
    args.train_path = os.path.expanduser(args.train_path)
    print(args)

    # train
    train(args)




