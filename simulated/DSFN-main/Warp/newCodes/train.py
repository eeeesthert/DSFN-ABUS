import argparse
import torch
from torch.utils.data import DataLoader, Subset
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import Network, build_train_model
from dataset import TrainDataset
import glob
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss, cal_nipple_heatmap_loss


last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
SUMMARY_DIR = os.path.join(last_path, 'summary')
MODEL_DIR = os.path.join(last_path, 'model')
writer = SummaryWriter(log_dir=SUMMARY_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


def _pair_losses(fixed_img, moved_img, fixed_heat, moved_heat, pair_valid, net):
    out = build_train_model(net, fixed_img, moved_img, fixed_heat, moved_heat)

    overlap_loss = cal_lp_loss(fixed_img, moved_img, out['output_H'], out['output_H_inv'], out['warp_mesh'], out['warp_mesh_mask'])
    nonoverlap_loss = 10 * inter_grid_loss(out['overlap'], out['mesh2']) + 10 * intra_grid_loss(out['mesh2'])
    heat_loss = cal_nipple_heatmap_loss(fixed_heat, out['depth_warp2'], overlap_mask=out['warp_mesh_mask'][:, 0:1, ...], pair_valid=pair_valid)

    return out, overlap_loss, nonoverlap_loss, heat_loss


def train(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_data = TrainDataset(data_path=args.train_path)
    if args.limit_cases > 0:
        input2_paths = train_data.datas["input2"]["image"]
        selected_indices, seen_case_ids = [], []
        for idx, p in enumerate(input2_paths):
            case_id = os.path.basename(p).split('_')[0]
            if case_id not in seen_case_ids and len(seen_case_ids) < args.limit_cases:
                seen_case_ids.append(case_id)
            if case_id in seen_case_ids:
                selected_indices.append(idx)
        train_data = Subset(train_data, selected_indices)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    ckpt_list = sorted(glob.glob(MODEL_DIR + "/*.pth"))
    if ckpt_list:
        checkpoint = torch.load(ckpt_list[-1])
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
    else:
        start_epoch = 0
        glob_iter = 0

    for epoch in range(start_epoch, args.max_epoch):
        net.train()
        for i, batch_value in enumerate(train_loader):
            input1, input2, input3, heat1, heat2, heat3, valid1, valid2, valid3 = [x.float() for x in batch_value]
            if torch.cuda.is_available():
                input1, input2, input3 = input1.cuda(), input2.cuda(), input3.cuda()
                heat1, heat2, heat3 = heat1.cuda(), heat2.cuda(), heat3.cuda()
                valid1, valid2, valid3 = valid1.cuda(), valid2.cuda(), valid3.cuda()

            optimizer.zero_grad()

            # pair 1: input1 -> input2 (input2 fixed)
            pair12_valid = valid1 * valid2
            out12, ov12, non12, h12 = _pair_losses(input2, input1, heat2, heat1, pair12_valid, net)
            # pair 2: input3 -> input2 (input2 fixed)
            pair32_valid = valid3 * valid2
            out32, ov32, non32, h32 = _pair_losses(input2, input3, heat2, heat3, pair32_valid, net)

            left_nonoverlap = torch.clamp(-torch.min(out12['mesh2'][..., 0]), min=0.0)
            right_nonoverlap = torch.clamp(torch.max(out32['mesh2'][..., 0]) - input2.size(-1), min=0.0)
            width_constraint = torch.abs(right_nonoverlap - args.alpha * left_nonoverlap)

            total_loss = (ov12 + ov32) + (non12 + non32) + args.heatmap_weight * (h12 + h32) + args.width_weight * width_constraint
            if not torch.isfinite(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            if i % 100 == 0:
                print(f"epoch={epoch} iter={i} loss={total_loss.item():.4f} ov={(ov12+ov32).item():.4f} non={(non12+non32).item():.4f} heat={(h12+h32).item():.4f} width={width_constraint.item():.4f}")
                writer.add_scalar('loss/total', total_loss.item(), glob_iter)

            glob_iter += 1

        scheduler.step()
        if ((epoch + 1) % 100 == 0 or (epoch + 1) == args.max_epoch):
            filename = 'epoch' + str(epoch + 1).zfill(3) + '_model_.pth'
            torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'glob_iter': glob_iter},
                       os.path.join(MODEL_DIR, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default=r'~/wwmt_tdsc/simulated/dsfn_warp_cv/fold1/train')
    parser.add_argument('--limit_cases', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--heatmap_weight', type=float, default=0.1)
    parser.add_argument('--width_weight', type=float, default=0.1)
    args = parser.parse_args()
    args.train_path = os.path.expanduser(args.train_path)
    print(args)
    train(args)
