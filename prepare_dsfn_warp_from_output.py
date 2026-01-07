import os
import argparse
from glob import glob

import SimpleITK as sitk
import numpy as np
import cv2


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_nrrd(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # [z, h, w]
    return arr


def normalize_to_uint8(slice2d, low=0.5, high=99.5):
    slice2d = slice2d.astype(np.float32)
    vmin, vmax = np.percentile(slice2d, [low, high])
    if vmax <= vmin:
        vmax = vmin + 1.0
    slice2d = np.clip(slice2d, vmin, vmax)
    slice2d = (slice2d - vmin) / (vmax - vmin + 1e-6)
    return (slice2d * 255.0).astype(np.uint8)


def get_case_dirs(out_root):
    dirs = []
    for name in os.listdir(out_root):
        full = os.path.join(out_root, name)
        if os.path.isdir(full):
            dirs.append(full)
    return sorted(dirs)


def process_case(case_dir,
                 dsfn_root,
                 resize_hw=(512, 512),
                 skip_empty=True):
    """
    case_dir: 例如 output/000
    里面有:
        DATA_000_AP_warp.nrrd
        DATA_000_MED_warp.nrrd
        DATA_000_LAT_warp.nrrd
        MASK_000_AP_warp.nrrd
        MASK_000_MED_warp.nrrd
        MASK_000_LAT_warp.nrrd

    目标：
        input1      <- AP
        input2      <- MED
        input3      <- LAT
        depthInput1 <- AP mask
        depthInput2 <- MED mask
        depthInput3 <- LAT mask
    """
    case_id = os.path.basename(case_dir.rstrip("\\/"))

    def data_path(view):
        return os.path.join(case_dir, f"DATA_{case_id}_{view}_warp.nrrd")

    def mask_path(view):
        return os.path.join(case_dir, f"MASK_{case_id}_{view}_warp.nrrd")

    # 三个视图：AP / MED / LAT
    views = ["AP", "MED", "LAT"]

    data_paths = {v: data_path(v) for v in views}
    mask_paths = {v: mask_path(v) for v in views}

    # 检查文件是否齐全
    for v in views:
        if not os.path.exists(data_paths[v]):
            print(f"[WARN] {case_id}: DATA_{v} 缺失，跳过该 case")
            return 0
        if not os.path.exists(mask_paths[v]):
            print(f"[WARN] {case_id}: MASK_{v} 缺失，跳过该 case")
            return 0

    # 读体数据
    vol_data = {v: read_nrrd(data_paths[v]) for v in views}
    vol_mask = {v: read_nrrd(mask_paths[v]) for v in views}

    # 形状检查
    shape0 = vol_data["AP"].shape
    for v in views:
        assert vol_data[v].shape == shape0, \
            f"{case_id}: DATA shape mismatch for {v}: {vol_data[v].shape}"
        assert vol_mask[v].shape == shape0, \
            f"{case_id}: MASK shape mismatch for {v}: {vol_mask[v].shape}"

    # 输出目录
    out_input = {
        "AP":  os.path.join(dsfn_root, "input1"),
        "MED": os.path.join(dsfn_root, "input2"),
        "LAT": os.path.join(dsfn_root, "input3"),
    }
    out_depth = {
        "AP":  os.path.join(dsfn_root, "depthInput1"),
        "MED": os.path.join(dsfn_root, "depthInput2"),
        "LAT": os.path.join(dsfn_root, "depthInput3"),
    }
    for d in list(out_input.values()) + list(out_depth.values()):
        ensure_dir(d)

    Z = shape0[0]
    saved = 0
    for z in range(Z):
        # 当前切片的三视图 mask
        masks_z = {v: vol_mask[v][z] for v in views}

        # 如果三张都没有结节，可以选择跳过
        if skip_empty and all(mz.max() == 0 for mz in masks_z.values()):
            continue

        # 逐视图处理：归一化 + resize + 转 3 通道图、mask
        imgs_3c = {}
        masks_2d = {}
        for v in views:
            img = vol_data[v][z]
            m = masks_z[v]

            img_u8 = normalize_to_uint8(img)
            img_r = cv2.resize(img_u8, resize_hw, interpolation=cv2.INTER_LINEAR)
            img_3c = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)

            m_bin = (m > 0).astype(np.uint8) * 255
            m_r = cv2.resize(m_bin, resize_hw, interpolation=cv2.INTER_NEAREST)

            imgs_3c[v] = img_3c
            masks_2d[v] = m_r

        tag = f"{case_id}_z{z:03d}.jpg"

        # 保存：AP→input1/depthInput1，MED→input2/depthInput2，LAT→input3/depthInput3
        cv2.imwrite(os.path.join(out_input["AP"],  tag), imgs_3c["AP"])
        cv2.imwrite(os.path.join(out_input["MED"], tag), imgs_3c["MED"])
        cv2.imwrite(os.path.join(out_input["LAT"], tag), imgs_3c["LAT"])

        cv2.imwrite(os.path.join(out_depth["AP"],  tag), masks_2d["AP"])
        cv2.imwrite(os.path.join(out_depth["MED"], tag), masks_2d["MED"])
        cv2.imwrite(os.path.join(out_depth["LAT"], tag), masks_2d["LAT"])

        saved += 1

    print(f"[CASE] {case_id}: saved {saved} slices.")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_root",
        type=str,
        default="./output_30",
        help="simulate_3views 生成的根目录（里面按 case_id 分子文件夹）"
    )
    parser.add_argument(
        "--dsfn_root",
        type=str,
        default="./dsfn_warp_train_3views",
        help="要生成的 DSFN Warp 训练集目录（包含 input1/2/3 & depthInput1/2/3）"
    )
    parser.add_argument(
        "--skip_empty",
        action="store_true",
        help="跳过三个视图都没有结节的切片"
    )
    args = parser.parse_args()

    case_dirs = get_case_dirs(args.output_root)
    print(f"Found {len(case_dirs)} cases in {args.output_root}")

    total = 0
    for cdir in case_dirs:
        total += process_case(
            cdir,
            dsfn_root=args.dsfn_root,
            skip_empty=args.skip_empty
        )
    print(f"[DONE] total saved slices: {total}")


if __name__ == "__main__":
    main()
