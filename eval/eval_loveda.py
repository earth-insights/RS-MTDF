import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataset.semi import SemiDataset
import torch.nn.functional as F
from model.semseg.dpt_distill_with_models_fuse import DPT
import numpy as np
from tqdm import tqdm
import yaml
import logging
from datetime import datetime
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
import os
config_path = "./configs/loveda.yaml"

# Load configuration from YAML file
with open(config_path, "r") as file:
    cfg = yaml.safe_load(file)
model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
model_path = ''
state_dict = torch.load(model_path)
model_state_dict = state_dict['model']
new_model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
model.load_state_dict(new_model_state_dict, strict=True)
model = model.cuda()  # Move the model to GPU


testset = SemiDataset(
    cfg['dataset'], cfg['data_root'], 'test'
)
testloader = DataLoader(
    testset, batch_size=8, pin_memory=True, num_workers=1, drop_last=False
)
def evaluate(model, loader, mode, cfg, multiplier=None, model_path=None):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    pred_meter = AverageMeter()
    target_meter = AverageMeter()

    # 自动根据 model_path 生成保存路径
    if model_path is not None:
        parts = model_path.strip(os.sep).split(os.sep)
        if len(parts) >= 2:
            folder_name = parts[-2]
        else:
            folder_name = "result"
        save_dir = os.path.join('./eval', folder_name)
        os.makedirs(save_dir, exist_ok=True)
        save_txt = os.path.join(save_dir, f"{folder_name}_result.txt")
    else:
        save_txt = "./eval/result.txt"
        os.makedirs("./eval", exist_ok=True)

    with torch.no_grad():
        for img, mask, _ in tqdm(loader, desc="Evaluating", unit="batch"):
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()

                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: row + grid, col: col + grid])
                        final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)

                pred = final
            else:
                assert mode == 'original'
                if multiplier is not None:
                    ori_h, ori_w = img.shape[-2:]
                    new_h = int(ori_h / multiplier + 0.5) * multiplier
                    new_w = int(ori_w / multiplier + 0.5) * multiplier
                    img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)

                pred = model(img)

                if multiplier is not None:
                    pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)

            pred = pred.argmax(dim=1)

            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            pred_sum = np.bincount(pred.cpu().numpy().reshape(-1), minlength=cfg['nclass'])
            target_sum = target

            intersection_meter.update(intersection)
            union_meter.update(union)
            pred_meter.update(pred_sum)
            target_meter.update(target)

    intersection = intersection_meter.sum
    union = union_meter.sum
    pred_sum = pred_meter.sum
    target_sum = target_meter.sum

    iou_class = intersection / (union + 1e-10)
    precision = intersection / (pred_sum + 1e-10)
    recall = intersection / (target_sum + 1e-10)
    f1_class = 2 * precision * recall / (precision + recall + 1e-10)

    mIoU = np.mean(iou_class) * 100
    mF1 = np.mean(f1_class) * 100
    total_pixels = np.sum(target_sum)
    p0 = np.sum(intersection) / (total_pixels + 1e-10)
    pe = np.sum(pred_sum * target_sum) / ((total_pixels ** 2) + 1e-10)
    kappa = (p0 - pe) / (1 - pe + 1e-10)

    # 打印结果
    print(f"mIoU: {mIoU:.2f}%")
    print(f"mF1: {mF1:.2f}%")
    print(f"Kappa: {kappa:.4f}")
    print("Per-class IoU & F1:")
    for i, (iou, f1) in enumerate(zip(iou_class, f1_class)):
        print(f"Class {i}: IoU = {iou * 100:.2f}%, F1 = {f1 * 100:.2f}%")

    # 保存结果到 txt 文件，包括 model_path
    with open(save_txt, "w") as f:
        if model_path is not None:
            f.write(f"Model path: {model_path}\n\n")
        f.write(f"mIoU: {mIoU:.2f}%\n")
        f.write(f"mF1: {mF1:.2f}%\n")
        f.write(f"Kappa: {kappa:.4f}\n\n")
        f.write("Per-class results:\n")
        for i, (iou, f1) in enumerate(zip(iou_class, f1_class)):
            f.write(f"Class {i}: IoU = {iou * 100:.2f}%, F1 = {f1 * 100:.2f}%\n")

    return mIoU, mF1, kappa, iou_class, f1_class


mIoU, mF1, kappa, iou_class, f1_class = evaluate(
    model, testloader, 'original', cfg, multiplier=14, model_path=model_path
) 