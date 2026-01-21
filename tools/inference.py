import argparse
import os
import sys
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from seg_core.datasets.base_dataset import RGBXDataset
from seg_core.datasets import transforms as T
from seg_core.models.backbones.resnet import ResNet
from seg_core.models.decoders.fcn_head import FCNHead
from seg_core.models.segmentor import RGBDSegmentor
from seg_core.utils.config_parser import parse_config

# 定义颜色盘 (生成 40 种随机颜色)
def get_color_palette(n_classes):
    np.random.seed(42) # 固定随机种子保证颜色一致
    return np.random.randint(0, 255, (n_classes, 3), dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs/inference_results")
    args = parser.parse_args()

    # 1. 配置与设备
    cfg = parse_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 模型
    rgb_backbone = ResNet(depth=50, pretrained=False)
    depth_backbone = ResNet(depth=50, pretrained=False)
    head = FCNHead(in_channels=2048, channels=512, num_classes=cfg.dataset.n_classes)
    model = RGBDSegmentor(rgb_backbone, depth_backbone, head, cfg.dataset.n_classes)
    
    # 加载权重
    print(f"Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 3. 数据集 (这里我们复用测试集进行推理，也可以改为读取单张图片)
    # 关键：推理时只做 Normalize，不要 Resize/Crop 导致变形，或者保持和 Validation 一致
    val_trans = T.Compose([
        T.Normalize(mean=cfg.transforms.mean, std=cfg.transforms.std),
        T.ToTensor()
    ])
    # 这里我们只推断前 50 张图作为演示
    dataset = RGBXDataset(cfg.dataset.root, split=cfg.dataset.val_split, transforms=val_trans)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    colors = get_color_palette(cfg.dataset.n_classes)
    print(f"Start inference on {len(dataset)} images. Saving to {args.output_dir}...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 50: break # 只跑前 50 张，避免磁盘爆满

            name = batch['name'][0]
            images = batch['image'].to(device)
            depths = batch['depth'].to(device)
            
            # 推理
            output = model({'image': images, 'depth': depths})
            pred = output['pred'].argmax(dim=1).cpu().numpy()[0] # (H, W)

            # --- 可视化 ---
            # 1. 还原 RGB 图像用于底图 (Denormalize)
            # input was (x - mean) / std -> x = input * std + mean
            mean = np.array(cfg.transforms.mean).reshape(3, 1, 1)
            std = np.array(cfg.transforms.std).reshape(3, 1, 1)
            rgb_vis = batch['image'][0].numpy()
            rgb_vis = (rgb_vis * std + mean) * 255.0
            rgb_vis = np.clip(rgb_vis, 0, 255).astype(np.uint8).transpose(1, 2, 0) # (H, W, 3)
            # OpenCV 使用 BGR，转一下
            rgb_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)

            # 2. 生成彩色 Mask
            h, w = pred.shape
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 遍历每个像素填色 (向量化加速)
            for cid in range(cfg.dataset.n_classes):
                color_mask[pred == cid] = colors[cid]

            # 3. 图像融合 (AddWeighted)
            # output = 0.6 * RGB + 0.4 * Mask
            vis_result = cv2.addWeighted(rgb_vis, 0.6, color_mask, 0.4, 0)

            # 4. 保存
            save_path = os.path.join(args.output_dir, f"{name}_pred.png")
            cv2.imwrite(save_path, vis_result)
            
            if i % 10 == 0:
                print(f"Saved {save_path}")

    print("Done!")

if __name__ == '__main__':
    main()
