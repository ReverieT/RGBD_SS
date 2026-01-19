"""
* 文件说明：验证RGB和深度图是对齐的，且标签转换正确。
"""
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 将项目根目录添加到 python path，确保能 import seg_core
sys.path.append(os.getcwd())

from seg_core.datasets.base_dataset import RGBXDataset
import seg_core.datasets.transforms as T

# ================= 配置区域 =================
# 请修改为你本地的数据集路径
DATA_ROOT = 'data/NYUDepthv2'  
# DATA_ROOT = 'data/SUNRGBD'  
# 假设你的 train.txt 也在这个目录下
SPLIT = 'train' 
# ===========================================

def denormalize(tensor, mean, std):
    """
    反归一化，用于可视化
    Tensor (3, H, W) -> Numpy (H, W, 3) [0, 255]
    """
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    img = tensor.numpy()
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img.transpose(1, 2, 0)

def visualize_batch(sample, index):
    """
    绘制 RGB, Depth, Label 三联图
    """
    img_tensor = sample['image']
    depth_tensor = sample['depth']
    label_tensor = sample['label']

    # 1. 处理 RGB
    img_vis = denormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 2. 处理 Depth (1, H, W) -> (H, W)
    # 之前归一化到了 [0, 1]，这里直接显示热力图即可
    depth_vis = depth_tensor.squeeze().numpy()

    # 3. 处理 Label
    # 255 是 Ignore, 显示为黑色; 0, 1, 2... 显示为不同灰度/颜色
    label_vis = label_tensor.numpy()
    # 为了可视化方便，把 255 (Ignore) 暂时变成 -1 或者某个显眼的颜色，这里用 cmap 处理
    
    # 开始绘图
    plt.figure(figsize=(12, 4))
    
    # RGB
    plt.subplot(1, 3, 1)
    plt.imshow(img_vis)
    plt.title(f"RGB (Sample {index})")
    plt.axis('off')

    # Depth
    plt.subplot(1, 3, 2)
    plt.imshow(depth_vis, cmap='inferno') # 使用热力图显示深度
    plt.title(f"Depth (Val: {depth_vis.min():.2f}~{depth_vis.max():.2f})")
    plt.axis('off')

    # Label
    plt.subplot(1, 3, 3)
    #创建一个自定义的显示，把255显示为白色，其他显示为彩色
    masked_label = np.ma.masked_where(label_vis == 255, label_vis)
    plt.imshow(masked_label, cmap='jet', interpolation='nearest')
    plt.title(f"Label (Unique: {np.unique(masked_label)})")
    plt.axis('off')

    plt.tight_layout()
    plt.show() # 如果是远程服务器，请用 plt.savefig(f'debug_{index}.png')

def main():
    # 1. 定义增强流程
    # 包含 Resize, RandomCrop, Flip, Normalize, ToTensor
    transforms = T.Compose([
        T.RandomScaleCrop(base_size=520, crop_size=480), # 模拟训练时的裁剪
        T.RandomHorizontalFlip(),
        T.Normalize(),
        T.ToTensor()
    ])

    # 2. 实例化 Dataset
    try:
        dataset = RGBXDataset(
            root_dir=DATA_ROOT, 
            split=SPLIT, 
            transforms=transforms,
            depth_ext='.png', # 你的深度图后缀
            label_ext='.png'  # 你的标签图后缀
        )
    except Exception as e:
        print(f"❌ 初始化 Dataset 失败: {e}")
        return

    print(f"✅ 成功加载数据集，共 {len(dataset)} 个样本。")
    print(f"   准备读取前 3 个样本进行检查...\n")

    # 3. 循环读取并检查
    for i in range(3):
        try:
            sample = dataset[i]
            
            # 打印 Tensor 形状信息
            print(f"--- Sample {i} ---")
            print(f"Image shape: {sample['image'].shape} (Expected: 3, H, W)")
            print(f"Depth shape: {sample['depth'].shape} (Expected: 1, H, W)")
            print(f"Label shape: {sample['label'].shape} (Expected: H, W)")
            
            # 检查 Label 数值
            unique_labels = torch.unique(sample['label'])
            print(f"Label classes in this crop: {unique_labels.tolist()} (255 is ignore)")
            
            # 可视化
            visualize_batch(sample, i)
            
        except Exception as e:
            print(f"❌ 读取样本 {i} 失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()