import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加路径以便调用你的 Dataset 类
sys.path.append(os.getcwd())
from seg_core.datasets.base_dataset import RGBXDataset

def analyze_label_values():
    # 1. 配置路径 (修改为你真实的数据路径)
    data_root = 'data/NYUDepthv2'
    
    # 2. 随机读取一张 Label
    # 直接用 Dataset 类读，省去拼路径的麻烦，但要关掉 transform 以便看原始值
    dataset = RGBXDataset(data_root, split='train', transforms=None)
    
    print(f"数据集共有 {len(dataset)} 张图片。")
    
    # 我们检查前 3 张有代表性的图
    for i in range(3):
        print(f"\n--- 分析第 {i} 张样本 ---")
        sample = dataset[i]
        label = np.array(sample['label']) # 转 numpy
        
        # 3. 统计数值
        unique_values = np.unique(label)
        print(f"图片名称: {sample['name']}")
        print(f"Label 尺寸: {label.shape}")
        print(f"包含的像素值 (Unique Values): {unique_values}")
        
        # 计算 255 的占比
        ignore_count = np.sum(label == 255)
        total_pixels = label.size
        print(f"255 (Ignore) 像素占比: {ignore_count / total_pixels * 100:.2f}%")
        
        # 4. 可视化：把 255 区域高亮显示
        plt.figure(figsize=(10, 5))
        
        # 左图：原始 Label (为了看清，把 255 暂时变成 0)
        show_label = label.copy()
        show_label[show_label == 255] = 0 
        plt.subplot(1, 2, 1)
        plt.imshow(show_label, cmap='jet')
        plt.title(f"Label (255 hidden as 0)\nClasses: {len(unique_values)-1}")
        plt.axis('off')
        
        # 右图：专门显示 255 在哪
        # 生成一个二值掩码：255的地方是白色，其他是黑色
        ignore_mask = (label == 255).astype(np.uint8) * 255
        plt.subplot(1, 2, 2)
        plt.imshow(ignore_mask, cmap='gray')
        plt.title("Where is 255?\n(White = Ignore/255)")
        plt.axis('off')
        
        # 保存或显示
        plt.tight_layout()
        # 如果是服务器没有屏幕，请用 savefig
        save_path = f"debug_label_analysis_{i}.png"
        plt.savefig(save_path)
        print(f"分析图已保存至: {save_path}")
        plt.close()

if __name__ == '__main__':
    analyze_label_values()