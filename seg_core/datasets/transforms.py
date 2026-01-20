import torch
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

class RandomHorizontalFlip(object):
    """
    随机水平翻转。
    确保 RGB, Depth, Label 同时翻转。
    """
    def __call__(self, sample):
        if random.random() < 0.5:
            sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
            sample['depth'] = sample['depth'].transpose(Image.FLIP_LEFT_RIGHT)
            if 'label' in sample:
                sample['label'] = sample['label'].transpose(Image.FLIP_LEFT_RIGHT)
        return sample

class RandomScaleCrop(object):
    """
    随机缩放 + 随机裁剪。
    这是语义分割最关键的增强步骤。
    """
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample.get('label')
        
        # --- 1. 随机缩放 (Scale) ---
        # 随机选择缩放比例，例如 0.5 到 2.0
        short_side = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        
        # 保持长宽比计算目标尺寸
        if h > w:
            ow = short_side
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_side
            ow = int(1.0 * w * oh / h)
            
        # 执行缩放 (RGB/Depth 用双线性，Label 用最近邻)
        img = img.resize((ow, oh), Image.BILINEAR)
        depth = depth.resize((ow, oh), Image.BILINEAR)
        if mask:
            mask = mask.resize((ow, oh), Image.NEAREST)

        # --- 2. Padding (如果图片小于裁剪尺寸) ---
        if short_side < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            
            # RGB 补黑色(0)
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            # Depth 补 0 (假设 0 代表最近/无效)
            depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0) 
            # Label 补 255 (Ignore Index)
            if mask:
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # --- 3. 随机裁剪 (Crop) ---
        # ★ 关键修正：生成一次坐标，应用三次 ★
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if mask:
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        sample['image'] = img
        sample['depth'] = depth
        if mask:
            sample['label'] = mask
        return sample

class Normalize(object):
    """
    标准化:
    RGB: (img - mean) / std
    Depth: 0-255 -> 0.0-1.0
    """
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32) / 255.0
        depth = np.array(sample['depth']).astype(np.float32)
        
        # Normalize RGB
        img -= self.mean
        img /= self.std
        
        # Normalize Depth (0~255 -> 0~1)
        # 注意：如果你的Depth是16bit的(0~65535)，这里需要改为 / 65535.0
        # 根据你的描述，它是 0-255 灰度图
        depth = depth / 255.0 

        # numpy (H, W, C) -> torch (C, H, W)
        sample['image'] = torch.from_numpy(img).permute(2, 0, 1).float()
        # numpy (H, W) -> torch (1, H, W)
        sample['depth'] = torch.from_numpy(depth).unsqueeze(0).float() 
        return sample

class ToTensor(object):
    """
    处理 Label 并转换为 Tensor
    逻辑：Label 0 (背景) -> 255 (Ignore)
          Label 1...N -> 0...N-1 (Class ID)
    """
    def __call__(self, sample):
        if 'label' in sample:
            mask = np.array(sample['label'], dtype=np.int32)
            
            # 创建全 255 的矩阵
            new_mask = np.full_like(mask, 255)
            
            # 将 >0 的类别减 1 并赋值回去 (1->0, 2->1 ...)
            # 0 (背景) 保持为 255，不参与 Loss 计算
            new_mask[mask > 0] = mask[mask > 0] - 1
            
            sample['label'] = torch.from_numpy(new_mask).long()
            
        return sample