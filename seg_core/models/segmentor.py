import torch
import torch.nn as nn
import torch.nn.functional as F

class RGBDSegmentor(nn.Module):
    """
    双流 RGB-D 分割模型主类 (Two-Stream Segmentor).
    结构:
        RGB Input   --> RGB Backbone   --\
                                          (+) --> Fused Features --> Decoder Head --> Output
        Depth Input --> Depth Backbone --/
    """
    def __init__(self, rgb_backbone, depth_backbone, head, n_classes):
        """
        Args:
            rgb_backbone (nn.Module): 处理 RGB 的骨干网络 (e.g. ResNet50)
            depth_backbone (nn.Module): 处理 Depth 的骨干网络 (e.g. ResNet50)
            head (nn.Module): 解码头 (e.g. FCNHead)
            n_classes (int): 类别数 (用于最后形状检查等，虽然Head里已经有了)
        """
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.depth_backbone = depth_backbone
        self.head = head
        self.n_classes = n_classes

    def forward(self, data_dict):
        """
        Args:
            data_dict (dict): 包含 'image', 'depth', 'label'
        Returns:
            dict: 包含 'pred' (logits) 和 'loss' (如果是训练模式)
        """
        # 1. 解包数据
        # (B, 3, H, W)
        img = data_dict['image']
        # (B, 1, H, W)
        depth = data_dict['depth']
        
        # 2. 深度图预处理
        # 标准 ResNet 期望 3 通道输入，而深度图只有 1 通道。
        # 策略：在通道维度复制 3 份 (B, 1, H, W) -> (B, 3, H, W)
        if depth.shape[1] == 1:
            depth = depth.repeat(1, 3, 1, 1)

        # 3. 特征提取 (Backbone Extraction)
        # 返回的是特征列表 [c1, c2, c3, c4]
        rgb_feats = self.rgb_backbone(img)
        depth_feats = self.depth_backbone(depth)

        # 4. 特征融合 (Feature Fusion)
        # 策略：逐层相加 (Element-wise Sum)
        # 这种 "Late Fusion" 或 "Multi-level Fusion" 策略简单且有效
        fused_feats = []
        for f_rgb, f_depth in zip(rgb_feats, depth_feats):
            fused_feats.append(f_rgb + f_depth)

        # 5. 解码预测 (Decode)
        # 输入融合后的特征，输出 (B, n_classes, H/32, W/32)
        logits = self.head(fused_feats)

        # 6. 上采样回原图尺寸 (Upsample)
        # 使用双线性插值
        logits = F.interpolate(logits, size=img.shape[2:], mode='bilinear', align_corners=False)

        # 7. 组装输出
        output = {'pred': logits}

        # 8. 计算损失 (仅在训练且有标签时)
        if self.training and 'label' in data_dict:
            label = data_dict['label'] # (B, H, W)
            # 计算交叉熵损失，自动忽略 255
            loss = F.cross_entropy(logits, label, ignore_index=255)
            output['loss'] = loss

        return output