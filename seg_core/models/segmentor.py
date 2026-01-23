import torch
import torch.nn as nn
import torch.nn.functional as F

class RGBDSegmentor(nn.Module):
    def __init__(self, rgb_backbone, depth_backbone=None, head=None, n_classes=40):
        super().__init__()
        self.n_classes = n_classes
        self.head = head
        
        # 智能检测：是否为单流统一模型 (如 DFormer)
        self.is_unified = getattr(rgb_backbone, 'is_unified', False)
        
        if self.is_unified:
            # 如果是 DFormer，我们只需要一个 backbone
            self.backbone = rgb_backbone
            # depth_backbone 此时应该被忽略或为 None
        else:
            # 传统的双流 ResNet
            self.rgb_backbone = rgb_backbone
            self.depth_backbone = depth_backbone

    def forward(self, data_dict):
        img = data_dict['image']
        depth = data_dict['depth']
        
        if self.is_unified:
            # === 分支 A: 单流模型 (DFormer) ===
            # 直接同时传入 RGB 和 Depth
            features = self.backbone(img, depth)
            # 智能判断 Head 需要什么输入
            if hasattr(self.head, 'input_indices'):
                # 如果是 HamHead，它定义了需要的索引 [1, 2, 3]
                selected_features = [features[i] for i in self.head.input_indices]
                logits = self.head(selected_features)
            else:
                # 默认 FCN，只取最后一个
                logits = self.head(features[-1])
            
        else:
            # === 分支 B: 双流模型 (ResNet) ===
            if depth.shape[1] == 1:
                depth = depth.repeat(1, 3, 1, 1)
                
            rgb_feats = self.rgb_backbone(img)
            depth_feats = self.depth_backbone(depth)
            
            # 简单的相加融合
            fused_feats = []
            for f_rgb, f_depth in zip(rgb_feats, depth_feats):
                fused_feats.append(f_rgb + f_depth)
                
            logits = self.head(fused_feats)

        # 后处理：上采样
        logits = F.interpolate(logits, size=img.shape[2:], mode='bilinear', align_corners=False)
        
        output = {'pred': logits}
        
        if self.training and 'label' in data_dict:
            output['loss'] = F.cross_entropy(logits, data_dict['label'], ignore_index=255)
            
        return output