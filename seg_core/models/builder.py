import torch
import torch.nn as nn
import logging
from seg_core.models.segmentor import RGBDSegmentor

# 定义常用 Backbone 的输出通道数字典
BACKBONE_CHANNELS = {
    'resnet50': [256, 512, 1024, 2048],
    'resnet101': [256, 512, 1024, 2048],
    'dformerv2_s': [64, 128, 256, 512],
    'dformerv2_b': [80, 160, 320, 512],
    'dformerv2_l': [112, 224, 448, 640],
}

def build_model(cfg):
    logger = logging.getLogger("RGBD_Train")
    
    # =================================================
    # 1. 构建 Backbone (保持不变)
    # =================================================
    backbone_name = cfg.model.backbone
    logger.info(f"Building Backbone: {backbone_name}")
    
    rgb_backbone = None
    depth_backbone = None
    feature_channels = None 

    if 'dformer' in backbone_name:
        from seg_core.models.backbones.dformer import DFormerv2_S, DFormerv2_B, DFormerv2_L
        if backbone_name == 'dformerv2_s':
            rgb_backbone = DFormerv2_S(pretrained=cfg.model.pretrained)
        elif backbone_name == 'dformerv2_b':
            rgb_backbone = DFormerv2_B(pretrained=cfg.model.pretrained)
        elif backbone_name == 'dformerv2_l':
            rgb_backbone = DFormerv2_L(pretrained=cfg.model.pretrained)
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented.")
        depth_backbone = None
        feature_channels = BACKBONE_CHANNELS[backbone_name]

    elif 'resnet' in backbone_name:
        from seg_core.models.backbones.resnet import ResNet
        depth = 50 if '50' in backbone_name else 101
        rgb_backbone = ResNet(depth=depth, pretrained=cfg.model.pretrained)
        depth_backbone = ResNet(depth=depth, pretrained=cfg.model.pretrained)
        feature_channels = BACKBONE_CHANNELS[f'resnet{depth}']

    else:
        raise NotImplementedError(f"Backbone {backbone_name} not supported yet.")

    # =================================================
    # 2. 构建 Decoder Head (新增 Ham 支持)
    # =================================================
    decoder_name = cfg.model.decoder
    logger.info(f"Building Decoder: {decoder_name}")
    
    head = None
    
    if decoder_name == 'fcn':
        from seg_core.models.decoders.fcn_head import FCNHead
        head = FCNHead(
            in_channels=feature_channels[-1], 
            channels=cfg.model.decoder_channels, 
            num_classes=cfg.dataset.n_classes
        )

    # ★★★ 新增：Ham Decoder 支持 ★★★
    elif decoder_name == 'ham':
        from seg_core.models.decoders.ham_head import LightHamHead
        
        # HamHead 需要融合 Backbone 的最后三个 Stage 
        # (例如 ResNet 的 layer2,3,4 或 DFormer 的 stage2,3,4)
        # 对应的索引通常是 [1, 2, 3] (0是第一层)
        selected_indices = [1, 2, 3]
        
        # 安全检查：防止 Backbone 层数不够 (虽然 ResNet/DFormer 都是4层)
        if len(feature_channels) < 4:
            logger.warning(f"Backbone only has {len(feature_channels)} stages, using all for HamHead.")
            selected_indices = list(range(len(feature_channels)))
        
        # 自动提取这几层的通道数，例如 [128, 256, 512]
        in_channels_list = [feature_channels[i] for i in selected_indices]
        
        head = LightHamHead(
            in_channels=in_channels_list,
            channels=cfg.model.decoder_channels, # 推荐 256 或 512
            num_classes=cfg.dataset.n_classes
        )
        
        # ★ 关键注入：告诉 Segmentor 这个 Head 需要哪几层特征
        head.input_indices = selected_indices

    elif decoder_name == 'MLPDecoder':
        raise NotImplementedError("MLPDecoder code needs to be added.")

    elif decoder_name == 'UPerNet':
        raise NotImplementedError("UPerNet code needs to be added.")

    else:
        raise NotImplementedError(f"Decoder {decoder_name} not supported.")

    # =================================================
    # 3. 组装 Segmentor
    # =================================================
    model = RGBDSegmentor(
        rgb_backbone=rgb_backbone, 
        depth_backbone=depth_backbone, 
        head=head, 
        n_classes=cfg.dataset.n_classes
    )
    
    return model