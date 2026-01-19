import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet101_Weights

class ResNet(nn.Module):
    """
    用于语义分割的 ResNet 骨干网络封装。
    特点：去掉最后的 FC 层，返回多尺度特征列表。
    """
    def __init__(self, depth=50, pretrained=True):
        super().__init__()
        
        # 1. 根据深度选择权重和模型结构
        if depth == 50:
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet50(weights=weights)
        elif depth == 101:
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet101(weights=weights)
        elif depth == 18:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
        else:
            raise NotImplementedError(f"ResNet depth {depth} not implemented.")

        # 2. 提取层结构
        # Stem (主干部分): 7x7 Conv -> BN -> ReLU -> MaxPool
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        
        # Encoder Layers (残差块)
        self.layer1 = backbone.layer1 # 1/4 分辨率
        self.layer2 = backbone.layer2 # 1/8 分辨率
        self.layer3 = backbone.layer3 # 1/16 分辨率
        self.layer4 = backbone.layer4 # 1/32 分辨率

        # 记录特征通道数，方便 Decoder 使用
        if depth >= 50:
            self.channels = [256, 512, 1024, 2048] # ResNet50/101 的通道数
        else:
            self.channels = [64, 128, 256, 512]    # ResNet18/34 的通道数

    def forward(self, x):
        """
        Args:
            x: Tensor (B, 3, H, W)
        Returns:
            features: List[Tensor] containing [c1, c2, c3, c4]
        """
        x = self.stem(x)
        
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        return [c1, c2, c3, c4]

if __name__ == '__main__':
    # 简单的测试代码，确保跑得通
    model = ResNet(depth=50)
    dummy_input = torch.randn(2, 3, 480, 640)
    feats = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print("Output features:")
    for i, f in enumerate(feats):
        print(f"  Stage {i+1}: {f.shape}")
    
    # 预期输出 (对于 ResNet50):
    # Stage 1: [2, 256, 120, 160] (1/4)
    # Stage 2: [2, 512, 60, 80]   (1/8)
    # Stage 3: [2, 1024, 30, 40]  (1/16)
    # Stage 4: [2, 2048, 15, 20]  (1/32)