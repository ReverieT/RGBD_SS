import torch
import torch.nn as nn

class FCNHead(nn.Module):
    """
    简单的全卷积解码头 (Fully Convolutional Head).
    结构: Conv(3x3) -> BN -> ReLU -> Dropout -> Conv(1x1) -> Output
    """
    def __init__(self, in_channels, channels, num_classes, dropout_ratio=0.1):
        """
        Args:
            in_channels (int): 输入特征的通道数 (例如 ResNet50 的 Layer4 是 2048)
            channels (int): 中间层的通道数 (通常设为 input 的 1/4，如 512)
            num_classes (int): 类别数 (NYUv2=40, SUNRGBD=37)
            dropout_ratio (float): Dropout 比率，防止过拟合
        """
        super().__init__()
        
        # 1. 特征转换模块 (3x3 卷积减少通道数并融合上下文)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Dropout 层
        self.dropout = nn.Dropout2d(dropout_ratio)
        
        # 3. 最终分类层 (1x1 卷积将通道数变为类别数)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        """
        Args:
            inputs: Backbone 输出的特征列表 [c1, c2, c3, c4] 
                    或者 已经是融合后的单个 Tensor
        """
        # 如果输入是列表(来自ResNet)，我们取最后一层 (C4)
        # 因为 FCN 通常只处理最高语义层级的特征
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            x = inputs[-1]
        else:
            x = inputs

        # 1. 卷积 + BN + ReLU
        x = self.conv_block(x)
        
        # 2. Dropout
        x = self.dropout(x)
        
        # 3. 映射到类别 (Output: B, NumClasses, H/32, W/32)
        output = self.conv_seg(x)
        
        return output