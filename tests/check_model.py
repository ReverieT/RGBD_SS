import sys
import os
import torch

# 添加路径
sys.path.append(os.getcwd())

from seg_core.models.backbones.resnet import ResNet
from seg_core.models.decoders.fcn_head import FCNHead
from seg_core.models.segmentor import RGBDSegmentor

def main():
    print("构建模型中...")
    
    # 1. 实例化两个 Backbone (假设用 ResNet50)
    # 预训练权重会自动下载
    rgb_backbone = ResNet(depth=50, pretrained=True)
    depth_backbone = ResNet(depth=50, pretrained=True)
    
    # 2. 实例化 Head
    # ResNet50 的 Layer4 输出通道是 2048
    # 我们将其降维到 512，假设 40 类 (NYUv2)
    head = FCNHead(in_channels=2048, channels=512, num_classes=40)
    
    # 3. 组装 Segmentor
    model = RGBDSegmentor(rgb_backbone, depth_backbone, head, n_classes=40)
    
    print("✅ 模型构建成功!")
    
    # 4. 构造伪造数据进行一次前向传播 (Forward Pass)
    batch_size = 2
    dummy_rgb = torch.randn(batch_size, 3, 480, 640)
    dummy_depth = torch.randn(batch_size, 1, 480, 640)
    # 构造一个随机标签 (0-39, 偶尔混入 255)
    dummy_label = torch.randint(0, 40, (batch_size, 480, 640))
    dummy_label[0, 0, 0] = 255 
    
    data_dict = {
        'image': dummy_rgb,
        'depth': dummy_depth,
        'label': dummy_label
    }
    
    print(f"输入数据形状: RGB {dummy_rgb.shape}, Depth {dummy_depth.shape}")
    
    # 5. 运行模型
    model.train() # 开启训练模式以计算 Loss
    output = model(data_dict)
    
    loss = output['loss']
    pred = output['pred']
    
    print(f"✅ 前向传播成功!")
    print(f"   输出预测形状: {pred.shape} (预期: [{batch_size}, 40, 480, 640])")
    print(f"   计算损失值: {loss.item()}")

if __name__ == '__main__':
    main()