import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

# 添加路径
sys.path.append(os.getcwd())

from seg_core.models.backbones.dformer import DFormerv2_S, DFormerv2_B, DFormerv2_L

from seg_core.models.backbones.resnet import ResNet
from seg_core.models.decoders.fcn_head import FCNHead
from seg_core.models.segmentor import RGBDSegmentor
from seg_core.utils.config_parser import parse_config

try:
    from thop import profile, clever_format
except ImportError:
    print("❌ Error: 'thop' library is not installed.")
    print("Please install it via: pip install thop")
    sys.exit(1)

# ==========================================
# 辅助 Wrapper
# Thop 只能传递 Tensor 参数，而我们的模型需要 Dict
# ==========================================
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, rgb, depth):
        # 将 Tensor 重新打包成字典
        return self.model({'image': rgb, 'depth': depth})

def get_args():
    parser = argparse.ArgumentParser(description="Benchmark FPS, Params, and FLOPs")
    parser.add_argument("--config", required=True, help="Config file path")
    # 默认使用 Config 里的 crop_size，也可以手动指定
    parser.add_argument("--height", type=int, default=None, help="Inference image height")
    parser.add_argument("--width", type=int, default=None, help="Inference image width")
    return parser.parse_args()

def main():
    args = get_args()
    cfg = parse_config(args.config)
    
    # 1. 确定输入尺寸
    # 如果命令行没指定，就用 Config 里的 crop_size (例如 480x480)
    # 注意：ResNet 等模型通常要求输入是 32 的倍数
    h = args.height if args.height else cfg.transforms.crop_size
    w = args.width if args.width else cfg.transforms.crop_size
    
    print(f"🚀 Starting Benchmark...")
    print(f"   Config: {args.config}")
    print(f"   Input Size: ({h}, {w})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. 构建模型
    if 'dformer' in cfg.model.backbone:
        # === 实例化 DFormer ===
        if cfg.model.backbone == 'dformerv2_s':
            backbone = DFormerv2_S(pretrained=cfg.model.pretrained)
            dec_channels = 512
        elif cfg.model.backbone == 'dformerv2_b':
            backbone = DFormerv2_B(pretrained=cfg.model.pretrained)
            dec_channels = 512
        # ... 其他变体
        head = FCNHead(in_channels=512, channels=cfg.model.decoder_channels, num_classes=cfg.dataset.n_classes)
        # ★ 关键：只传一个 backbone，Segmentor 会自动识别 is_unified=True
        model = RGBDSegmentor(backbone, head=head, n_classes=cfg.dataset.n_classes)
    else:
        # === 实例化 ResNet ===
        rgb_backbone = ResNet(depth=50, pretrained=cfg.model.pretrained)
        depth_backbone = ResNet(depth=50, pretrained=cfg.model.pretrained)
        head = FCNHead(in_channels=2048, channels=cfg.model.decoder_channels, num_classes=cfg.dataset.n_classes)
        model = RGBDSegmentor(rgb_backbone, depth_backbone, head, cfg.dataset.n_classes)
    
    real_model = model
    real_model.eval()
    real_model.to(device)
    
    # 3. 统计参数量 (Params)
    # 过滤掉不需要梯度的参数（如果有冻结层的话）
    trainable_params = sum(p.numel() for p in real_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in real_model.parameters())
    
    print("\n" + "="*30)
    print(f"📊 Model Parameters")
    print(f"   Total Params:     {total_params / 1e6:.2f} M")
    print(f"   Trainable Params: {trainable_params / 1e6:.2f} M")

    # 4. 统计 FLOPs (计算复杂度)
    # 构造 Dummy Input
    dummy_rgb = torch.randn(1, 3, h, w).to(device)
    dummy_depth = torch.randn(1, 1, h, w).to(device)
    
    # 使用 Wrapper 适配 thop
    wrapped_model = ModelWrapper(real_model)
    
    try:
        flops, params = profile(wrapped_model, inputs=(dummy_rgb, dummy_depth), verbose=False)
        flops_readable, params_readable = clever_format([flops, params], "%.3f")
        print(f"   FLOPs (G):        {flops / 1e9:.3f} G")
        # print(f"   (Thop format: {flops_readable})")
    except Exception as e:
        print(f"   FLOPs Calculation Failed: {e}")

    # 5. 测速 (FPS)
    print("\n" + "="*30)
    print(f"⏱️  Measuring FPS (Batch Size = 1)...")
    
    # 预热 (Warm up) - 让 GPU 进入工作状态
    print("   Warming up GPU...")
    with torch.no_grad():
        for _ in range(50):
            _ = real_model({'image': dummy_rgb, 'depth': dummy_depth})
    
    # 正式计时
    iterations = 200
    print(f"   Running {iterations} iterations...")
    
    # 使用 torch.cuda.Event 进行精确计时
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    timings = []
    
    with torch.no_grad():
        for _ in range(iterations):
            starter.record()
            _ = real_model({'image': dummy_rgb, 'depth': dummy_depth})
            ender.record()
            
            # 等待 GPU 完成
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # 毫秒
            timings.append(curr_time)
            
    mean_time_ms = np.mean(timings)
    std_time_ms = np.std(timings)
    fps = 1000 / mean_time_ms
    
    print(f"   Latency: {mean_time_ms:.2f} ms ± {std_time_ms:.2f} ms")
    print(f"   FPS:     {fps:.2f}")
    print("="*30 + "\n")

if __name__ == '__main__':
    main()