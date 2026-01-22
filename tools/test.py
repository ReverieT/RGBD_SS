import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.getcwd())

from seg_core.models.backbones.dformer import DFormerv2_S, DFormerv2_B, DFormerv2_L

from seg_core.utils import dist_utils
from seg_core.datasets.base_dataset import RGBXDataset
from seg_core.datasets import transforms as T
from seg_core.models.backbones.resnet import ResNet
from seg_core.models.decoders.fcn_head import FCNHead
from seg_core.models.segmentor import RGBDSegmentor
from seg_core.utils.config_parser import parse_config
from seg_core.utils.metrics import Evaluator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()

def main():
    args = get_args()
    cfg = parse_config(args.config)

    # 1. 初始化 DDP
    dist_utils.init_distributed_mode(args)
    device = torch.device(args.gpu)

    # 2. 构建模型
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

    model.to(device)

    # 3. 加载权重
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 处理 DDP 保存时可能带有的 'module.' 前缀
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    # DDP 包装 (为了支持多卡并行推理)
    model = DDP(model, device_ids=[args.gpu])
    model.eval()

    # 4. 数据集
    val_trans = T.Compose([
        T.Normalize(mean=cfg.transforms.mean, std=cfg.transforms.std),
        T.ToTensor()
    ])
    dataset_val = RGBXDataset(cfg.dataset.root, split=cfg.dataset.val_split, transforms=val_trans)
    val_sampler = DistributedSampler(dataset_val, shuffle=False)
    val_loader = DataLoader(dataset_val, batch_size=1, sampler=val_sampler, num_workers=4, pin_memory=True)

    # 5. 评估器
    evaluator = Evaluator(cfg.dataset.n_classes)
    evaluator.reset()

    # 6. 推理循环
    if dist_utils.is_main_process():
        print("Start evaluating...")

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch['image'].to(device, non_blocking=True)
            depths = batch['depth'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            output = model({'image': images, 'depth': depths})
            preds = output['pred'].argmax(dim=1)

            # 更新混淆矩阵
            # 注意：将 GPU 上的 tensor 转回 CPU numpy
            np_labels = labels.cpu().numpy()
            np_preds = preds.cpu().numpy()
            
            # 过滤 Ignore Index (255)
            # Evaluator 内部逻辑需要 label 也在 0~num_class 范围内
            # 所以我们只传有效部分给 evaluator，或者依靠 evaluator 的 mask 逻辑
            # 我们之前在 Evaluator._generate_matrix 里写了 mask = (gt >= 0) & (gt < num_class)
            # 255 不小于 40，所以会被自动忽略，直接传进去即可
            evaluator.add_batch(np_labels, np_preds)

            if i % 10 == 0 and dist_utils.is_main_process():
                print(f"Processed {i}/{len(val_loader)}")

    # 7. 多卡汇总结果
    # 我们需要把所有卡上的 confusion_matrix 加起来
    local_cm = torch.from_numpy(evaluator.confusion_matrix).to(device)
    dist_utils.reduce_mean(local_cm, dist_utils.get_world_size()) # 注意：reduce_mean 是求平均
    # 但混淆矩阵应该是求和 (SUM)。
    # 修正：我们直接用 torch.distributed.all_reduce
    # 因为 reduce_mean 里面做了除法，我们这里不需要除法
    
    # 重新获取原始矩阵进行求和
    local_cm = torch.from_numpy(evaluator.confusion_matrix).to(device)
    torch.distributed.all_reduce(local_cm, op=torch.distributed.ReduceOp.SUM)
    
    # 更新回 evaluator
    evaluator.confusion_matrix = local_cm.cpu().numpy()

    # 8. 计算指标
    if dist_utils.is_main_process():
        acc = evaluator.pixel_accuracy()
        acc_class = evaluator.pixel_accuracy_class()
        mIoU = evaluator.mean_intersection_over_union()
        fwIoU = evaluator.frequency_weighted_intersection_over_union()

        print("\n" + "="*20 + " Test Results " + "="*20)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Pixel Accuracy:       {acc:.4f}")
        print(f"Pixel Accuracy Class: {acc_class:.4f}")
        print(f"mIoU:                 {mIoU:.4f}")
        print(f"FWIoU:                {fwIoU:.4f}")
        print("="*54)

if __name__ == '__main__':
    main()