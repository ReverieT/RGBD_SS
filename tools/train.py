import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from seg_core.models.backbones.dformer import DFormerv2_S, DFormerv2_B, DFormerv2_L
# 添加项目根目录到 Path
sys.path.append(os.getcwd())

# 导入我们自己写的模块
from seg_core.utils import dist_utils
from seg_core.datasets.base_dataset import RGBXDataset
from seg_core.datasets import transforms as T
from seg_core.models.backbones.resnet import ResNet
from seg_core.models.decoders.fcn_head import FCNHead
from seg_core.models.segmentor import RGBDSegmentor
from seg_core.utils.config_parser import parse_config


def get_args_parser():
    parser = argparse.ArgumentParser(description="RGB-D Segmentation Training")
    parser.add_argument("--config", default="configs/nyu.yaml", help="config file path")
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for torch.distributed.launch")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    return parser

# ========================
# 1. 训练一个 Epoch (Function)
# ========================
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    logger = logging.getLogger("RGBD_Train")
    
    # 用于记录平均 Loss
    mean_loss = torch.zeros(1).to(device)
    
    for i, batch in enumerate(data_loader):
        start_time = time.time()
        
        # 1. 数据搬运
        images = batch['image'].to(device, non_blocking=True)
        depths = batch['depth'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        # 2. 混合精度前向传播 (AMP)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            output = model({'image': images, 'depth': depths, 'label': labels})
            loss = output['loss']

        # 3. 反向传播
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 4. 日志记录 (同步多卡 Loss)
        loss_value = loss.item()
        # 这里为了速度，不是每个 step 都做 all_reduce，只在打印时做
        # 但为了统计准确，我们简单地累加
        mean_loss = (mean_loss * i + loss_value) / (i + 1)

        if i % print_freq == 0:
            # 同步所有显卡的 Loss 以便打印
            reduced_loss = dist_utils.reduce_mean(torch.tensor(loss_value).to(device), dist_utils.get_world_size())
            
            if dist_utils.is_main_process():
                logger.info(
                    f"Epoch: [{epoch}][{i}/{len(data_loader)}] "
                    f"Lr: {optimizer.param_groups[0]['lr']:.6f} "
                    f"Loss: {reduced_loss.item():.4f} "
                    f"Time: {time.time() - start_time:.3f}s"
                )

    return mean_loss

# ========================
# 2. 验证 (Function)
# ========================
def validate(model, data_loader, device, n_classes):
    model.eval()
    logger = logging.getLogger("RGBD_Train")
    
    total_correct = 0
    total_label = 0
    
    # 注意：这里只计算 Pixel Accuracy 作为简单验证
    # 完整的 mIoU 计算通常需要 Confusion Matrix，比较占篇幅，后续可添加
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images = batch['image'].to(device, non_blocking=True)
            depths = batch['depth'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            output = model({'image': images, 'depth': depths})
            preds = output['pred'] # (B, C, H, W)
            
            # 简单计算 Argmax
            preds = torch.argmax(preds, dim=1) # (B, H, W)
            
            # 过滤掉 Ignore Index (255)
            mask = labels != 255
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_label += mask.sum().item()

    # 多卡汇总
    total_correct = torch.tensor(total_correct).float().to(device)
    total_label = torch.tensor(total_label).float().to(device)
    
    dist_utils.reduce_mean(total_correct, dist_utils.get_world_size())
    dist_utils.reduce_mean(total_label, dist_utils.get_world_size())
    
    acc = total_correct / (total_label + 1e-6)
    
    if dist_utils.is_main_process():
        logger.info(f"Validation Pixel Acc: {acc.item():.4f}")
        
    return acc.item()

# ========================
# 3. 主程序
# ========================
import logging # Re-import to fix scope issues if any

def main():
    args = get_args_parser().parse_args()
    cfg = parse_config(args.config) # 从 YAML 读取配置

    # 1. 初始化分布式环境
    dist_utils.init_distributed_mode(args)
    device = torch.device(args.gpu)

    # 2. 设置 Logger (只在主进程)
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir, exist_ok=True)
    logger = dist_utils.setup_logger("RGBD_Train", cfg.save_dir, args.rank)
    logger.info(f"Git commit: ... (Optional)")
    logger.info(f"Args: {args}") 
    logger.info(f"Config:\n{cfg}")

    # 3. 构建 Dataset & DataLoader
    # 定义 Transforms
    train_trans = T.Compose([
        T.RandomScaleCrop(cfg.transforms.base_size, cfg.transforms.crop_size),
        T.RandomHorizontalFlip(),
        T.Normalize(),
        T.ToTensor()
    ])
    val_trans = T.Compose([
        T.Normalize(),
        T.ToTensor()
    ])

    # 实例化 Dataset
    dataset_train = RGBXDataset(cfg.dataset.root, split='train', transforms=train_trans)
    dataset_val = RGBXDataset(cfg.dataset.root, split='test', transforms=val_trans)

    # 实例化 Sampler (DDP 必须)
    train_sampler = DistributedSampler(dataset_train, shuffle=True)
    val_sampler = DistributedSampler(dataset_val, shuffle=False)

    train_loader = DataLoader(
        dataset_train, batch_size=cfg.loader.batch_size, 
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        dataset_val, batch_size=1, # 验证时通常 batch_size=1
        sampler=val_sampler, num_workers=4, pin_memory=True
    )

    # 4. 构建模型
    logger.info(f"Building model with backbone: {cfg.model.backbone}")

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

    # 转换 SyncBatchNorm (多卡训练必备，提升小 Batch 下的性能)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 包装 DDP
    model_without_ddp = model
    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False) # find_unused=True 仅在调试时开启

    # 5. Optimizer & Loss
    params = [p for p in model.parameters() if p.requires_grad]
    # === 新增：根据 Config 选择优化器 ===
    if cfg.optimizer.type == 'AdamW':
        optimizer = optim.AdamW(
            params, 
            lr=cfg.optimizer.lr, 
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        # 默认为 SGD
        optimizer = optim.SGD(
            params, 
            lr=cfg.optimizer.lr, 
            momentum=cfg.optimizer.momentum, 
            weight_decay=cfg.optimizer.weight_decay
        )
    
    # 简单的学习率衰减
    lr_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=cfg.epochs, power=0.9)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scaler = torch.cuda.amp.GradScaler() # 混合精度

    # 6. 开始循环
    logger.info("Start training...")
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        # DDP 重要步骤：设置 epoch 使得 shuffle 生效
        train_sampler.set_epoch(epoch)
        
        # Train
        train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, cfg.print_freq, scaler)
        lr_scheduler.step()
        
        # Val
        if epoch % 1 == 0: # 每个 epoch 都验证
            acc = validate(model, val_loader, device, cfg.dataset.n_classes)
            
            # Save Checkpoint (只在主进程)
            if dist_utils.is_main_process():
                save_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args
                }
                # 保存最新
                torch.save(save_dict, os.path.join(cfg.save_dir, 'checkpoint_last.pth'))
                # 保存最佳
                if acc > best_acc:
                    best_acc = acc
                    torch.save(save_dict, os.path.join(cfg.save_dir, 'checkpoint_best.pth'))
                    logger.info(f"New best checkpoint saved with Acc: {best_acc:.4f}")

    total_time = time.time() - start_time
    logger.info(f"Training finished. Total time: {total_time:.1f}s")
    dist_utils.cleanup()

if __name__ == '__main__':
    main()
