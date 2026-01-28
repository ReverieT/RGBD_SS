import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.getcwd())

from seg_core.utils import dist_utils
from seg_core.datasets.base_dataset import RGBXDataset
from seg_core.datasets import transforms as T
from seg_core.utils.config_parser import parse_config
from seg_core.utils.metrics import Evaluator
from seg_core.models.builder import build_model

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
    model = build_model(cfg)

    model.to(device)

    # 3. 加载权重
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 处理 DDP 保存时可能带有的 'module.' 前缀
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        # 基础名称转换
        name = k[7:] if k.startswith('module.') else k
        if 'decode_head' in name:
            name = name.replace('decode_head.', 'head.')
            name = name.replace('.bn.', '.norm.')
            name = name.replace('conv_seg', 'cls_seg')
        
        # 过滤掉权重里多余的、但模型里没有的 bias
        if name in model_dict:
            new_state_dict[name] = v
        else:
            if dist_utils.is_main_process():
                print(f"  Discarding unexpected key: {name}")

    # 【核心操作】：补全那些模型里有、但权重里没有的参数 (如 ham_in.norm)
    for k, v in model_dict.items():
        if k not in new_state_dict:
            new_state_dict[k] = v
            if dist_utils.is_main_process():
                print(f"  Keeping initialized key: {k}")

    # 现在 new_state_dict 的 key 和 model 完全一致了
    model.load_state_dict(new_state_dict, strict=False)

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
