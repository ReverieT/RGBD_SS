import os
import torch
import torch.distributed as dist
import logging

def init_distributed_mode(args):
    """
    初始化 DDP 环境。
    支持 torchrun (推荐) 和 SLURM 等方式。
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun 启动方式
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM 启动方式
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    
    print(f'| distributed init (rank {args.rank}): env://', flush=True)
    
    # 初始化进程组
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        init_method='env://',
        world_size=args.world_size, 
        rank=args.rank
    )
    torch.distributed.barrier() # 等待所有进程启动

def cleanup():
    """销毁进程组"""
    dist.destroy_process_group()

def is_main_process():
    """判断是否为主进程 (Rank 0)"""
    return get_rank() == 0

def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_mean(tensor, nprocs):
    """
    将所有显卡上的 tensor 求平均。
    用于计算 Loss 和 Accuracy 的全局平均值。
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def setup_logger(name, save_dir, rank, filename="train.log"):
    """
    设置 Logger，只有 Rank 0 会写入文件和打印到控制台。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 格式
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S'
    )

    # 只有主进程添加 handler
    if rank == 0:
        # 控制台 Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # 文件 Handler
        if save_dir:
            fh = logging.FileHandler(os.path.join(save_dir, filename), mode='w')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    
    return logger