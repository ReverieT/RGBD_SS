import os
import yaml
from easydict import EasyDict as edict

def load_yaml(file_path):
    """读取单个 YAML 文件"""
    with open(file_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def merge_config(base_cfg, override_cfg):
    """
    递归合并两个字典。
    override_cfg 中的值会覆盖 base_cfg。
    """
    for k, v in override_cfg.items():
        if isinstance(v, dict) and k in base_cfg:
            # 如果两个都是字典，递归合并
            merge_config(base_cfg[k], v)
        else:
            # 否则直接覆盖
            base_cfg[k] = v
    return base_cfg

def parse_config(config_path):
    """
    主解析函数，支持 _base_ 继承机制。
    """
    # 1. 确保路径存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 2. 读取当前文件
    cfg = load_yaml(config_path)
    
    # 3. 处理继承逻辑
    if '_base_' in cfg:
        base_files = cfg['_base_']
        # 允许 _base_ 是单个字符串或列表
        if isinstance(base_files, str):
            base_files = [base_files]
            
        final_cfg = {}
        
        # 获取当前 config 文件的目录，用于解析相对路径
        current_dir = os.path.dirname(config_path)
        
        # 依次加载 base 文件
        for base_file in base_files:
            # 处理相对路径
            base_path = os.path.join(current_dir, base_file)
            base_path = os.path.abspath(base_path)
            
            # 递归调用 parse_config (支持多级继承)
            base_cfg = parse_config(base_path)
            
            # 合并到 final_cfg
            merge_config(final_cfg, base_cfg)
            
        # 4. 最后将当前文件的内容合并进去（优先级最高）
        # 删除 _base_ 字段，不需要它进入最终配置
        del cfg['_base_']
        merge_config(final_cfg, cfg)
        
        # 转换为 EasyDict，这样可以用 cfg.dataset.name 访问，而不用 cfg['dataset']['name']
        return edict(final_cfg)
    
    else:
        # 如果没有 _base_，直接返回
        return edict(cfg)

# 测试代码
if __name__ == '__main__':
    # 假设你已经创建了上述 yaml 文件
    cfg = parse_config('configs/nyu_v2/resnet50_baseline.yaml')
    print(cfg)
    print(f"Dataset: {cfg.dataset.name}")     # 来自 base
    print(f"Lr: {cfg.optimizer.lr}")          # 来自 base
    print(f"Backbone: {cfg.model.backbone}")  # 来自 current
    print(f"Batch Size: {cfg.loader.batch_size}") # 来自 override (应该输出 8)