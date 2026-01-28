import argparse
import os
import sys
import torch

sys.path.append(os.getcwd())

from seg_core.utils.config_parser import parse_config
from seg_core.models.builder import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Diagnose model vs checkpoint keys')
    parser.add_argument('--config', required=True, help='train config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file path')
    return parser.parse_args()


def analyze_keys(model_keys, ckpt_keys):
    """分析两组键名的匹配情况"""
    # 清理权重键名（去除 module. 前缀）
    cleaned_ckpt_keys = [k[7:] if k.startswith('module.') else k for k in ckpt_keys]
    ckpt_set = set(cleaned_ckpt_keys)
    model_set = set(model_keys)
    
    # 集合操作
    common = model_set & ckpt_set
    only_in_model = model_set - ckpt_set
    only_in_ckpt = ckpt_set - model_set
    
    return {
        'common': sorted(common),
        'only_in_model': sorted(only_in_model),
        'only_in_ckpt': sorted(only_in_ckpt),
        'coverage': len(common) / len(model_set) if model_set else 0
    }


def simulate_mapping(ckpt_keys, rules):
    """模拟应用键名映射规则后的效果"""
    mapped = {}
    for k in ckpt_keys:
        original = k[7:] if k.startswith('module.') else k
        mapped_name = original
        for old, new in rules:
            mapped_name = mapped_name.replace(old, new)
        if mapped_name != original:
            mapped[original] = mapped_name
    return mapped


def main():
    args = parse_args()
    
    # 解析配置并构建模型
    cfg = parse_config(args.config)
    model = build_model(cfg)
    model.eval()
    
    # 获取模型状态字典
    model_keys = list(model.state_dict().keys())
    
    # 加载权重
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    ckpt_keys = list(state_dict.keys())
    
    print("\n" + "="*70)
    print("🔍 模型与权重键名诊断报告")
    print("="*70)
    
    # 基础统计
    print(f"\n📊 基础统计:")
    print(f"   模型定义参数数量: {len(model_keys)}")
    print(f"   权重文件参数数量: {len(ckpt_keys)}")
    
    # 原始匹配分析（仅去除 module. 前缀）
    result = analyze_keys(model_keys, ckpt_keys)
    print(f"\n📋 直接匹配情况（仅去除 'module.' 前缀）:")
    print(f"   ✓ 匹配的键: {len(result['common'])} ({result['coverage']*100:.1f}%)")
    print(f"   ✗ 仅权重中有: {len(result['only_in_ckpt'])}")
    print(f"   ⚠ 仅模型中有: {len(result['only_in_model'])}")
    
    # 详细列出仅存在于权重中的键（这些会被忽略）
    if result['only_in_ckpt']:
        print(f"\n🔴 【关键】仅存在于权重文件中（加载时将被丢弃）:")
        for k in result['only_in_ckpt'][:20]:  # 限制显示数量
            print(f"   - {k}")
        if len(result['only_in_ckpt']) > 20:
            print(f"   ... 还有 {len(result['only_in_ckpt'])-20} 个")
    
    # 详细列出仅存在于模型中的键（这些保持随机初始化）
    if result['only_in_model']:
        print(f"\n🟡 【关键】仅存在于模型定义中（将使用初始化值）:")
        for k in result['only_in_model'][:20]:
            print(f"   - {k}")
        if len(result['only_in_model']) > 20:
            print(f"   ... 还有 {len(result['only_in_model'])-20} 个")
    
    # 模拟单流模型的映射规则
    print("\n" + "-"*70)
    print("🧪 模拟单流模型键名映射（decode_head→head, bn→norm, conv_seg→cls_seg）:")
    
    mapping_rules = [
        ('decode_head.', 'head.'),
        ('.bn.', '.norm.'),
        ('conv_seg', 'cls_seg')
    ]
    
    simulated = simulate_mapping(ckpt_keys, mapping_rules)
    
    # 应用模拟映射后的匹配情况
    mapped_ckpt_keys = []
    for k in ckpt_keys:
        clean = k[7:] if k.startswith('module.') else k
        for old, new in mapping_rules:
            clean = clean.replace(old, new)
        mapped_ckpt_keys.append(clean)
    
    mapped_result = analyze_keys(model_keys, mapped_ckpt_keys)
    print(f"   应用映射后匹配率: {mapped_result['coverage']*100:.1f}%")
    print(f"   改善程度: {(mapped_result['coverage']-result['coverage'])*100:.1f}%")
    
    if simulated:
        print(f"\n   映射示例（全部）:")
        # for old, new in list(simulated.items())[:20]:
        for old, new in list(simulated.items()):
            status = "✓ 匹配成功" if new in model_keys else "✗ 仍不匹配"
            print(f"   {old:<50} → {new:<30} {status}")
    
    # 架构类型建议
    print("\n" + "="*70)
    print("💡 诊断建议:")
    
    # 检测双流特征
    has_encode_rgb = any('encode_rgb' in k for k in ckpt_keys)
    has_encode_depth = any('encode_depth' in k for k in ckpt_keys)
    has_dual_decode = any('decode_head_rgb' in k or 'decode_head_depth' in k for k in ckpt_keys)
    
    if has_encode_rgb or has_encode_depth or has_dual_decode:
        print("   检测到双流结构特征（encode_rgb/depth 或 decode_head_rgb/depth）")
        print("   建议: 使用严格模式（仅去除 module. 前缀），不要做 decode_head→head 替换")
    else:
        print("   检测到单流结构特征")
        if 'decode_head' in str(result['only_in_ckpt']) and 'head.' in str(result['only_in_model']):
            print("   建议: 需要应用 decode_head→head 替换")
        elif result['coverage'] < 0.9:
            print(f"   警告: 当前匹配率仅 {result['coverage']*100:.1f}%，建议检查键名映射规则")
        else:
            print(f"   匹配率良好 ({result['coverage']*100:.1f}%)，可直接加载")
    
    print("="*70)


if __name__ == '__main__':
    main()