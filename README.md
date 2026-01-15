# RGB-D Semantic Segmentation Framework (PyTorch)

这是一个基于 PyTorch 实现的模块化 RGB-D（RGB + 深度信息）语义分割代码库。旨在提供灵活的模型设计、高效的**多卡分布式训练 (DDP)** 支持，以及针对 **NYUDepthv2** 和 **SUNRGBD** 数据集的标准处理流程。

## 🌟 主要特性 (Features)

* **多模态支持**: 专为 RGB-D 数据设计，内置多种模态融合策略（Early/Middle/Late Fusion）。
* **分布式训练**: 原生支持 PyTorch `DistributedDataParallel` (DDP)，支持多机多卡训练。
* **配置驱动**: 所有实验参数通过 YAML 配置文件管理，通过继承机制减少冗余。
* **模块化设计**: 骨干网络 (Backbone)、解码头 (Decoder)、融合模块 (Fusion) 高度解耦，易于二次开发。
* **完备流程**: 包含数据增强（RGB-D 同步变换）、训练、验证、推理及可视化。

## 📂 代码结构 (File Structure)

本项目的目录组织如下，采用配置与代码分离的设计模式：

```text
RGBD-Segmentation/
├── configs/                     # [配置中心] 存放所有 YAML 实验配置文件
│   ├── _base_/                  # 基础配置 (数据集路径、默认数据增强、Runtime参数)
│   ├── nyu_v2/                  # 针对 NYUDepthv2 的实验配置 (如: deeplabv3_resnet50.yaml)
│   └── sunrgbd/                 # 针对 SUNRGBD 的实验配置 (如: segformer_b2.yaml)
│
├── data/                        # [数据目录] (建议软链接到实际数据存储位置)
│   ├── NYUDepthv2/              # NYUDepthv2 原始数据及处理后的数据
│   └── SUNRGBD/                 # SUNRGBD 原始数据
│
├── outputs/                     # [实验输出] 训练过程中自动生成
│   └── {experiment_name}/       # 按实验名称自动创建文件夹
│       ├── checkpoints/         # 保存的模型权重 (.pth)
│       ├── logs/                # Tensorboard/Wandb 日志文件
│       └── visual_results/      # 验证集预测结果可视化
│
├── scripts/                     # [Shell 脚本] 快捷启动脚本
│   ├── train_dist.sh            # 启动分布式训练 (使用 torchrun)
│   └── test.sh                  # 启动测试/评估
│
├── seg_core/                    # [核心代码库] 
│   ├── datasets/                # 数据集与数据加载
│   │   ├── base_dataset.py      # 数据集基类
│   │   ├── nyu.py               # NYU 数据集读取逻辑
│   │   ├── sunrgbd.py           # SUNRGBD 数据集读取逻辑
│   │   ├── transforms.py        # ★ RGB-D 同步数据增强 (几何变换需同步，光照仅RGB)
│   │   └── loader.py            # DataLoader 构建器 (含 DistributedSampler)
│   │
│   ├── models/                  # 模型定义
│   │   ├── backbones/           # 骨干网络 (ResNet, Swin, MixTransformer 等)
│   │   ├── decoders/            # 分割解码头 (ASPP, MLP Decoder 等)
│   │   ├── fusion/              # ★ RGB-D 融合模块 (Add, Concat, Attention, SE-Block)
│   │   └── segmentor.py         # 模型组装器 (Encoder-Decoder Wrapper)
│   │
│   ├── losses/                  # 损失函数 (CrossEntropy, Dice, OHEM 等)
│   └── utils/                   # 通用工具
│       ├── dist_utils.py        # 分布式训练通信工具 (all_reduce, rank获取等)
│       ├── logger.py            # 日志记录系统
│       ├── metrics.py           # 评价指标计算 (mIoU, Pixel Acc)
│       └── visualizer.py        # 预测结果上色与融合显示
│
├── tools/                       # [Python 入口]
│   ├── train.py                 # 训练主入口
│   ├── test.py                  # 测试主入口
│   └── inference.py             # 单图推理脚本
│
├── requirements.txt             # Python 依赖包列表
└── README.md                    # 项目说明文档