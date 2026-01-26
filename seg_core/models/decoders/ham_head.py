import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================================================================
# 1. 基础组件 (替代 mmcv/mmseg 的依赖)
# ==================================================================

class ConvModule(nn.Module):
    """
    一个标准的卷积块：Conv2d -> BatchNorm -> ReLU
    替代 mmcv.cnn.ConvModule
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """
    封装 F.interpolate，处理可能的警告
    """
    return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

# ==================================================================
# 2. 矩阵分解算法核心 (NMF2D)
# ==================================================================

class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()
        self.spatial = args.get("SPATIAL", True)
        self.S = args.get("MD_S", 1)
        self.D = args.get("MD_D", 512)
        self.R = args.get("MD_R", 64)
        self.train_steps = args.get("TRAIN_STEPS", 6)
        self.eval_steps = args.get("EVAL_STEPS", 7)
        self.inv_t = args.get("INV_T", 100)
        self.eta = args.get("ETA", 0.9)
        self.rand_init = args.get("RAND_INIT", True)

    def _build_bases(self, B, S, D, R, device=None):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, "bases"):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer("bases", bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)
        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device=None):
        bases = torch.rand((B * S, D, R), device=device)
        bases = F.normalize(bases, dim=1)
        return bases

    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)
        return coef

class Hamburger(nn.Module):
    def __init__(self, ham_channels=512, ham_kwargs=dict()):
        super().__init__()
        self.ham_in = ConvModule(ham_channels, ham_channels, 1)
        self.ham = NMF2D(ham_kwargs)
        self.ham_out = ConvModule(ham_channels, ham_channels, 1)

    def forward(self, x):
        enjoy = self.ham_in(x)
        # enjoy = F.relu(enjoy, inplace=True) # ham_in 已经包含了 ReLU
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)
        return ham

# ==================================================================
# 3. LightHamHead 主类
# ==================================================================

class LightHamHead(nn.Module):
    """
    LightHamHead from SegNeXt.
    Args:
        in_channels (list[int]): Backbone输出的通道数列表，例如 [128, 256, 512].
        channels (int): 解码头的中间维度 (ham_channels).
        num_classes (int): 分割类别数.
    """
    def __init__(self, in_channels, channels, num_classes, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        
        # 1. Squeeze: 将多层特征融合并压缩到 channels 维度
        # 输入维度是所有选中层通道数之和 (Concatenation)
        total_in_channels = sum(in_channels)
        self.squeeze = ConvModule(total_in_channels, channels, 1)

        # 2. Hamburger: 核心矩阵分解模块
        # MD_R=16 是原论文在 Cityscapes 上的默认配置
        ham_kwargs = kwargs.get('ham_kwargs', {'MD_S': 1, 'MD_D': 512, 'MD_R': 16, 'TRAIN_STEPS': 6})
        # 注意：如果 channels 不是 512，需要动态更新 MD_D
        ham_kwargs['MD_D'] = channels 
        
        self.hamburger = Hamburger(ham_channels=channels, ham_kwargs=ham_kwargs)

        # 3. Align: 对齐特征
        self.align = ConvModule(channels, channels, 1)
        
        # 4. Classifier: 最终预测层
        self.dropout = nn.Dropout(0.1)
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs):
        """
        inputs: List[Tensor], 来自 Backbone 的多层特征
        """
        # 1. 统一分辨率 (Resize to the size of the first input)
        # inputs[0] 通常是分辨率最高的特征图
        target_size = inputs[0].shape[2:]
        
        resized_inputs = []
        for x in inputs:
            if x.shape[2:] != target_size:
                x = resize(x, size=target_size, mode='bilinear', align_corners=False)
            resized_inputs.append(x)

        # 2. 拼接 (Concat)
        x = torch.cat(resized_inputs, dim=1)

        # 3. 降维 (Squeeze)
        x = self.squeeze(x)

        # 4. 全局建模 (Hamburger)
        x = self.hamburger(x)

        # 5. 对齐 (Align)
        x = self.align(x)

        # 6. 分类
        x = self.dropout(x)
        output = self.cls_seg(x)
        
        return output