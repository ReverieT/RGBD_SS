import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from typing import Tuple
from collections import OrderedDict
import math

# ==========================================
# 辅助模块 (保持原样，移除无关引用)
# ==========================================
class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.SyncBatchNorm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.SyncBatchNorm(embed_dim),
        )
    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        return x

class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out_dim, 3, 2, 1)
        self.norm = nn.SyncBatchNorm(out_dim)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.reduction(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        return x

def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)

class GeoPriorGen(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
    
    def generate_depth_decay(self, H, W, depth_grid):
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H, W):
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w], indexing='ij') # Fixed indexing warning
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask
    
    def generate_1d_depth_decay(self, H, W, depth_grid):
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        return mask

    def generate_1d_decay(self, l):
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, HW_tuple, depth_map, split_or_not=False):
        depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)
        if split_or_not:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)
            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])
            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w
            geo_prior = ((sin, cos), (mask_h, mask_w))
        else:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(HW_tuple[0], HW_tuple[1], -1)
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])
            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_d
            geo_prior = ((sin, cos), mask)
        return geo_prior

class Decomposed_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)

    def forward(self, x, rel_pos, split_or_not=False):
        bsz, h, w, _ = x.size()
        (sin, cos), (mask_h, mask_w) = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)
        qr_w = qr.transpose(1, 2)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2) + mask_w.transpose(1, 2)
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)
        qr_h = qr.permute(0, 3, 1, 2, 4)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2) + mask_h.transpose(1, 2)
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) + lepe
        return self.out_proj(output)

class Full_GSA(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)

    def forward(self, x, rel_pos, split_or_not=False):
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)
        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos).flatten(2, 3)
        kr = angle_transform(k, sin, cos).flatten(2, 3)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4).flatten(2, 3)
        qk_mat = torch.softmax(qr @ kr.transpose(-1, -2) + mask, -1)
        output = torch.matmul(qk_mat, vr).transpose(1, 2).reshape(bsz, h, w, -1) + lepe
        return self.out_proj(output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, activation_fn=F.gelu, dropout=0.0, activation_dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) # DWConv 作用在高维(ffn_dim)上
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout_module = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x 的形状: (B, H, W, embed_dim) -> 例如 64通道
        
        # 1. 第一层投影 + 激活 (升维到 ffn_dim，例如 256通道)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        
        # 2. ★关键修正★：在这里记录残差 (此时 x 是 256通道)
        # DFormer 的设计是：在 FFN 内部的高维空间里做残差
        residual = x
        
        # 3. 深度卷积
        x = self.dwconv(x)
        
        # 4. 相加 (256 + 256，维度匹配！)
        x = x + residual
        
        # 5. 第二层投影 (降维回 embed_dim，例如 64通道)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

class RGBD_Block(nn.Module):
    def __init__(self, split_or_not, embed_dim, num_heads, ffn_dim, drop_path=0.0, layerscale=False, layer_init_values=1e-5, init_value=2, heads_range=4):
        super().__init__()
        self.layerscale = layerscale
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        if split_or_not:
            self.Attention = Decomposed_GSA(embed_dim, num_heads)
        else:
            self.Attention = Full_GSA(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.cnn_pos_encode = DWConv2d(embed_dim, 3, 1, 1)
        self.Geo = GeoPriorGen(embed_dim, num_heads, init_value, heads_range)
        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)

    def forward(self, x, x_e, split_or_not=False):
        x = x + self.cnn_pos_encode(x)
        b, h, w, d = x.size()
        geo_prior = self.Geo((h, w), x_e, split_or_not=split_or_not)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.Attention(self.layer_norm1(x), geo_prior, split_or_not))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.layer_norm2(x)))
        else:
            x = x + self.drop_path(self.Attention(self.layer_norm1(x), geo_prior, split_or_not))
            x = x + self.drop_path(self.ffn(self.layer_norm2(x)))
        return x

class BasicLayer(nn.Module):
    def __init__(self, embed_dim, out_dim, depth, num_heads, init_value, heads_range, ffn_dim, drop_path, norm_layer, split_or_not, downsample, use_checkpoint, layerscale, layer_init_values):
        super().__init__()
        self.split_or_not = split_or_not
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            RGBD_Block(split_or_not, embed_dim, num_heads, ffn_dim, drop_path[i] if isinstance(drop_path, list) else drop_path, layerscale, layer_init_values, init_value, heads_range)
            for i in range(depth)
        ])
        if downsample is not None:
            self.downsample = downsample(dim=embed_dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_e):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x=x, x_e=x_e, split_or_not=self.split_or_not)
            else:
                x = blk(x, x_e, split_or_not=self.split_or_not)
        if self.downsample is not None:
            x_down = self.downsample(x)
            return x, x_down
        else:
            return x, x

# ==========================================
# 主模型 DFormerv2 (Cleaned Version)
# ==========================================
class dformerv2(nn.Module):
    def __init__(self, out_indices=(0, 1, 2, 3), embed_dims=[64, 128, 256, 512], depths=[2, 2, 8, 2], num_heads=[4, 4, 8, 16], init_values=[2, 2, 2, 2], heads_ranges=[4, 4, 6, 6], mlp_ratios=[4, 4, 3, 3], drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, layerscales=[False, False, False, False], layer_init_values=1e-6, pretrained=None):
        super().__init__()
        # 核心标记：告诉 Segmentor 我是单流模型，需要同时接受 RGB 和 Depth
        self.is_unified = True 
        
        self.out_indices = out_indices
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dims[0], norm_layer=norm_layer if self.patch_norm else None)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                init_value=init_values[i_layer],
                heads_range=heads_ranges[i_layer],
                ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                split_or_not=(i_layer != 3),
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                layerscale=layerscales[i_layer],
                layer_init_values=layer_init_values,
            )
            self.layers.append(layer)

        self.extra_norms = nn.ModuleList()
        for i in range(3):
            self.extra_norms.append(nn.LayerNorm(embed_dims[i + 1]))
        
        self.apply(self._init_weights)
        
        # 自动加载预训练
        if pretrained:
            self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except: pass

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # 使用原生 PyTorch 加载
            checkpoint = torch.load(pretrained, map_location='cpu')
            if "model" in checkpoint: _state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint: _state_dict = checkpoint["state_dict"]
            else: _state_dict = checkpoint

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith("backbone."):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v
            
            # 宽容加载 (strict=False)
            msg = self.load_state_dict(state_dict, strict=False)
            print(f"Loaded DFormer weights from {pretrained}")
            print(f"Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")

    def forward(self, x, x_e):
        # x: RGB [B, 3, H, W]
        # x_e: Depth [B, 1, H, W]
        x = self.patch_embed(x)
        
        # --- 修正后的深度图处理逻辑 ---
        # 目标：确保 x_e 是 [B, 1, H, W] 格式
        
        if x_e.dim() == 3: 
            # 如果输入是 [B, H, W]，增加一个通道维度
            x_e = x_e.unsqueeze(1)
        elif x_e.dim() == 4:
            if x_e.shape[1] == 3: 
                # 如果是 [B, 3, H, W]，取第一个通道
                x_e = x_e[:, 0:1, :, :]
            # 如果已经是 [B, 1, H, W]，什么都不做 (之前就是这里错了！)
        # ---------------------------
        
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, x = layer(x, x_e) # 传入 RGB 和 Depth
            if i in self.out_indices:
                if i != 0:
                    x_out = self.extra_norms[i - 1](x_out)
                out = x_out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        
        return list(outs)

# ==========================================
# 供外部调用的构建函数
# ==========================================
def DFormerv2_S(pretrained=None, **kwargs):
    return dformerv2(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        heads_ranges=[4, 4, 6, 6],
        pretrained=pretrained,
        **kwargs,
    )

def DFormerv2_B(pretrained=None, **kwargs):
    return dformerv2(
        embed_dims=[80, 160, 320, 512],
        depths=[4, 8, 25, 8],
        num_heads=[5, 5, 10, 16],
        heads_ranges=[5, 5, 6, 6],
        layerscales=[False, False, True, True],
        pretrained=pretrained,
        **kwargs,
    )

def DFormerv2_L(pretrained=None, **kwargs):
    return dformerv2(
        embed_dims=[112, 224, 448, 640],
        depths=[4, 8, 25, 8],
        num_heads=[7, 7, 14, 20],
        heads_ranges=[6, 6, 6, 6],
        layerscales=[False, False, True, True],
        pretrained=pretrained,
        **kwargs,
    )