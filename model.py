import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
# from ops_dcnv3 import modules as opsm
from config_intern_g import config_dict
import os
import sys
models_path = os.path.join(os.getcwd(), "submodules", "internimage", "classification", "models")
dcn_path = os.path.join(os.getcwd(), "submodules", "internimage", "classification", )
sys.path.append(models_path)
sys.path.append(dcn_path)

from models.intern_image import InternImage
from ops_dcnv3 import modules as opsm

class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.norm1 = nn.LayerNorm(out_channels // 2)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x.permute(0, 2, 3, 1)


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        dim,
        qkv_bias=False,
        qk_scale=None,
        attn_head_dim=None,
        atten_drop=0.0,
        out_dim=None,
        projection_drop=0.0,
    ):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        assert all_head_dim == dim

        self.query = nn.Linear(dim, all_head_dim, bias=False)
        self.key = nn.Linear(dim, all_head_dim, bias=False)
        self.value = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.query_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.key_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.value_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.query_bias = None
            self.key_bias = None
            self.value_bias = None

        self.atten_dropout = nn.Dropout(atten_drop)
        self.projection = nn.Linear(all_head_dim, out_dim)
        self.projection_drop = nn.Dropout(projection_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        Nk = k.shape[1]
        Nv = v.shape[1]

        query_bias = self.q_bias if self.q_bias is not None else None
        key_bias = self.k_bias if self.k_bias is not None else None
        value_bias = self.v_bias if self.v_bias is not None else None

        q = F.linear(input=x, weight=self.q.weight, bias=query_bias)
        q = (
            q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        )  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=key_bias)
        k = k.reshape(B, Nk, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=value_bias)
        v = v.reshape(B, Nv, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentiveBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_head_dim=None,
        qk_scale=None,
        atten_drop=0.0,
        projection_drop=0.0,
        out_dim=None,
        drop_path=0.0,
    ):
        super(AttentiveBlock, self).__init__()

        self.q_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.k_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.v_norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.cross_dcn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            atten_drop=atten_drop,
            projection_drop=projection_drop,
            attn_head_dim=attn_head_dim,
            out_dim=out_dim,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)

        x = self.cross_dcn(x_q, k=x_k, v=x_v)

        return x


class AttentionPoolingBlock(AttentiveBlock):
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv = x
        pos_q, pos_k = 0, 0
        x = super().forward(
            x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None
        )
        x = x.squeeze(1)
        return x


class Downsampling(nn.Module):
    def __init__(self, channels):
        super(Downsampling, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=int(channels * 2),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.norm = nn.LayerNorm(int(channels * 2))

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class BasicInternImageLayer(nn.Module):
    def __init__(
        self,
        core_op,
        channels,
        groups,
        mlp_ratio=4.0,
        drop=0.0,
        with_cp=False,
        post_norm=False,
        drop_path=0.0,
        layer_scale=None,
        res_post_norm=True,
        offset_scale=1.0,
        dw_kernel_size=None,
        center_feature_scale=True,
        remove_center=True,
    ):
        super(BasicInternImageLayer, self).__init__()

        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = nn.LayerNorm(self.channels)
        self.post_norm = post_norm
        self.dcn = core_op(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer="GELU",
            norm_layer="LN",
            dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
            center_feature_scale=center_feature_scale,  # for InternImage-H/G
            remove_center=remove_center,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(self.channels)
        self.mlp = MLP(
            in_features=self.channels,
            hidden_features=int(self.channels * self.mlp_ratio),
            out_features=self.channels,
            dropout_rate=drop,
        )
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(channels), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(channels), requires_grad=True
            )
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = nn.LayerNorm(self.channels)
            self.res_post_norm2 = nn.LayerNorm(self.channels)

    def forward(self, x):
        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm:  # for InternImage-H/G
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x))))
                    x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class BasicInternImageBlock(nn.Module):
    def __init__(
        self,
        core_op,
        channels,
        depth,
        groups,
        center_feature_scale=True,
        drop=0.0,
        offset_scale=1.0,
        dw_kernel_size=None,
        remove_center=True,
        mlp_ratio=4.0,
        downsample=True,
        post_norm_block_ids=None,
        with_cp=False,
        post_norm=False,
        drop_path=0.0,
        layer_scale=None,
        res_post_norm=True,
    ):
        super(BasicInternImageBlock, self).__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale

        self.blocks = nn.ModuleList(
            [
                BasicInternImageLayer(
                    core_op=core_op,
                    channels=channels,
                    groups=groups,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    post_norm=post_norm,
                    layer_scale=layer_scale,
                    offset_scale=offset_scale,
                    with_cp=with_cp,
                    dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                    res_post_norm=res_post_norm,  # for InternImage-H/G
                    center_feature_scale=center_feature_scale,  # for InternImage-H/G
                    remove_center=remove_center,  # for InternImage-H/G
                )
                for i in range(depth)
            ]
        )
        if not self.post_norm or center_feature_scale:
            self.norm = nn.LayerNorm(channels)
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None:  # for InternImage-H/G
            self.post_norms = nn.ModuleList(
                [nn.LayerNorm(channels, eps=1e-6) for _ in post_norm_block_ids]
            )
        self.downsample = Downsampling(channels=channels) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (self.post_norm_block_ids is not None) and (
                i in self.post_norm_block_ids
            ):
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x)  # for InternImage-H/G
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x


class InternImageCustom(nn.Module):
    def __init__(
        # Pulled from https://github.com/OpenGVLab/InternImage/blob/master/classification/configs/internimage_g_22kto1k_512.yaml
        self,
        core_op="DCNv3",
        num_classes=1000,
        depths=[2, 2, 48, 4],
        channels=512,
        post_norm=True,
        mlp_ratio=4.0,
        use_clip_projector=True,
        level2_post_norm_block_ids=[5, 11, 17, 23, 29, 35, 41, 47],
        remove_center=False,
        drop_rate=0.0,
        drop_path_rate=0.1,
        level2_post_norm=True,
        drop_path_type="linear",
        layer_scale=None,
        offset_scale=1.0,
        groups=[16, 32, 64, 128],
        with_cp=True,
        dw_kernel_size=5,
        res_post_norm=False,
        center_feature_scale=True,
    ):
        super(InternImageCustom, self).__init__()
        self.core_op = core_op
        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2 ** (self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        self.use_clip_projector = use_clip_projector
        self.level2_post_norm_block_ids = level2_post_norm_block_ids
        self.remove_center = remove_center

        in_chans = 3
        self.patch_embed = Stem(
            in_channels=in_chans,
            out_channels=channels,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if drop_path_type == "uniform":
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            post_norm_block_ids = (
                level2_post_norm_block_ids if level2_post_norm and (i == 2) else None
            )  # for InternImage-H/G
            level = BasicInternImageBlock(
                core_op=getattr(opsm, core_op),
                channels=int(channels * 2**i),
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                post_norm_block_ids=post_norm_block_ids,  # for InternImage-H/G
                res_post_norm=res_post_norm,  # for InternImage-H/G
                center_feature_scale=center_feature_scale,  # for InternImage-H/G
                remove_center=remove_center,  # for InternImage-H/G
            )
            self.levels.append(level)

        pretrain_embed_dim, _stride, attnpool_num_heads, clip_embed_dim = (
            1024,
            2,
            16,
            768,
        )
        self.dcnv3_head_x4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_features,
                out_channels=pretrain_embed_dim * (_stride**2),
                kernel_size=1,
            ),
            nn.PixelShuffle(_stride),
        )
        self.dcnv3_head_x3 = nn.Conv2d(
            in_channels=self.num_features // 2,
            out_channels=pretrain_embed_dim,
            kernel_size=1,
        )
        self.clip_projector = AttentionPoolingBlock(
            dim=pretrain_embed_dim,
            num_heads=attnpool_num_heads,
            qkv_bias=True,
            qk_scale=None,
            projection_drop=0.0,
            atten_drop=0.0,
            out_dim=clip_embed_dim,
        )
        self.fc_norm = nn.LayerNorm(clip_embed_dim, eps=1e-6)
        self.head = (
            nn.Linear(clip_embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_layers = len(depths)
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_deform_weights(self, m):
        if isinstance(m, getattr(opsm, self.core_op)):
            m._reset_parameters()

    @torch.jit.ignore
    def lr_decay_keywords(self, decay_ratio=0.87):
        lr_ratios = {}

        # blocks
        idx = 0
        for i in range(4):
            layer_num = 3 - i  # 3 2 1 0
            for j in range(self.depths[layer_num]):
                block_num = self.depths[layer_num] - j - 1
                tag = "levels.{}.blocks.{}.".format(layer_num, block_num)
                decay = 1.0 * (decay_ratio**idx)
                lr_ratios[tag] = decay
                idx += 1
        # patch_embed (before stage-1)
        lr_ratios["patch_embed"] = lr_ratios["levels.0.blocks.0."]
        # levels.0.downsample (between stage-1 and stage-2)
        lr_ratios["levels.0.downsample"] = lr_ratios["levels.1.blocks.0."]
        lr_ratios["levels.0.norm"] = lr_ratios["levels.1.blocks.0."]
        # levels.1.downsample (between stage-2 and stage-3)
        lr_ratios["levels.1.downsample"] = lr_ratios["levels.2.blocks.0."]
        lr_ratios["levels.1.norm"] = lr_ratios["levels.2.blocks.0."]
        # levels.2.downsample (between stage-3 and stage-4)
        lr_ratios["levels.2.downsample"] = lr_ratios["levels.3.blocks.0."]
        lr_ratios["levels.2.norm"] = lr_ratios["levels.3.blocks.0."]
        return lr_ratios

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)

        x = self.conv_head(x.permute(0, 3, 1, 2))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_features_seq_out(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        seq_out = []
        for level in self.levels:
            x, x_ = level(x, return_wo_downsample=True)
            seq_out.append(x_)
        return seq_out

    def forward_clip_projector(self, x):  # for InternImage-H/G
        xs = self.forward_features_seq_out(x)
        x1, x2, x3, x4 = xs

        x1 = x1.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x2 = x2.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x3 = x3.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x4 = x4.permute(0, 3, 1, 2)  # NHWC -> NCHW

        x4 = self.dcnv3_head_x4(x4)
        x = x4
        x3 = self.dcnv3_head_x3(x3)
        x = x + x3

        x = x.flatten(-2).transpose(1, 2).contiguous()
        x = self.clip_projector(x)
        x = self.fc_norm(x)

        return x

    def forward(self, x):
        x = self.forward_clip_projector(x)
        x = self.head(x)
        return x