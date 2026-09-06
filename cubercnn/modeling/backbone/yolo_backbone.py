# =====================================================================
# YOLO26 Backbone + PAN Neck for Detectron2
# =====================================================================
import math
import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=(1, 3)):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=k, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k=(k, k))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        out = self.cv2(torch.cat(y, 1))
        return x + out if self.add else out

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        h = dim + self.key_dim * num_heads * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)

class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), 1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

YOLO26_SCALES = {
    # size: (depth_multiple, width_multiple, max_channels)
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}

def make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor

def scale_ch(c, width, max_ch):
    return make_divisible(min(c, max_ch) * width, 8)

def scale_repeat(n, depth):
    return max(round(n * depth), 1) if n > 1 else n


class YOLO26(Backbone):
    def __init__(self, cfg, input_shape=None):
        super().__init__()
        size = cfg.MODEL.BACKBONE.SIZE.lower()
        assert size in YOLO26_SCALES, f"MODEL.YOLO26.SIZE must be one of the {list(YOLO26_SCALES)},got {size}"
        depth, width, max_ch = YOLO26_SCALES[size]
        in_ch = input_shape.channels if input_shape is not None else 3

        ch = lambda c: scale_ch(c, width, max_ch)
        rep = lambda n: scale_repeat(n, depth)

        # ---- stem: P1/2, P2/4 ----
        self.stem1 = Conv(in_ch, ch(64), 3, 2)                                  # 0  P1/2
        self.stem2 = Conv(ch(64), ch(128), 3, 2)                                # 1  P2/4
        self.stage2 = C3k2(ch(128), ch(256), rep(2), c3k=False, e=0.25)         # 2

        # ---- P3/8 ----
        self.down3 = Conv(ch(256), ch(256), 3, 2)                               # 3
        self.stage3 = C3k2(ch(256), ch(512), rep(2), c3k=False, e=0.25)         # 4  -> p3

        # ---- P4/16 ----
        self.down4 = Conv(ch(512), ch(512), 3, 2)                               # 5
        self.stage4 = C3k2(ch(512), ch(512), rep(2), c3k=True)                  # 6  -> p4

        # ---- P5/32 ----
        self.down5 = Conv(ch(512), ch(1024), 3, 2)                              # 7
        self.stage5 = C3k2(ch(1024), ch(1024), rep(2), c3k=True)                # 8
        self.sppf = SPPF(ch(1024), ch(1024), k=5, n=3, shortcut=False)          # 9
        self.c2psa = C2PSA(ch(1024), ch(1024), rep(2))                          # 10 -> p5

        self._out_feature_channels = {"p3": ch(512), "p4": ch(512), "p5": ch(1024)}
        self._out_feature_strides = {"p3": 8, "p4": 16, "p5": 32}
        self._out_features = ["p3", "p4", "p5"]

    def forward(self, x):
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.stage2(x)

        x = self.down3(x)
        p3 = self.stage3(x)

        x = self.down4(p3)
        p4 = self.stage4(x)

        x = self.down5(p4)
        x = self.stage5(x)
        x = self.sppf(x)
        p5 = self.c2psa(x)

        return {"p3": p3, "p4": p4, "p5": p5}

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }

@BACKBONE_REGISTRY.register()
def build_yolo26_backbone(cfg, input_shape):
    return YOLO26(cfg, input_shape)

# TODO We only output p3-p5 here, complete full output in the future
class YOLO26PAN(Backbone):
    def __init__(self, cfg, bottom_up):
        super().__init__()
        self.bottom_up = bottom_up

        size = cfg.MODEL.BACKBONE.SIZE.lower()
        assert size in YOLO26_SCALES
        depth, width, max_ch = YOLO26_SCALES[size]
        ch = lambda c: scale_ch(c, width, max_ch)
        rep = lambda n: scale_repeat(n, depth)

        in_shapes = bottom_up.output_shape()
        c3, c4, c5 = in_shapes["p3"].channels, in_shapes["p4"].channels, in_shapes["p5"].channels

        # ---- top-down (FPN) ----
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k2_n4 = C3k2(c5 + c4, ch(512), rep(2), c3k=False)          # N4

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k2_p3 = C3k2(ch(512) + c3, ch(256), rep(2), c3k=False)     # out P3

        # ---- bottom-up (PAN) ----
        self.down1 = Conv(ch(256), ch(256), 3, 2)
        self.c3k2_p4 = C3k2(ch(256) + ch(512), ch(512), rep(2), c3k=False)  # out P4

        self.down2 = Conv(ch(512), ch(512), 3, 2)
        self.c3k2_p5 = C3k2(ch(512) + c5, ch(1024), rep(2), c3k=True)     # out P5

        # ---- NEW: Projection layers to unify output channels ----
        # Projecting to 256 channels to align with DenseHead's expected input
        out_dim = 256
        self.proj_p3 = Conv(ch(256), out_dim, 1, 1)
        self.proj_p4 = Conv(ch(512), out_dim, 1, 1)
        self.proj_p5 = Conv(ch(1024), out_dim, 1, 1)

        self._out_feature_channels = {"p3": out_dim, "p4": out_dim, "p5": out_dim}
        self._out_feature_strides = {"p3": 8, "p4": 16, "p5": 32}
        self._out_features = ["p3", "p4", "p5"]

    def forward(self, x):
        bottom = self.bottom_up(x)
        p3, p4, p5 = bottom["p3"], bottom["p4"], bottom["p5"]

        n4 = self.c3k2_n4(torch.cat([self.up1(p5), p4], 1))
        out_p3 = self.c3k2_p3(torch.cat([self.up2(n4), p3], 1))

        out_p4 = self.c3k2_p4(torch.cat([self.down1(out_p3), n4], 1))
        out_p5 = self.c3k2_p5(torch.cat([self.down2(out_p4), p5], 1))

        # Apply projections to uniformize channel depth
        out_p3 = self.proj_p3(out_p3)
        out_p4 = self.proj_p4(out_p4)
        out_p5 = self.proj_p5(out_p5)

        return {"p3": out_p3, "p4": out_p4, "p5": out_p5}

    @property
    def size_divisibility(self):
        return 32  

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }

@BACKBONE_REGISTRY.register()
def build_yolo26_pan_backbone(cfg, input_shape):
    bottom_up = YOLO26(cfg, input_shape)
    return YOLO26PAN(cfg, bottom_up)