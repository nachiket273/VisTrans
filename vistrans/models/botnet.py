"""Bottleneck Transformers

A Pytorch Implementation for Bottleneck Transformers.
'Bottleneck Transformers for Visual Recognition'
- https://arxiv.org/abs/2101.11605

Official tensorflow implementation is available at:
- https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
"""
import torch
import torch.nn as nn
from vistrans.util import ntuple
from torchvision import models


PRETRAINED_MODELS = [
    'botnet50',
    'botnet101',
    'botnet152'
]


DEFAULT_CFG = {
    'img_size': 224,
    'in_ch': 3,
    'num_classes': 1000,
    'groups': 1,
    'norm_layer': None,
    'n_heads': 4,
    'pos_enc_type': 'relative'
}


def get_actn(actn):
    actn = actn.strip()
    actn = actn.lower()
    if actn == 'relu':
        return nn.ReLU(inplace=True)
    return nn.SiLU(inplace=True)


def absolute_logits(w, h, q):
    emb = h[:, None, :] + w[None, :, :]
    abs_logits = torch.einsum('bhxyd,pqd->bhxypq', q, emb)
    return abs_logits


def rel_to_abs(rel_logits):
    bs, heads, length, _ = rel_logits.shape
    col_pad = torch.zeros((bs, heads, length, 1), dtype=rel_logits.dtype)
    x = torch.cat([rel_logits, col_pad], dim=3)
    flat_x = torch.reshape(x, [bs, heads, -1])
    flat_pad = torch.zeros((bs, heads, length - 1), dtype=x.dtype)
    flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
    final_x = torch.reshape(flat_x_padded, [bs, heads, length + 1,
                                            2 * length - 1])
    final_x = final_x[:, :, :length, length - 1:]
    return final_x


def relative_logits_1d(q, k, transpose_mask):
    _, heads, h, w, _ = q.shape
    rel_logits = torch.einsum('bhxyd,md->bhxym', q, k)
    rel_logits = torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
    rel_logits = rel_to_abs(rel_logits)
    rel_logits = torch.reshape(rel_logits, [-1, heads, h, w, w])
    rel_logits = torch.unsqueeze(rel_logits, dim=3)
    rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1)
    rel_logits = rel_logits.permute(*transpose_mask)
    return rel_logits


def relative_logits(w, h, q):
    rel_w = relative_logits_1d(q, w, transpose_mask=[0, 1, 2, 4, 3, 5])
    rel_h = relative_logits_1d(q.permute(0, 1, 3, 2, 4), h,
                               transpose_mask=[0, 1, 4, 2, 5, 3])
    return rel_w + rel_h


class RelativePosEmb(nn.Module):
    def __init__(self, fmap_dim, head_dim):
        super().__init__()
        h, w = ntuple(fmap_dim, 2)
        h, w = int(h), int(w)
        self.scale = head_dim ** -0.5
        self.h = nn.Parameter(torch.Tensor(2*h-1, head_dim))
        self.w = nn.Parameter(torch.Tensor(2*w-1, head_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w, self.scale)
        nn.init.normal_(self.h, self.scale)

    def forward(self, q, k):
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        logits += relative_logits(self.w, self.h, q)
        return logits


class AbsolutePosEmb(nn.Module):
    def __init__(self, fmap_dim, head_dim):
        super().__init__()
        h, w = ntuple(fmap_dim, 2)
        h, w = int(h), int(w)
        self.w = nn.Parameter(torch.Tensor(w, head_dim))
        self.h = nn.Parameter(torch.Tensor(h, head_dim))
        self.scale = head_dim ** -0.5
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.h, self.scale)
        nn.init.normal_(self.w, self.scale)

    def forward(self, q, k):
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        logits += absolute_logits(self.w, self.h, q)
        return logits


class GroupPointWise(nn.Module):
    def __init__(self, in_dim, n_heads=4, proj_factor=1, target_dim=None):
        super().__init__()
        if target_dim is not None:
            proj_ch = target_dim // proj_factor
        else:
            proj_ch = in_dim // proj_factor
        self.w = nn.Parameter(torch.Tensor(in_dim, n_heads,
                                           proj_ch // n_heads))
        nn.init.normal_(self.w, std=0.01)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out = torch.einsum('bhwc,cnp->bnhwp', x, self.w)
        return out


class MHSA(nn.Module):
    def __init__(self, in_dim, fmap_dim, n_heads=4, pos_enc_type='relative'):
        super().__init__()
        self.q = GroupPointWise(in_dim, n_heads)
        self.k = GroupPointWise(in_dim, n_heads)
        self.v = GroupPointWise(in_dim, n_heads)

        if pos_enc_type == 'relative':
            self.pos_emb = RelativePosEmb(fmap_dim, in_dim//n_heads)
        else:
            self.pos_emb = AbsolutePosEmb(fmap_dim, in_dim//n_heads)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        _, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)
        logits = self.pos_emb(q, k)
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = weights.softmax(dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        out = attn_out.permute(0, 3, 1, 2)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, fmap_dim, stride=1, downsample=None,
                 groups=1, norm_layer=None, mhsa=False, n_heads=4,
                 pos_enc_type='relative'):
        super().__init__()
        assert pos_enc_type in ['relative', 'absolute']
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = norm_layer(out_ch)
        if not mhsa:
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                                   stride=stride, padding=1, groups=groups,
                                   dilation=1)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(out_ch, fmap_dim, n_heads=n_heads,
                                   pos_enc_type=pos_enc_type))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = norm_layer(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(out_ch * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            x = self.downsample(x)
        out += x
        return self.relu(out)


class BotnetX(nn.Module):
    def __init__(self, layers, img_size=224, in_ch=3, num_classes=1000,
                 groups=1, n_heads=4, norm_layer=None,
                 pos_enc_type='relative'):
        super().__init__()
        if not norm_layer:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.img_size = img_size
        self.pos_enc_type = pos_enc_type
        self.n_heads = n_heads

        self.conv1 = nn.Conv2d(in_ch, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        if self.conv1.stride[0] == 2:
            self.img_size /= 2
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.maxpool.stride == 2:
            self.img_size /= 2

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2, True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, out_ch, num_blocks, stride=1, mhsa=False):
        downsample = None
        if stride != 1 or self.inplanes != out_ch * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_ch * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(out_ch * Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, out_ch, self.img_size, stride,
                                 downsample, self.groups, mhsa=mhsa,
                                 norm_layer=self.norm_layer,
                                 n_heads=self.n_heads,
                                 pos_enc_type=self.pos_enc_type))
        if stride == 2:
            self.img_size /= 2

        self.inplanes = out_ch * Bottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.inplanes, out_ch, self.img_size,
                                     groups=self.groups, mhsa=mhsa,
                                     norm_layer=self.norm_layer,
                                     n_heads=self.n_heads,
                                     pos_enc_type=self.pos_enc_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class BotNet():
    def __init__(self):
        super().__init__()

    @classmethod
    def create_model(cls, layers, img_size=224, in_ch=3, num_classes=1000,
                     groups=1, norm_layer=None, n_heads=4,
                     pos_enc_type='relative'):
        model = BotnetX(layers=layers, img_size=img_size, in_ch=in_ch,
                        num_classes=num_classes, groups=groups,
                        norm_layer=norm_layer, n_heads=n_heads,
                        pos_enc_type=pos_enc_type)
        return model

    @classmethod
    def list_pretrained(cls):
        return PRETRAINED_MODELS

    @classmethod
    def _is_valid_model_name(cls, name):
        name = name.strip()
        name = name.lower()
        return name in PRETRAINED_MODELS

    @classmethod
    def _get_default_cfg(cls):
        return DEFAULT_CFG

    @classmethod
    def create_pretrained(cls, name, img_size=224, in_ch=3,
                          num_classes=1000, n_heads=4,
                          pos_enc_type='relative'):
        if not BotNet._is_valid_model_name(name):
            raise ValueError("The pretrained model is not available.")

        if name == 'botnet50':
            model = models.resnet50(pretrained=True)
        elif name == 'botnet101':
            model = models.resnet101(pretrained=True)
        else:
            model = models.resnet152(pretrained=True)

        if in_ch != 3:
            model.conv1 = nn.Conv2d(in_ch, model.conv1.out_channels,
                                    kernel_size=model.conv1.kernel_size,
                                    stride=model.conv1.stride,
                                    padding=model.conv1.padding,
                                    bias=model.conv1.bias)

        if num_classes != model.fc.out_features:
            model.fc = nn.Linear(in_features=model.fc.in_features,
                                 out_features=num_classes, bias=model.fc.bias)

        out_ch = model.layer4[0].conv2.in_channels
        model.layer4[0].conv2 = nn.ModuleList()
        model.layer4[0].conv2.append(MHSA(out_ch, img_size//16,
                                          n_heads=n_heads,
                                          pos_enc_type=pos_enc_type))
        model.layer4[0].conv2.append(nn.AvgPool2d(2, 2))
        model.layer4[0].conv2 = nn.Sequential(*model.layer4[0].conv2)

        out_ch = model.layer4[1].conv2.in_channels
        model.layer4[1].conv2 = nn.ModuleList()
        model.layer4[1].conv2.append(MHSA(out_ch, img_size//32,
                                          n_heads=n_heads,
                                          pos_enc_type=pos_enc_type))
        model.layer4[1].conv2 = nn.Sequential(*model.layer4[1].conv2)

        out_ch = model.layer4[2].conv2.in_channels
        model.layer4[2].conv2 = nn.ModuleList()
        model.layer4[2].conv2.append(MHSA(out_ch, img_size//32,
                                          n_heads=n_heads,
                                          pos_enc_type=pos_enc_type))
        model.layer4[2].conv2 = nn.Sequential(*model.layer4[2].conv2)

        return model
