"""Bottleneck Transformers

A Pytorch Implementation for Bottleneck Transformers.
'Bottleneck Transformers for Visual Recognition'
- https://arxiv.org/abs/2101.11605

Official tensorflow implementation is available at:
- https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
"""
import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from vistrans.util import ntuple


PRETRAINED_MODELS = [

]

PRETRAINED_URLS = {

}

DEFAULT_CFG = {

}


def get_actn(actn):
    actn = actn.strip()
    actn = actn.lower()
    if actn == 'relu':
        return nn.ReLU(inplace=True)
    return nn.SiLU(inplace=True)


class RelativePosEmb(nn.Module):
    def __init__(self, fmap_dim, head_dim):
        super().__init__()


class AbsolutePosEmb(nn.Module):
    def __init__(self, fmap_dim, head_dim):
        super().__init__()
        h, w = ntuple(fmap_dim, 2)
        scale = head_dim ** -0.5
        self.h = nn.Parameter(torch.randn(h, head_dim) * scale)
        self.w = nn.Parameter(torch.randn(w, head_dim) * scale)

    def forward(self, x):
        emb = self.h.unsqueeze(1) + self.w.unsqueeze(0)
        emb = emb.view(-1, emb.shape[-1])
        return torch.einsum('b h i d, j d -> b h i j', x, emb)


class MHSA(nn.Module):
    def __init__(self, in_dim, fmap_dim, n_heads=4, head_dim=128,
                 pos_enc_type='relative'):
        super().__init__()
        self.n_heads = n_heads
        self.scale = head_dim ** -0.5
        hidden_dim = self.n_heads * head_dim * 3
        self.conv1 = Conv2d(in_dim, hidden_dim, stride=1, bias=False)
        if pos_enc_type == 'relative':
            self.pos_emb = RelativePosEmb(fmap_dim, head_dim)
        else:
            self.pos_emb = AbsolutePosEmb(fmap_dim, head_dim)

    def forward(self, x):
        q, k, v = self.conv1(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(t.shape[0], self.n_heads,
                                       t.shape[1]//self.n_heads,
                                       -1).permute(0, 1, 3, 2), (q, k, v))
        q *= self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        sim += self.pos_emb(q)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 1, 3, 2)
        out = out.view(out.shape[0], out.shape[1] * out.shape[2],
                       out.shape[3]//self.n_heads, self.n_heads)
        return out


class BotBlock(nn.Module):
    def __init__(self, in_dim, fmap_dim, out_dim, proj_factor=4, n_head=4,
                 downsample=True, pos_enc_type='relative', actn='relu',
                 head_dim=128):
        super().__init__()
        kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)
        self.actn = get_actn(actn)
        if in_dim != out_dim or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_dim),
                self.actn
            )
        else:
            self.shortcut = nn.Identity()

        mhsa_in_dim = out_dim//proj_factor
        mhsa_out_dim = head_dim * n_head

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, mhsa_in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mhsa_in_dim),
            self.actn
        )

        self.mhsa = MHSA(mhsa_in_dim, fmap_dim, n_head=n_head,
                         head_dim=head_dim, pos_enc_type=pos_enc_type)

        conv2 = []
        if downsample:
            conv2.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))
        conv2.append(nn.BatchNorm2d(mhsa_out_dim))
        conv2.append(self.actn)
        self.conv2 = nn.Sequential(*conv2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mhsa_out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim)
        )

        self.actn = actn
        self._init_weights()

    def _init_weights(self):
        # zero initialize last batch normalization layer
        nn.init.zeros_(self.conv3[-1].weight)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.mhsa(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += shortcut
        return self.actn(x)


class BotStack(nn.Module):
    def __init__(self, in_dim, fmap_dim, out_dim, proj_factor=4, n_head=4,
                 downsample=True, pos_enc_type='relative', actn='relu',
                 head_dim=128, num_layers=3):
        super().__init__()
        fmap_size = ntuple(fmap_dim, 2)
        self.in_dim = in_dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            dim = in_dim if i == 0 else out_dim
            fmap_div = 2 if i != 0 and downsample else 1
            layer_fmap_size = fmap_size/fmap_div
            layer_fmap_size = ntuple(layer_fmap_size, 2)

            layers.append(BotBlock(
                dim,
                layer_fmap_size,
                out_dim,
                proj_factor=proj_factor,
                n_head=n_head,
                downsample=(downsample and i == 0),
                pos_enc_type=pos_enc_type,
                actn=actn,
                head_dim=head_dim
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BotNet():
    def __init__(self):
        super().__init__()

    @classmethod
    def create_model(cls):
        pass

    @classmethod
    def list_pretrained(cls):
        pass

    @classmethod
    def _is_valid_model_name(cls, name):
        name = name.strip()
        name = name.lower()
        return name in PRETRAINED_MODELS

    @classmethod
    def _get_url(cls, name):
        return PRETRAINED_URLS[name]

    @classmethod
    def _get_default_cfg(cls):
        return DEFAULT_CFG

    @classmethod
    def _get_cfg(cls, name):
        cfg = BotNet._get_default_cfg()
        return cfg

    @classmethod
    def create_pretrained(cls):
        pass
