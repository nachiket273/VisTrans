""" Vision Transformer

A PyTorch implementation of Vision Transformers:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
- https://arxiv.org/abs/2010.11929

The official jax code is available at
https://github.com/google-research/vision_transformer
"""
from copy import deepcopy
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from vistrans.util import ntuple
from vistrans.util import get_pretrained_weights


PRETRAINED_MODELS = [
    'vit_s16_224',
    'vit_b16_224',
    'vit_b16_384',
    'vit_b32_384',
    'vit_l16_224',
    'vit_l16_384',
    'vit_l32_384'
]

PRETRAINED_URLS = {
    'vit_s16_224': 'https://github.com/nachiket273/VisTrans/releases/download/v0.001/vit_s16_224_45607315.pth',
    'vit_b16_224': 'https://github.com/nachiket273/VisTrans/releases/download/v0.001/vit_b16_224_f3d9f89c.pth',
    'vit_b16_384': 'https://github.com/nachiket273/VisTrans/releases/download/v0.001/vit_b16_384_a14a3f44.pth',
    'vit_b32_384': 'https://github.com/nachiket273/VisTrans/releases/download/v0.001/vit_b32_384_d6ce8791.pth',
    'vit_l16_224': 'https://github.com/nachiket273/VisTrans/releases/download/v0.001/vit_l16_224_e4578f73.pth',
    'vit_l16_384': 'https://github.com/nachiket273/VisTrans/releases/download/v0.001/vit_l16_384_75cf650d.pth',
    'vit_l32_384': 'https://github.com/nachiket273/VisTrans/releases/download/v0.001/vit_l32_384_2218c801.pth'
}

DEFAULT_CFG = {
    'img_size': 224,
    'patch_size': 16,
    'in_ch': 3,
    'num_classes': 1000,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4.,
    'drop_rate': 0.,
    'attention_drop_rate': 0.,
    'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    'bias': True
}


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.img_size = ntuple(img_size, 2)
        self.patch_size = ntuple(patch_size, 2)
        self.num_patches = (self.img_size[1]//self.patch_size[1]) \
            * (self.img_size[0]//self.patch_size[0])
        self.patch_embed = nn.Conv2d(in_ch, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, h, w = x.shape
        if h != self.img_size[0] or w != self.img_size[1]:
            raise ValueError('Input image size (%d*%d) doesn\'t match model'
                             'image size (%d*%d)' % (h, w, self.img_size[0],
                                                     self.img_size[1]))
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        return patches


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 dropout=0., act_layer=nn.GELU):
        super().__init__()
        if not hidden_features:
            hidden_features = in_features
        if not out_features:
            out_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.actn = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.actn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads)**-0.5

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c//self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_head=12, mlp_ratio=4,
                 attention_drop_rate=0., dropout_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_head
        self.attn = Attention(embed_dim, num_heads=n_head,
                              attn_dropout=attention_drop_rate)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim*mlp_ratio),
                       dropout=dropout_ratio, act_layer=act_layer)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.norm1.weight, 1.0)
        nn.init.constant_(self.norm1.bias, 0)
        nn.init.constant_(self.norm2.weight, 1.0)
        nn.init.constant_(self.norm2.bias, 0)

    def forward(self, x):
        y = self.norm1(x)
        y = self.attn(y)
        x = x + self.dropout(y)
        y = self.norm2(x)
        return x + self.mlp(y)


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(encoder_layer)
                                     for _ in range(num_layers)])
        self.norm = norm_layer(encoder_layer.embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        return out


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 drop_rate=0., attention_drop_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), bias=True):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(in_ch, img_size, patch_size,
                                      self.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1,
                                                  embed_dim))
        self.dropout = nn.Dropout(p=drop_rate)

        enc_layer = EncoderLayer(embed_dim, n_head=num_heads,
                                 mlp_ratio=mlp_ratio,
                                 attention_drop_rate=attention_drop_rate,
                                 dropout_ratio=drop_rate,
                                 norm_layer=norm_layer, bias=bias)

        self.enc = Encoder(enc_layer, depth)

        # Classifier head
        self.head = nn.Linear(self.embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.2)
        nn.init.trunc_normal_(self.cls_token, std=0.2)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.)

    def forward(self, x):
        bs = x.shape[0]
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        x = self.patch_embed(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.enc(x)
        return self.head(x[:, 0])


class VisionTransformer():
    def __init__(self):
        super().__init__()

    @classmethod
    def create_model(cls, img_size=224, patch_size=16, in_ch=3,
                     num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                     mlp_ratio=4., drop_rate=0., attention_drop_rate=0.,
                     norm_layer=partial(nn.LayerNorm, eps=1e-6),
                     bias=True):

        return ViT(img_size, patch_size, in_ch, num_classes, embed_dim, depth,
                   num_heads, mlp_ratio, drop_rate, attention_drop_rate,
                   norm_layer, bias)

    @classmethod
    def list_pretrained(cls):
        return PRETRAINED_MODELS

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
        cfg = VisionTransformer._get_default_cfg()
        if name == 'vit_s16_224':
            cfg['depth'] = 8
            cfg['mlp_ratio'] = 3
            cfg['num_heads'] = 8
            cfg['bias'] = False
        elif name == 'vit_b16_384':
            cfg['img_size'] = 384
        elif name == 'vit_b32_384':
            cfg['img_size'] = 384
            cfg['patch_size'] = 32
        elif name == 'vit_l16_224':
            cfg['depth'] = 24
            cfg['embed_dim'] = 1024
            cfg['num_heads'] = 16
        elif name == 'vit_l16_384':
            cfg['img_size'] = 384
            cfg['depth'] = 24
            cfg['embed_dim'] = 1024
            cfg['num_heads'] = 16
        elif name == 'vit_l32_384':
            cfg['img_size'] = 384
            cfg['depth'] = 24
            cfg['embed_dim'] = 1024
            cfg['num_heads'] = 16
            cfg['patch_size'] = 32
        return cfg

    @classmethod
    def create_pretrained(cls, name, img_size=0, in_ch=0,
                          num_classes=0):
        if not VisionTransformer._is_valid_model_name(name):
            raise ValueError('Available pretrained models: ' +
                             ', '.join(PRETRAINED_MODELS))

        cfg = VisionTransformer._get_cfg(name)
        cfg['conv1'] = 'patch_embed.patch_embed'
        cfg['classifier'] = 'head'
        cfg['strict'] = True

        img_size = cfg['img_size'] if img_size == 0 else img_size
        patch_size = cfg['patch_size']
        in_ch = cfg['in_ch'] if in_ch == 0 else in_ch
        num_classes = cfg['num_classes'] if num_classes == 0 else num_classes

        if img_size % patch_size != 0:
            raise ValueError("Image size should be divisible by patch size.")

        url = VisionTransformer._get_url(name)
        model = ViT(img_size, patch_size, in_ch, num_classes, cfg['embed_dim'],
                    cfg['depth'], cfg['num_heads'], cfg['mlp_ratio'],
                    cfg['drop_rate'], cfg['attention_drop_rate'],
                    cfg['hybrid'], cfg['norm_layer'], cfg['bias'])

        state_dict = get_pretrained_weights(url, cfg, num_classes, in_ch,
                                            check_hash=True)

        # Update state_dict if img_size doesn't match.
        # Based of https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
        # and https://github.com/rwightman/pytorch-image-models/blob/95feb1da41c1fe95ce9634b83db343e08224a8c5/timm/models/vision_transformer.py#L464
        if(model.pos_embed.shape[1] != state_dict['pos_embed'].shape[1]):
            posemb_tok = state_dict['pos_embed'][:, :1]
            posemb_grid = state_dict['pos_embed'][0, 1:]
            gs_old = int(math.sqrt(len(posemb_grid)))
            gs_new = int(math.sqrt(model.pos_embed.shape[1]-1))
            posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0,
                                                                             3,
                                                                             1,
                                                                             2)
            posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new),
                                        mode='bilinear', align_corners=True)
            posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new *
                                                                  gs_new, -1)
            state_dict['pos_embed'] = torch.cat([posemb_tok, posemb_grid],
                                                dim=1)

        model.load_state_dict(state_dict, strict=cfg['strict'])
        return model
