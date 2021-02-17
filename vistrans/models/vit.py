""" Vision Transformer

A PyTorch implementation of Vision Transformers:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
- https://arxiv.org/abs/2010.11929

The official jax code is available at
https://github.com/google-research/vision_transformer
"""
from copy import deepcopy
from functools import partial
import torch
import torch.nn as nn

from vistrans.util import ntuple


# TO-DO: Implement hybrid embedding (features from some backbone as input)
class HybridEmbed(nn.Module):
    def __init__(self):
        super().__init__()


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


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_head=12, mlp_ratio=4,
                 attention_drop_rate=0., dropout_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_head
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=n_head,
                                          dropout=attention_drop_rate,
                                          bias=bias)
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
        y, _ = self.attn(y, y, y)
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


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 drop_rate=0., attention_drop_rate=0., hybrid=False,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), bias=True):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        if hybrid:
            self.patch_embed = HybridEmbed()  # not-implemented currently
        else:
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


def assign_weights_from_pretrained(model, model1, depth=12):
    model.patch_embed.patch_embed.weight = model1.patch_embed.proj.weight
    model.patch_embed.patch_embed.bias = model1.patch_embed.proj.bias
    model.enc.norm.weight = model1.norm.weight
    model.enc.norm.bias = model1.norm.bias
    model.head.weight = model1.head.weight
    model.head.bias = model1.head.bias
    model.cls_token = model1.cls_token

    # If input image size is different, this will need different treatment.
    model.pos_embed = model1.pos_embed

    for i in range(depth):
        model.enc.layers[i].norm1.weight = model1.blocks[i].norm1.weight
        model.enc.layers[i].norm1.bias = model1.blocks[i].norm1.bias
        model.enc.layers[i].norm2.weight = model1.blocks[i].norm2.weight
        model.enc.layers[i].norm2.bias = model1.blocks[i].norm2.bias
        model.enc.layers[i].mlp.fc1.weight = model1.blocks[i].mlp.fc1.weight
        model.enc.layers[i].mlp.fc1.bias = model1.blocks[i].mlp.fc1.bias
        model.enc.layers[i].mlp.fc2.weight = model1.blocks[i].mlp.fc2.weight
        model.enc.layers[i].mlp.fc2.bias = model1.blocks[i].mlp.fc2.bias
        model.enc.layers[i].attn.in_proj_weight \
            = model1.blocks[i].attn.qkv.weight
        model.enc.layers[i].attn.in_proj_bias = model1.blocks[i].attn.qkv.bias
        model.enc.layers[i].attn.out_proj.weight \
            = model1.blocks[i].attn.proj.weight
        model.enc.layers[i].attn.out_proj.bias \
            = model1.blocks[i].attn.proj.bias
