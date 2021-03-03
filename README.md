# VisTrans
Implementations of transformers based models for different vision tasks

## Version 0.001 (03/03/2021)
-----------------------------
Pretrained Pytorch [Vision Transformer](https://arxiv.org/abs/2010.11929) Models including following
<br>
* vit_s16_224
* vit_b16_224
* vit_b16_384
* vit_b32_384
* vit_l16_224
* vit_l16_384
* vit_l32_384
<br>
Implementation based off [official jax repository](https://github.com/google-research/vision_transformer) and [timm's implementation](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)<br>

## Install
---------------------
```Python
pip install git+https://github.com/nachiket273/VisTrans
```
1) List Pretrained Models.
```Python
from vistrans import VisionTransformer
VisionTransformer.list_pretrained()
```
2) Create Pretrained Models.
```Python
from vistrans import VisionTransformer
model = VisionTransformer.create_pretrained(name, img_size, in_ch, num_classes)
```
3) Create Custom Model
```Python
from vistrans import VisionTransformer
model = VisionTransformer.create_model(img_size, patch_size, in_ch, num_classes,
                                       embed_dim, depth, num_heads, mlp_ratio,
                                       drop_rate, attention_drop_rate, hybrid,
                                       norm_layer, bias)
```
