from model.swin_block_noswin import NoSwinblock
from model.swin_block_swin import Swinblock
from model.patch_embed import PatchEmbed
from model.patch_merge import PatchMerging

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, LayerNormalization, Activation, AveragePooling1D, Flatten, Softmax
from tensorflow.keras import Model
from tensorflow import nn

# Inspiration from https://www.kaggle.com/code/raufmomin/vision-transformer-vit-from-scratch
# https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow/blob/main/models/swin_transformer.py
# https://keras.io/examples/vision/swin_transformers/
class SwinModel(Model):
    def __init__(self, dim, resolution=[224,224], num_heads=[3,6,12,24], depths=[2, 2, 6, 2], window_size=7, shift_size=3,
                 mlp_ratio=4., qkv_bias=True, mlp_drop= 0., attn_drop=0., proj_drop=0., drop_ratio= 0.,
                 ff_drop=0., act_layer=Activation(tf.nn.gelu), norm_layer=LayerNormalization,
                 patch_size=2, in_chans=3, num_classes=1000):

        super().__init__()

        self.depths = depths
        #PatchEmb
        self.img_size = resolution[0]
        self.patch_size = patch_size
        self.in_chans = in_chans

        #NoSwin
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.resolution = resolution
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_ratio = drop_ratio
        #Swin
        self.shift_size = shift_size
        self.norm = norm_layer
        self.mlp_drop = mlp_drop
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.ff_drop = ff_drop
        self.act_layer = act_layer

        #DownSample
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size,
                                      in_chans=self.in_chans, embed_dim=self.dim,
                                      norm_layer=self.norm)

        self.resolution = [self.resolution[0] // self.patch_size,
                           self.resolution[1] // self.patch_size]

        self.transformer_layers = []

        for n in range(len(depths)):
            for i in range(self.depths[n] // 2):
                self.transformer_layers.append(NoSwinblock(dim=self.dim, resolution=self.resolution,
                                    num_heads=num_heads[n], window_size=self.window_size,
                                    mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
                                    drop_ratio=self.drop_ratio))

                self.transformer_layers.append(Swinblock(dim=self.dim, resolution=self.resolution,
                              num_heads=num_heads[n], window_size=self.window_size,
                              shift_size=self.shift_size, mlp_ratio=self.mlp_ratio,
                              qkv_bias=self.qkv_bias, mlp_drop=self.mlp_drop,
                              attn_drop=self.attn_drop, proj_drop=self.proj_drop,
                              ff_drop=self.ff_drop, act_layer=self.act_layer,
                              norm_layer=self.norm))

            if n < len(depths)-1:
                self.transformer_layers.append(PatchMerging(input_resolution=self.resolution,
                                                         dim=self.dim, norm_layer=self.norm))

                self.resolution = [self.resolution[0] // 2, self.resolution[1] // 2]
                self.dim = self.dim * 2

        self.linear = Dense(self.num_classes)
        self.softmax = Softmax(axis=-1)

    def call(self, x):
        x = self.patch_embed(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = AveragePooling1D(pool_size=1)(x)
        x = Flatten()(x)
        out = self.linear(x)

        return out