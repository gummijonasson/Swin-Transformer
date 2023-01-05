import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, Softmax, Dropout, Activation, Conv2D
from tensorflow.keras import initializers
import collections

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return(x,x)
class PatchEmbed(Layer):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, data_format="channels_last")
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5)
        else:
            self.norm = None

    def call(self, x):
        B, H, W, C = x.shape
        x = self.proj(x)
        x = tf.reshape(x, [-1, self.num_patches, self.embed_dim])
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops