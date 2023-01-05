import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, Softmax, Dropout, Activation
from tensorflow.keras import initializers

class TruncatedDense(Dense):
    def _init_(self, units, use_bias=True, initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)):
        super()._init_(units, use_bias=use_bias, kernel_initializer=initializer)

class PatchMerging(Layer):
    
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = TruncatedDense(2 * dim, use_bias=False)
        self.norm = norm_layer(epsilon=1e-5)

    def call(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, [-1, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = tf.reshape(x, [-1, (H//2) * (W//2), 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim # merging
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim # reduction
        return flops