# -*- coding: utf-8 -*-
# @Time    : 12/9/22 8:45 PM
# @Author  : Wenliang Guo
# @email   : wg2397@columbia.edu
# @FileName: swin_block_swin.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, Softmax, Dropout, Activation
from tensorflow.keras import initializers

def window_partition(x, ws):
    '''
    :param x: input feature with size (B, H, W, D)
    :param ws: window size
    :return portioned x: (B*num of windows, size of window, size of window, D)
    '''
    B,H,W,D = x.shape
    x = tf.reshape(x, shape=[-1,H//ws,ws,W//ws,ws,D])
    windows = tf.transpose(x,perm=[0,1,3,2,4,5])
    windows = tf.reshape(windows,shape=[-1,ws,ws,D])
    return windows

def window_reverse(windows, window_size, H, W):
    '''
    :param windows: (num_windows*B, window_size, window_size, D)
    :param window_size: Window size
    :param H: height of feature
    :param W: width of feature
    :return: (B, H, W, D)
    '''
    
    B, ws, ws, D = windows.shape
    x = tf.reshape(windows, shape=[-1, H // window_size, W // window_size, window_size, window_size, D])    # [B', h_nw, w_nw, ws, ws, D']
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])                                                            # [B', h_nw, ws, w_nw, ws, D']
    x = tf.reshape(x, shape=[-1, H, W, D])                                                                  # [B, H, W, D]
    return x

def create_mask(H,W,window_size, shift_size):
    '''
    :param H: height of feature
    :param W: width of feature
    :param window_size: window size
    :param shift_size: shift size
    :return: ([1, H, W, 1])
    '''
    mask = np.zeros([1, H, W, 1])
    mask[:, :window_size, :window_size, :] = 0
    mask[:, :window_size, window_size:window_size + shift_size, :] = 1
    mask[:, :window_size, -shift_size:, :] = 2
    mask[:, window_size:window_size + shift_size, :window_size, :] = 3
    mask[:, window_size:window_size + shift_size, window_size:window_size + shift_size, :] = 4
    mask[:, window_size:window_size + shift_size, -shift_size:, :] = 5
    mask[:, -shift_size:, :window_size, :] = 6
    mask[:, -shift_size:, window_size:window_size + shift_size, :] = 7
    mask[:, -shift_size:, -shift_size:, :] = 8
    mask = tf.constant(mask)

    return mask

class MLP(Layer):
    def __init__(self, hidden_neurons, out_neurons, act, drop_ratio):
        '''
        :param hidden_neurons: numebr of hidden-layer neurons
        :param out_neurons: number of output-layer neurons
        :param act: activation function
        :param drop_ratio: dropout ratio
        '''
        super().__init__()

        self.fc1 = Dense(hidden_neurons)
        self.activation = act
        self.fc2 = Dense(out_neurons)
        self.drop = Dropout(drop_ratio)

    def call(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Window_Attention(Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., act = None):
        '''
        :param dim: number of channels of the input feature
        :param window_size: height / width of shifted_window
        :param num_heads: number of heads in multi-head attention
        :param qkv_bias: Boolaen, whether a bias is employed on Q, V and K
        :param attn_drop: dropout ratio for attention calculation
        :param proj_drop: dropout ratio for fc layer after attention calculation
        :param act: activation function
        '''

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.att_drop = Dropout(attn_drop)
        self.out_drop = Dropout(proj_drop)

        self.qkv_fc = Dense(self.dim*3, activation = act, use_bias=qkv_bias)
        self.att_fc = Dense(self.dim, activation = act)
        self.softmax = Softmax(axis=-1)
        #  create attention bias and index table
        ## create index table
        index_X, index_Y = tf.meshgrid(
            tf.range(self.window_size),
            tf.range(self.window_size))     # (WS, WS)
        self.index_table = tf.stack((
            tf.reshape(index_X, [-1]),
            tf.reshape(index_Y, [-1])))      # (2,WS*WS)
        ## calculate relative index with range [-M+1, M-1]
        self.index_table = self.index_table[:,None,:] - self.index_table[:,:,None]              # (2,WS*WS,WS*WS)
        ## sclae to range [0, 2M-2] to prevent negative indices
        self.index_table += self.window_size-1                                                  # (2,WS*WS,WS*WS)
        ## times by 2M-1 to prevent the same sum of indices within each row
        self.index_table = tf.reduce_sum(self.index_table*(2*self.window_size-1), axis=0)       # (WS*WS,WS*WS)

        ##create bias table and initialize with truncated normal distributation
        initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=100)
        self.bias_table = tf.Variable(
            initializer(shape=((2*self.window_size-1)*(self.window_size-1), self.num_heads)))   #((2*WS-1)*(2*WS-1),nh)

    def call(self, x, mask):
        '''
        :param x: (B*NW, WS*WS, C)
        :param mask: (NW, WS*WS, WS*Ws)
        :return:  (B*NW, WS*WS, C)
        '''

        # project input x to *3 dims
        B, L, C = x.shape
        qkv = tf.transpose(self.qkv_fc(x), perm=[2,0,1])                                    # (3*C, B*NW, WS*WS)
        q, k, v = qkv[:C,:,:], qkv[C:2*C,:,:], qkv[2*C:,:,:]                                # (C, B*NW, WS*WS)

        # split q,k,v to multiplt heads
        q = tf.reshape(q, [-1, self.num_heads, L, C//self.num_heads])                        # (B*NW, nh, L, C//nh) L -> WS*WS
        k = tf.reshape(k, [-1, self.num_heads, L, C // self.num_heads])                      # (B*NW, nh, L, C//nh)
        v = tf.reshape(v, [-1, self.num_heads, L, C // self.num_heads])                      # (B*NW, nh, L, C//nh)

        # calculate q @ k.T
        qk_prod = tf.matmul(q, tf.transpose(k, perm=[0,1,3,2]))                             # (B*NW, nh, L, L)

        # add relative-position bias
        position_bias = tf.transpose(
            tf.gather(self.bias_table, self.index_table), perm=[2,0,1])                     # (nh, L, L)
        qk_prod += position_bias                                                            # (B*NW, nh, L, L)

        # mask
        NW = mask.shape[0]
        qk_prod = tf.reshape(qk_prod, shape=[-1,NW,self.num_heads,L,L]) + mask[:,None,:,:]   # (B, NW, nh, L, L)
        qk_prod = tf.reshape(qk_prod, shape=[-1,self.num_heads,L,L])                        # (B*NW, nh, L, L)
        qk_prod = self.softmax(qk_prod)
        qk_prod = self.att_drop(qk_prod)

        # calculate softmax(q@k.T)@v
        att = tf.transpose(tf.matmul(qk_prod, v), perm=[0,2,1,3])                           # (B*NW, L, nh, C//nh)
        att = tf.reshape(att, shape=[-1, L, C])                                              # (B*NW, L, C)

        # projection
        att = self.att_fc(att)
        att = self.out_drop(att)
        return att

class Swinblock(Layer):
    def __init__(self, dim, resolution, num_heads, window_size=7, shift_size=3,
                 mlp_ratio=4., qkv_bias=True, mlp_drop= 0., attn_drop=0., proj_drop=0.,
                 ff_drop=0., act_layer=Activation(tf.nn.gelu), norm_layer=LayerNormalization):
        '''
        :param dim: number of channels of the input feature
        :param resolution: resolution of inout feature
        :param num_heads: number of heads in multi-head attention
        :param window_size: size of shifted_window
        :param shift_size: stride of shifted window
        :param mlp_ratio: hidden neurons / input neurons
        :param qkv_bias: Boolaen, whether a bias is employed on Q, V and K
        :param mlp_drop: ratio of mlp dropout
        :param attn_drop: ratio of attention dropout
        :param proj_drop: ratio of projection drop
        :param ff_drop: ratio of dropout in feed-forward netowrk
        :param act_layer: ativation
        :param norm_layer: normalization
        '''
        super().__init__()
        self.norm = norm_layer()
        self.resolution = resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.att = Window_Attention(dim=dim,
                                    window_size=window_size,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=proj_drop,
                                    act=act_layer)

        self.mlp = MLP(hidden_neurons=dim*mlp_ratio, out_neurons=dim, act=act_layer, drop_ratio=mlp_drop)
        self.dropout = Dropout(ff_drop)

        # create mask
        H, W = self.resolution
        self.mask = create_mask(H, W, self.window_size, self.shift_size)                        # [1, H, W, 1]
        self.mask = window_partition(self.mask, self.window_size)                                # [1*NW, ws, ws, 1]
        self.mask = tf.reshape(self.mask, shape=[-1,self.window_size*self.window_size])         # [NW, ws*ws]
        self.mask = tf.repeat(self.mask[:, None, :], repeats=self.window_size**2, axis=1) - \
                    tf.repeat(self.mask[:, :, None], repeats=self.window_size**2, axis=2)       # [NW,ws*ws,ws*ws]
        # replace non-zero values with strong negative values to realize masking
        self.mask = tf.where(self.mask==0, 0, -100)                                             # [NW,ws*ws,ws*ws]
        self.mask = tf.cast(self.mask, tf.float32)

    def call(self, x):
        B, L, D = x.shape                                                                       # L=H*W
        H,W = self.resolution

        x_residual = x
        # layer normalization
        x = self.norm(x)                                                                        # [B,L,D]
        x = tf.reshape(x, shape=[-1, H, W, D])                                                   # [B,H,W,D]
        # SW-MSA
        ## shift feature
        x = tf.roll(x, shift=[0,-self.shift_size,-self.shift_size,0], axis=[0,1,2,3])           # [B,H,W,D]

        ## window partitation
        x = window_partition(x, self.window_size)                                               # [B*num of windows,ws,ws,D]
        x = tf.reshape(x,shape=[-1,self.window_size*self.window_size,D])                        # [B*num of windows,ws*ws,D]

        ## self-attention
        x = self.att(x, mask = self.mask)                                                       # [B*num of windows,ws*ws,D]

        ## merge windows
        x = tf.reshape(x,shape=[-1,self.window_size,self.window_size,D])                        # [B*num of windows,ws,ws,D]

        ## reverse shift
        shifted_x = window_reverse(x,self.window_size,H,W)
        x = tf.roll(shifted_x,
                    shift=[0, self.shift_size, self.shift_size, 0],
                    axis=[0, 1, 2, 3])                                                          # [B, H, W, D]
        x = tf.reshape(x, shape=[-1,H*W,D])

        ## LN & MLP
        x = x_residual + x
        x_residual = x
        x = self.mlp(self.norm(x))
        x = x_residual + x

        return x




