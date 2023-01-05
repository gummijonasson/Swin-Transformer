import tensorflow as tf
import numpy as np
#from tensorflow import keras
from tensorflow.keras import Sequential, initializers
from tensorflow.keras.layers import Dropout, LayerNormalization, Layer, Activation, Dense, Softmax
from tensorflow.nn import gelu


#Two layer MLP with GELU non-linearity inbetween
def MLP(hidden_neurons, out_neurons, drop_ratio):
    '''
    :param hidden_neurons: numebr of hidden-layer neurons
    :param out_neurons: number of output-layer neurons
    :param drop_ratio: dropout ratio
    '''
    mlp = Sequential([
        Dense(hidden_neurons),
        Activation(gelu),
        Dropout(drop_ratio),
        Dense(out_neurons),
        Dropout(drop_ratio)
    ])
    return mlp


# Inspiration from https://github.com/rwightman/pytorch-image-models/blob/a6e8598aaf90261402f3e9e9a3f12eac81356e9d/timm/models/layers/drop.py#L140
# https://paperswithcode.com/method/droppath
def drop_path(x, drop_rate):
    if drop_rate == 0:               #Not applied to testing
        return x

    keep = 1 - drop_rate
    shape = (x.shape[0],) + (1,) + (x.ndim - 1)
    random_tensor = keep + tf.random.uniform(shape, dtype=x.dtype)
    out = tf.math.divide(x, keep) * random_tensor.floor()
    return out

# Inspiration from https://keras.io/examples/vision/swin_transformers/
# https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow/blob/main/models/swin_transformer.py
# https://www.tensorflow.org/api_docs/python/tf/math/floor
def window_partition(x, window_size):
    '''
    :param x: input feature with size (-1, H, W, C)
    :param window_size: window size
    :returns portioned x: (-1*num of windows, window_size, window_size, C)
    '''
    B, H, W, C = x.shape
    x = tf.reshape(x, shape = (-1, tf.floor(H / window_size), window_size, tf.floor(W / window_size), window_size, C ))
    x = tf.transpose(x, (0,1,2,3,4,5))
    out = tf.reshape(x, shape = (-1, window_size, window_size, C))
    return out

# Inspiration from https://keras.io/examples/vision/swin_transformers/
# https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow/blob/main/models/swin_transformer.py
# https://www.tensorflow.org/api_docs/python/tf/math/floor
def window_reverse(windows, window_size, H, W):
    '''
    :param windows: (num_windows*B, window_size, window_size, C)
    :param window_size: Window size
    :param H: height of feature
    :param W: width of feature
    :returns: (B, H, W, C)
    '''
    B, ws, ws, C = windows.shape
    x = tf.reshape(windows, shape = (-1, tf.floor(H / window_size), tf.floor(W / window_size),window_size,window_size, C ))     #[B', h_nw, w_nw, ws, ws, C']
    x = tf.transpose(x, (0,1,2,3,4,5))                                                                                          #[B', h_nw, ws, w_nw, ws, C']
    out = tf.reshape(x, shape = (-1, window_size, window_size, C))                                                              #[B,H,W,C]
    return out

# Inspiration from https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow/blob/main/models/swin_transformer.py
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
class WindowAttention(Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout=0.):
        '''
        :param dim: number of channels of the input feature
        :param window_size: height / width of shifted_window
        :param num_heads: number of heads in multi-head attention
        :param qkv_bias: Boolean, whether a bias is employed on Q, V and K
        :param dropout: dropout ratio 
        '''
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.dropout = dropout

        self.qkv = Dense(self.dim*3, activation = gelu, use_bias=qkv_bias)
        
        #  create attention bias and index table
        ## create index table
        idx_X, idx_Y = tf.meshgrid(
            tf.range(self.window_size),
            tf.range(self.window_size))     # (WS, WS)
        self.idx_table = tf.stack((
            tf.reshape(idx_X, [-1]),
            tf.reshape(idx_Y, [-1])))      # (2,WS*WS)
        ## calculate relative index with range [-M+1, M-1]
        self.idx_table = self.idx_table[:,None,:] - self.idx_table[:,:,None]              # (2,WS*WS,WS*WS)
        ## sclae to range [0, 2M-2] to prevent negative indices
        self.idx_table += self.window_size-1                                                  # (2,WS*WS,WS*WS)
        ## times by 2M-1 to prevent the same sum of indices within each row
        self.idx_table = tf.reduce_sum(self.idx_table*(2*self.window_size-1), axis=0)       # (WS*WS,WS*WS)

        ##create bias table and initialize with truncated normal distributation
        initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=100)
        self.bias_table = tf.Variable(
            initializer(shape=((2*self.window_size-1)*(self.window_size-1), self.num_heads)))   #((2*WS-1)*(2*WS-1),nh)

        self.linear = Dense(self.dim, activation = gelu)

    def call(self, x):
        '''
        :param x: (-1*NW, WS*WS, C)
        :returns:  (-1*NW, WS*WS, C)
        '''

        # project input x to *3 dims
        _, S, C = x.shape       #S=size, C=Channels
        qkv = tf.transpose(self.qkv(x), perm=[2,0,1])                                    # (3*C, -1*NW, WS*WS)
        q, k, v = qkv[:C,:,:], qkv[C:2*C,:,:], qkv[2*C:,:,:]                                # (C, -1*NW, WS*WS)

        # split q,k,v to multiplt heads
        q = tf.reshape(q, shape=(-1, self.num_heads, S, tf.floor(C / self.num_heads)))                        # (-1*NW, nh, S, C//nh) S -> WS*WS
        k = tf.reshape(k, shape=(-1, self.num_heads, S, tf.floor(C / self.num_heads)))                      # (-1*NW, nh, S, C//nh)
        v = tf.reshape(v, shape=(-1, self.num_heads, S, tf.floor(C / self.num_heads)))                      # (-1*NW, nh, S, C//nh)

        # calculate q @ k.T
        k_T = tf.transpose(k, perm=[0,1,3,2])
        qk_prod = tf.matmul(q, k_T)                             # (B*NW, nh, S, S)

        # add relative-position bias
        position_bias = tf.transpose(tf.gather(self.bias_table, self.idx_table), perm=[2,0,1])                     # (nh, S, S)
        qk_prod += position_bias

        #NO MASK

        qk_prod = Softmax(axis=-1)(qk_prod)
        qk_prod = Dropout(self.dropout)(qk_prod)

        # calculate softmax(q@k.T)@v
        attention = tf.transpose(tf.matmul(qk_prod, v), perm=[0,2,1,3])                           # (-1*NW, S, nh, C//nh)
        attention = tf.reshape(attention, shape=(-1, S, C))                                              # (-1*NW, S, C)

        # projection
        attention = self.linear(attention)
        attention = self.linear(attention)
        return attention


## No attention mask
## Attention windows with no mask 
# Inspiration from https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow/blob/main/models/swin_transformer.py
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
# https://keras.io/examples/vision/swin_transformers/
class NoSwinblock(Layer):
    def __init__(self, dim, resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_ratio= 0.):
        '''
        :param dim: number of channels of the input feature
        :param resolution: resolution of inout feature
        :param num_heads: number of heads in multi-head attention
        :param window_size: size of shifted_window
        :param mlp_ratio: hidden neurons / input neurons
        :param qkv_bias: Boolaen, whether a bias is employed on Q, V and K
        :param drop_ratio: dropout ratio
        '''
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.resolution = resolution
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_ratio = drop_ratio

        self.layer_norm = LayerNormalization(epsilon=1e-5)
        self.window_attention = WindowAttention(self.dim, self.window_size, self.num_heads, self.qkv_bias, self.drop_ratio)
        self.mlp = MLP(hidden_neurons=(self.dim * self.mlp_ratio), out_neurons=self.dim, drop_ratio=self.drop_ratio)

    def call(self, x):
        B, S, C = x.shape                #S=Size, C=Channels
        H, W = self.resolution             #Size = Height*Width
        
        #Save skip
        skip_x = x

        #LayerNorm
        x = self.layer_norm(x)         #[B,S,C]
        x = tf.reshape(x, shape=(-1, H, W, C))          #[B,H,W,D]

        #NO SHIFT

        #Window partition
        x = window_partition(x, self.window_size)                       #[B*num of windows,ws*ws,C]
        x = tf.reshape(x, shape=(-1, self.window_size ** 2, C))         #[B*num of windows,ws*ws,C]

        #Window attention
        x = self.window_attention(x)      #[B*num of windows,ws*ws,C]
        x = tf.reshape(x, shape=(-1, self.window_size, self.window_size, C))                                    #[B*num of windows,ws*ws,C], Reshape the windows after self-attention

        #Window reverse
        x = window_reverse(x, self.window_size, H, W)
        x = tf.reshape(x, shape=(-1, H * W, C))                         #[B,H,W,C]

        #Drop Path
        ##x = drop_path(x)

        #Skip connection
        x = x + skip_x
        skip_x = x

        #LayerNorm 
        x = self.layer_norm(x)

        #MLP
        x = self.mlp(x)

        #Drop Path
        ##x = drop_path(x)

        #Skip connection
        x = skip_x + x

        return x