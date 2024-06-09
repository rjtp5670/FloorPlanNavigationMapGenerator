import tensorflow as tf
import numpy as np
import argparse

from typing import List
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras import layers, Sequential

def data_loader_bd_rm_from_tfrecord(batch_size=1): # TF Record 형식의 데이터셋에서 배치 단위로 데이터를 로드하는 함수.
	paths = open('/content/drive/MyDrive/Colab Notebooks/create_tfrecords/dataset/r3d_train_temp2.txt', 'r').read().splitlines()
	loader_dict = read_bd_rm_record('/content/drive/MyDrive/Colab Notebooks/create_tfrecords/dataset/newyork_train.tfrecords', batch_size=batch_size, size=applied_size)
	num_batch = len(paths) // batch_size
	return loader_dict, num_batch

GPU_ID = '0'

def _max_pool2d(tensor, size = 2, stride = 2):
  size = (2,2)
  stride = stride if isinstance(stride, (tuple, list)) else [stride, stride]
  x = layers.MaxPool2D(pool_size=size, strides=stride)(tensor)
  return x

def _upconv2d(tensor, dim, size=4, stride=2, pad='SAME', act='relu', name='upconv'):

  [batch_size, h, w, in_dim] = tensor.shape.as_list()
  size = size if isinstance(size, (tuple, list)) else [size, size]
  stride = stride if isinstance(stride, (tuple, list)) else [stride, stride]
  kernel_shape = [size[0], size[1], in_dim, dim]

  W = tf.keras.initializers.he_normal()

  if pad == 'SAME':
    out_shape = [batch_size, h*stride[0], w*stride[1], dim]
  else:
    out_shape = [batch_size, (h-1)*stride[1]+size[0],
                (w-1)*stride[2]+size[1], dim]
    print("NAME=",out_shape)

  upconv = layers.Conv2DTranspose(dim, size, strides=stride, padding=pad, kernel_initializer=W, activation=None, name=name)

  if act == 'relu':
    x = layers.Activation('relu', name=name + '/relu')(upconv(tensor))
  elif act == 'linear':
    x = upconv(tensor)

  x.set_shape(out_shape)
  return x

def _conv2d (tensor, dim, size=3, stride=1, pad='SAME', act='relu', norm='none', G=16, bias=True, name='conv', dtype=tf.float32):
  size = size if isinstance(size, (tuple, list)) else [size, size]
  stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]

  # Xavier (He) Initialization
  initializer = tf.keras.initializers.he_normal()

  conv_layer = layers.Conv2D(dim, kernel_size=size,
                             strides=stride[2:], padding=pad,
                             use_bias=bias, kernel_initializer=initializer, name=name + '/conv'
                             )
  x = conv_layer(tensor)

  if act == 'relu':
    x = layers.Activation(act, name=name + '/relu')(x)
  elif act == 'linear':
    x = layers.Activation(activation=None)(x)

  return x

def _up_bilinear(tensor, dim, shape, name='upsample'):
  out = _conv2d(tensor, dim=dim, size=1, act='linear', name=name+'/1x1_conv')
  return tf.image.resize(out, shape)

# following three function used for combining context features
def _constant_kernel(shape, value=1.0, diag=False, flip=False, regularizer=None, trainable=None, name=None):
  name = 'fixed_w' if name is None else name+'/fixed_w'
  dtype = tf.float32

  with tf.device('/device:GPU:'+'0'):
    if not diag: # Diagonal 인경우
      k = tf.Variable(initial_value=tf.constant(value, shape=shape, dtype=tf.float32),
                      shape=shape, dtype=dtype, name=name,
                      trainable=trainable, regularizer=regularizer)
    else:
      w = tf.eye(shape[0], num_columns=shape[1])
      if flip:
        w = tf.reshape(w, (shape[0], shape[1], 1))
        w = tf.image.flip_left_right(w)
      w = tf.reshape(w, shape)
      k = tf.Variable(initial_value=w, dtype=dtype, name=name,
                      trainable=trainable, regularizer=regularizer)
  return k

def _context_conv2d(tensor, dim=1, size=7, diag=False, flip=False, stride=1, name='cconv'):
  """
  Implement using identity matrix, combine neighbour pixels without bias, current only accept depth 1 of input tensor

  Args:
    diag: create diagnoal identity matrix
    transpose: transpose the diagnoal matrix
  """
  in_dim = tensor.shape.as_list()[-1] # suppose to be 1
  size = size if isinstance(size, (tuple, list)) else [size, size]
  stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
  kernel_shape = [size[0], size[1], in_dim, dim]

  out = _conv2d(tensor, dim=dim, stride=stride, pad='SAME', name=name)

  return out

def _non_local_context(tensor1, tensor2, stride=4, name='non_local_context'): # Spatial Contextual Feature

    """Use 1/stride image size of identity one rank kernel to combine context features, default is half image size, embedding between encoder and decoder part
    Args:
    stride: define the neighbour size
    """

    assert tensor1.shape.as_list() == tensor2.shape.as_list(), "input tensor should have same shape"

    [N, H, W, C] = tensor1.shape.as_list()

    hs = H // stride if (H // stride) > 1 else (stride-1)
    vs = W // stride if (W // stride) > 1 else (stride-1)

    hs = hs if (hs%2!=0) else hs+1
    vs = hs if (vs%2!=0) else vs+1

    # compute attention map
    a = _conv2d(tensor1, dim=C, name=name+'/fa1') # Room Boundary
    a = _conv2d(a, dim=C, name=name+'fa2') # Room Boundary
    a = layers.Activation('sigmoid')(_conv2d(a, dim=1, size=1, act='linear', norm=None, name=name+'/a'))

    # reduce the tensor depth
    x = _conv2d(tensor2, dim=C, name=name+'/fx1') # Room Type
    x = _conv2d(x, dim=1, size=1, act='linear', norm=None, name=name+'/x')

    # Multiply Attention Weights & Room Type Features
    x = a * x

    # Compute Direction Kernels
    h = _context_conv2d(x, size=[hs, 1], name=name+'/cc_h') # h
    v = _context_conv2d(x, size=[1, vs], name=name+'/cc_v') # v
    d1 = _context_conv2d(x, size=[hs, vs], diag=True, name=name+'/cc_d1') # d
    d2 = _context_conv2d(x, size=[hs, vs], diag=True, flip=True, name=name+'/cc_d2') # d_t

    # Compute Direction Kernels preventing blurring
    c1 = a*(h+v+d1+d2)

    # expand to dim
    c1 = _conv2d(c1, dim=C, size=1, act='linear', norm=None, name=name+'/expand')

    # further convolution to learn richer feature
    features = tf.concat([tensor2, c1], axis=3, name=name+'/in_context_concat')
    out = _conv2d(features, dim=C, name=name+'/conv2')

    # return out, a
    return out, None

class GRNLayer(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(GRNLayer, self).__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.channels,),
            initializer="zeros",
            trainable=True,
            name='gamma'  # 명시적 이름 지정
        )
        self.beta = self.add_weight(
            shape=(self.channels,),
            initializer="zeros",
            trainable=True,
            name='beta'  # 명시적 이름 지정
        )

    def call(self, inputs):
        norm = tf.norm(inputs, axis=-1, keepdims=True)
        scaled = inputs * (self.gamma * norm + self.beta)
        return scaled

class LayerScale(layers.Layer):
    def __init__(self, init_values: float, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',  # 명시적 이름 지정
            shape=(self.projection_dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True
        )

    def call(self, x, training=False):
        return x * self.gamma

def _convnext_block_v2(tensor, dim, size=7, stride=1, pad='SAME', act='gelu', norm='ln', G=1, bias=False, drop_path_rate=0.3, name='convnext', first_block=False):
    if first_block:
        # Adjust the channel dimension to match the block's dimension
        tensor = layers.Conv2D(dim, (1, 1), strides=(1, 1), padding="same", name=name + '/adjust_dims')(tensor)

    if norm == 'ln':
        x = layers.LayerNormalization(epsilon=1e-6, name=name + '/ln')(tensor)
    else:
        x = tensor

    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size=size, strides=stride, padding=pad, depth_multiplier=1,
                               use_bias=bias, name=name + '/depthwise_conv')(x)

    # Pointwise convolution to expand channels
    x = layers.Conv2D(4 * dim, 1, padding='valid', use_bias=bias, name=name + '/pwconv1')(x)

    # Apply GELU activation
    if act == 'gelu':
        x = tf.nn.gelu(x, approximate=True, name=name + '/gelu')

    # x = tf.nn.relu(x, name=name + '/relu')

    # Apply Global Response Normalization
    x = GRNLayer(4 * dim, name=name + '/grn')(x)

    # Second pointwise convolution to project channels back to original dimension
    x = layers.Conv2D(dim, 1, padding='valid', use_bias=bias, name=name + '/pwconv2')(x)

    if drop_path_rate > 0.0:
        x = layers.Dropout(rate=drop_path_rate)(x, training=True)

    # Skip connection and addition
    output = layers.Add(name=name + '/add')([tensor, x])
    return output

def _convnext_block_v1(tensor, dim, size=7, stride=1, pad='SAME', act='gelu', norm='ln', bias=False, drop_path_rate=0.0, name='convnext', first_block=False, block_id=0):
    layer_name_prefix = f"{name}_{block_id}"

    if first_block:
        # 차원 맞추기 위한 조건적 조정
        tensor = layers.Conv2D(dim, (1, 1), strides=(1, 1), padding="same", name=f"{layer_name_prefix}/adjust_dims")(tensor)

    if norm == 'ln':
        x = layers.LayerNormalization(epsilon=1e-6, name=f"{layer_name_prefix}/ln")(tensor)
    else:
        x = tensor

    x = layers.DepthwiseConv2D(kernel_size=size, strides=stride, padding=pad, depth_multiplier=1,
                               use_bias=bias, name=f"{layer_name_prefix}/depthwise_conv")(x)

    x = layers.Conv2D(4 * dim, 1, padding='valid', use_bias=bias, name=f"{layer_name_prefix}/pwconv1")(x)

    if act == 'gelu':
        x = tf.nn.gelu(x, approximate=True, name=f"{layer_name_prefix}/gelu")

    # x = tf.nn.relu(x, name=name + '/relu')

    x = layers.Conv2D(dim, 1, padding='valid', use_bias=bias, name=f"{layer_name_prefix}/pwconv2")(x)

    if drop_path_rate > 0.0:
        x = layers.Dropout(rate=drop_path_rate, name=f"{layer_name_prefix}/dropout")(x, training=True)

    # LayerScale application
    x = LayerScale(init_values=1e-6 , projection_dim=dim, name=f"{layer_name_prefix}/layer_scale")(x)

    output = layers.Add(name=f"{layer_name_prefix}/add")([tensor, x])  # Skip connection 적용

    return output

def apply_masks(input_tensor, mask, mask_token):
    shape = tf.shape(input_tensor)
    n, h, w, c = shape[0], shape[1], shape[2], shape[3]
    mask = tf.image.resize(mask, [h, w])
    mask = tf.tile(tf.expand_dims(mask, axis=0), [n, 1, 1, 1])
    mask_token = tf.tile(mask_token[:, :, :, :c], [n, h, w, 1])
    return input_tensor * (1.0 - mask) + mask_token * mask

def create_masks(height, width, mask_ratio=0.6, patch_size=32):
    num_patches = (height // patch_size) * (width // patch_size)

    def mask_single_image(_):
        mask = tf.random.uniform((num_patches,), 0, 1, dtype=tf.float32) < mask_ratio
        mask = tf.reshape(mask, [height // patch_size, width // patch_size])
        mask = tf.cast(mask, tf.float32)
        mask = tf.image.resize(mask[..., tf.newaxis], [height, width], method='nearest')
        return mask

    return mask_single_image

class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, mask_token, **kwargs):
        super(MaskLayer, self).__init__(**kwargs)
        self.mask_token = mask_token

    def call(self, inputs, masks):
        return apply_masks(inputs, masks, self.mask_token)

def build_f_net(inputs, model_type='pico') -> List[tf.Tensor]:
    if model_type == 'atto':
        depths = [2, 2, 6, 2]
        dims = [40, 80, 160, 320]
    elif model_type == 'pico':
        depths = [2, 2, 8, 2]
        dims = [80, 160, 320, 640]
    else:
        raise ValueError("Unsupported model type. Choose either 'atto' or 'pico'.")

    # Feature Extraction (Encoder)
    # Stage 1
    conv1 = _convnext_block_v2(inputs, dim=dims[0], name='encoder_stage1_1', first_block=True)
    conv1 = _convnext_block_v2(conv1, dim=dims[0], name='encoder_stage1_2', first_block=False)
    if model_type == 'pico' and depths[0] > 2:
        conv1 = _convnext_block_v2(conv1, dim=dims[0], name='encoder_stage1_3', first_block=False)
    conv1 = _conv2d(conv1, dim=64, act='relu', name='conv1_1')
    pool1 = _max_pool2d(conv1)

    # Stage 2
    conv2 = _convnext_block_v2(pool1, dim=dims[1], name='encoder_stage2_1', first_block=True)
    conv2 = _convnext_block_v2(conv2, dim=dims[1], name='encoder_stage2_2', first_block=False)
    if model_type == 'pico' and depths[1] > 2:
        conv2 = _convnext_block_v2(conv2, dim=dims[1], name='encoder_stage2_3', first_block=False)
    conv2 = _conv2d(conv2, dim=128, act='relu', name='conv2_1')
    pool2 = _max_pool2d(conv2)

    # Stage 3
    conv3 = _convnext_block_v2(pool2, dim=dims[2], name='encoder_stage3_1', first_block=True)
    conv3 = _convnext_block_v2(conv3, dim=dims[2], name='encoder_stage3_2', first_block=False)
    conv3 = _convnext_block_v2(conv3, dim=dims[2], name='encoder_stage3_3', first_block=False)
    conv3 = _convnext_block_v2(conv3, dim=dims[2], name='encoder_stage3_4', first_block=False)
    conv3 = _convnext_block_v2(conv3, dim=dims[2], name='encoder_stage3_5', first_block=False)
    conv3 = _convnext_block_v2(conv3, dim=dims[2], name='encoder_stage3_6', first_block=False)
    if model_type == 'pico':
        for i in range(7, depths[2] + 1):
            conv3 = _convnext_block_v2(conv3, dim=dims[2], name=f'encoder_stage3_{i}', first_block=False)
    conv3 = _conv2d(conv3, dim=256, act='relu', name='conv3_1')
    pool3 = _max_pool2d(conv3)

    # Stage 4
    conv4 = _convnext_block_v2(pool3, dim=dims[3], name='encoder_stage4_1', first_block=True)
    conv4 = _convnext_block_v2(conv4, dim=dims[3], name='encoder_stage4_2', first_block=False)
    if model_type == 'pico' and depths[3] > 2:
        conv4 = _convnext_block_v2(conv4, dim=dims[3], name='encoder_stage4_3', first_block=False)
    conv4 = _conv2d(conv4, dim=512, act='relu', name='conv4_1')
    pool4 = _max_pool2d(conv4)

    # Stage 5
    conv5 = _conv2d(pool4, dim=512, act='relu', name='conv5_1')
    conv5 = _conv2d(conv5, dim=512, act='relu', name='conv5_2')
    pool5 = _max_pool2d(conv5)

    pools = [pool1, pool2, pool3, pool4, pool5]

    return pools

def build_cw_net(pools):
    # Room Boundary (Decoder)
    pool1, pool2, pool3, pool4, pool5 = pools

    up2 = (_upconv2d(pool5, dim=256, act='linear', name='up2_1') +
           _conv2d(pool4, dim=256, act='linear', name='pool4_s'))
    up2_cw = _conv2d(up2, dim=256, act='linear', name='up2_3')

    up4 = (_upconv2d(up2_cw, dim=128, act='linear', name='up4_1') +
           _conv2d(pool3, dim=128, act='linear', name='pool3_s'))
    up4_cw = _conv2d(up4, dim=128, act='linear', name='up4_3')

    up8 = (_upconv2d(up4_cw, dim=64, act='linear', name='up8_1') +
           _conv2d(pool2, dim=64, act='linear', name='pool2_s'))
    up8_cw = _conv2d(up8, dim=64, act='linear', name='up8_2')

    up16 = (_upconv2d(up8_cw, dim=32, act='linear', name='up16_1') +
            _conv2d(pool1, dim=32, act='linear', name='pool1_s'))
    up16_cw = _conv2d(up16, dim=32, act='linear', name='up16_2')

    return [up2_cw, up4_cw, up8_cw, up16_cw]

def build_r_net(pools, cw):
    # Room Types (Decoder)
    pool1, pool2, pool3, pool4, pool5 = pools
    up2_cw, up4_cw, up8_cw, up16_cw = cw

    up2 = (_upconv2d(pool5, dim=256, act='linear', name='up2_1r') +
           _conv2d(pool4, dim=256, act='linear', name='pool4_r'))
    up2 = _conv2d(up2, dim=256, name='up2_r', act='relu')
    up2, _ = _non_local_context(up2_cw, up2, name='context_up2')

    up4 = (_upconv2d(up2, dim=128, act='linear', name='up4_1r') +
           _conv2d(pool3, dim=128, act='linear', name='pool3_r'))
    up4 = _convnext_block_v1(up4, dim=128, name='up4_2r')
    up4, _ = _non_local_context(up4_cw, up4, name='context_up4')

    up8 = (_upconv2d(up4, dim=64, act='linear', name='up8_1r') +
           _conv2d(pool2, dim=64, act='linear', name='pool2_r'))
    up8 = _convnext_block_v1(up8, dim=64, name='up8_2r')
    up8, _ = _non_local_context(up8_cw, up8, name='context_up8')

    up16 = (_upconv2d(up8, dim=32, act='linear', name='up16_1r') +
            _conv2d(pool1, dim=32, act='linear', name='pool1_r'))
    up16 = _convnext_block_v1(up16, dim=32, name='up16_2r')
    up16_r, _ = _non_local_context(up16_cw, up16, name='context_up16')

    return [up16_cw, up16_r]

applied_size = 256

def deepfloorplanModel(config: argparse.Namespace = None, dtype=tf.float32):
    input_shape = [applied_size, applied_size, 3]
    inputs = layers.Input(shape=input_shape, name='input_layer')

    # Create mask function and generate masks
    mask_function = create_masks(input_shape[0], input_shape[1], mask_ratio=0.6, patch_size=32)
    masks = mask_function(0)

    # Apply masks to the inputs
    masked_inputs = inputs * masks

    # Create mask token with the maximum number of channels
    max_channels = max([64, 128, 256, 512])
    mask_token = tf.Variable(tf.zeros((1, 1, 1, max_channels), dtype=dtype), trainable=True, name="mask_token")

    # Build encoder (feature extraction)
    pools = build_f_net(masked_inputs)

    # Apply masks to the encoder output before passing to the decoders
    masked_pools = [MaskLayer(mask_token)(pool, masks) for pool in pools]

    # Build decoders
    cw = build_cw_net(pools)

    up16_cw, up16_r = build_r_net(masked_pools, cw)

    # Compute logits
    logits_cw = _up_bilinear(up16_cw, dim=3, shape=(applied_size, applied_size), name='logits_cw')
    logits_r = _up_bilinear(up16_r, dim=2, shape=(applied_size, applied_size), name='logits_r')
    logits = [logits_cw, logits_r]

    # Create model
    model = Model(inputs=masked_inputs, outputs=logits, name="DEEP_NET")
    return model, masks