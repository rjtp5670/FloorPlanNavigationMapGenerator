import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, models, Input, Sequential
from keras import layers
import tensorflow_addons as tfa
import numpy as np
import os
import argparse
import keras
import os
from typing import List

GPU_ID = '0'

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # 현재 프로그램이 필요할 때만 메모리를 할당하도록 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 프로그램 시작 후에 GPU 설정을 변경할 수 없을 때 발생
        print(e)
# mixed_precision 모듈에서 set_global_policy 함수를 임포트합니다

from tensorflow.keras.mixed_precision import set_global_policy

### 전역 정책으로 'mixed_float16'을 사용하도록 설정합니다
# set_global_policy('mixed_float16')

class GRNLayer(layers.Layer):
    def __init__(self, channels, **kwargs):
        super(GRNLayer, self).__init__(**kwargs)
        self.channels = channels
        self.gamma = self.add_weight(shape=(channels,), initializer="zeros", trainable=True)
        self.beta = self.add_weight(shape=(channels,), initializer="zeros", trainable=True)

    def call(self, inputs):
        norm = tf.norm(inputs, axis=-1, keepdims=True)
        scaled = inputs * (self.gamma * norm + self.beta)
        return scaled

class LayerScale(layers.Layer):
    def __init__(self, init_values: float, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.gamma = self.add_weight(
            shape=(projection_dim,),
            initializer=keras.initializers.Constant(init_values),
        )

    def call(self, x, training=False):
        return x * self.gamma

def create_masks(height, width, mask_ratio, patch_size):
    num_patches = (height // patch_size) * (width // patch_size)
    num_masked = int(mask_ratio * num_patches)

    def mask_single_image(_):
        mask = tf.random.uniform((num_patches,), 0, 1, dtype=tf.float32) < mask_ratio
        mask = tf.reshape(mask, [height // patch_size, width // patch_size])
        mask = tf.cast(mask, tf.float32)
        mask = tf.image.resize(mask[..., tf.newaxis], [height, width], method='nearest')
        return mask

    return mask_single_image

def _convnext_block_v2(tensor, dim, size=7, stride=1, pad='SAME', act='none', norm='linear', G=1, bias=False, drop_path_rate=0.0, name='convnext', first_block=False):
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
    
    # Apply Global Response Normalization
    x = GRNLayer(4 * dim, name=name + '/grn')(x)
    
    # Second pointwise convolution to project channels back to original dimension
    x = layers.Conv2D(dim, 1, padding='valid', use_bias=bias, name=name + '/pwconv2')(x)

    if drop_path_rate > 0.0:
        x = layers.Dropout(rate=drop_path_rate)(x, training=True)

    # Skip connection and addition
    output = layers.Add(name=name + '/add')([tensor, x])
    return output

def _convnext_block_v1(tensor, dim, size=7, stride=1, pad='SAME', act='none', norm='linear', bias=False, drop_path_rate=0.0, name='convnext', first_block=False, block_id=0):
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
    
    x = layers.Conv2D(dim, 1, padding='valid', use_bias=bias, name=f"{layer_name_prefix}/pwconv2")(x)
    
    if drop_path_rate > 0.0:
        x = layers.Dropout(rate=drop_path_rate, name=f"{layer_name_prefix}/dropout")(x, training=True)
    
    if first_block:
        output = x  # 첫 번째 블록이면 차원 조정 레이어가 이미 적용되었으므로 바로 반환
    else:
        output = layers.Add(name=f"{layer_name_prefix}/add")([tensor, x])  # Skip connection 적용

    return output

def data_loader_bd_rm_from_tfrecord(batch_size=1): # TF Record 형식의 데이터셋에서 배치 단위로 데이터를 로드하는 함수.
	paths = open('/content/drive/MyDrive/Colab Notebooks/create_tfrecords/dataset/r3d_train_temp2.txt', 'r').read().splitlines()
	loader_dict = read_bd_rm_record('/content/drive/MyDrive/Colab Notebooks/create_tfrecords/dataset/newyork_train.tfrecords', batch_size=batch_size, size=512)
	num_batch = len(paths) // batch_size
	return loader_dict, num_batch

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

def build_fcmae_encoder(input_shape=(512, 512, 3)):
    input_tensor = Input(shape=input_shape, name='encoder_input')
    x = input_tensor

    # Stage 1
    x = _convnext_block_v2(x, dim=40, name='encoder_stage1_1', first_block=True)
    x_1 = _convnext_block_v2(x, dim=40, name='encoder_stage1_2', first_block=False)

    # Stage 2
    x = _convnext_block_v2(x_1, dim=80, name='encoder_stage2_1', first_block=True)
    x_2 = _convnext_block_v2(x, dim=80, name='encoder_stage2_2', first_block=False)

    # Stage 3
    x = _convnext_block_v2(x_2, dim=160, name='encoder_stage3_1', first_block=True)
    x = _convnext_block_v2(x, dim=160, name='encoder_stage3_2', first_block=False)
    x = _convnext_block_v2(x, dim=160, name='encoder_stage3_3', first_block=False)
    x = _convnext_block_v2(x, dim=160, name='encoder_stage3_4', first_block=False)
    x = _convnext_block_v2(x, dim=160, name='encoder_stage3_5', first_block=False)
    x_3 = _convnext_block_v2(x, dim=160, name='encoder_stage3_6', first_block=False)

    # Stage 4
    x = _convnext_block_v2(x_3, dim=320, name='encoder_stage4_1', first_block=True)
    x_4 = _convnext_block_v2(x, dim=320, name='encoder_stage4_2', first_block=False)

    # Dimension reduction convolution
    x_5 = layers.Conv2D(512, (1, 1), padding='same', name='dim_reduction')(x_4)
    # x = [x_1, x_2, x_3, x_4]
    return models.Model(inputs=input_tensor, outputs=x_5, name='ConvNext_encoder')

def build_cw_fcmae_decoder(input_tensor, name='cw_decoder'):
    b1 = _convnext_block_v1(input_tensor, dim=512, block_id=1, first_block=True)  
    b2 = _convnext_block_v1(b1, dim=320, block_id=2, first_block=True)  
    b3 = _convnext_block_v1(b2, dim=160, block_id=3, first_block=True)  
    b4= _convnext_block_v1(b3, dim=80, block_id=4, first_block=True)  
    b5= _convnext_block_v1(b4, dim=40, block_id=5, first_block=True)  
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(b5)
    return models.Model(inputs=input_tensor, outputs=decoded, name='CW_Decoder'), [b1, b2, b3, b4, b5]

def build_r_fcmae_decoder(input_tensor, share_blocks, name='r_decoder'):
    b1, b2, b3, b4, b5 = share_blocks
    x = _convnext_block_v1(input_tensor, dim=512, block_id=11, first_block=True)
    x = layers.Add()([x, b1])
    x = _convnext_block_v1(x, dim=320, block_id=22, first_block=True)
    x = layers.Add()([x, b2])
    x = _convnext_block_v1(x, dim=160, block_id=33, first_block=True)
    x = layers.Add()([x, b3])
    x = _convnext_block_v1(x, dim=80, block_id=44, first_block=True)
    x = layers.Add()([x, b4])
    x = _convnext_block_v1(x, dim=40, block_id=55, first_block=True)
    x = layers.Add()([x, b5])
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return models.Model(inputs=input_tensor, outputs=decoded, name='R_Decoder')

def deepfloorplanModel(input_shape=[512, 512, 3], mask_ratio=0.6, config: argparse.Namespace = None):
    input_tensor = Input(shape=input_shape, name='input_layer')
    
    # Mask 생성 및 적용
    mask_function = create_masks(input_shape[0], input_shape[1], mask_ratio, patch_size=32)
    masks = mask_function(0)
    masked_img = input_tensor * masks
    
    base = {"depths" : [3, 3, 9, 3], "dims" : [96, 192, 384, 768]}
    pico = {"depths" :[2, 2, 6, 2], "dims" : [40, 80, 160, 320]}
    nano = {"depths" :[1], "dims" : [96]}
    depths = nano["depths"]
    dims = nano["dims"]

    # Encoder
    encoder = build_fcmae_encoder(input_shape=input_shape)
    encoded_features = encoder(masked_img)

    # CW Decoder
    cw_decoder_model, share_blocks = build_cw_fcmae_decoder(encoded_features, name="cw_decoder")
    logits_cw = cw_decoder_model(encoded_features)
    logits_cw = _up_bilinear(logits_cw, dim=3, shape=(512, 512), name='logits_cw')

    # R Decoder
    r_decoder_model = build_r_fcmae_decoder(encoded_features, share_blocks, name="r_decoder")
    logits_r = r_decoder_model(encoded_features)
    logits_r = _up_bilinear(logits_r, dim=4, shape=(512, 512), name='logits_r')

    # 최종 모델 구성
    model = Model(inputs=input_tensor, outputs=[logits_cw, logits_r], name="DeepFloorPlanNet")
    model.summary() 
    
    # model.compile(optimizer='adam', loss={'cw': 'binary_crossentropy', 'r': 'categorical_crossentropy'},
    #               loss_weights={'cw': 1.0, 'r': 1.0})
    
    return model

# 데이터 로딩 및 모델 학습 부분에서 배치 크기를 적용
# 예: dataset.batch(batch_size)
import gc
# 학습 전에 가비지 컬렉션 실행
gc.collect()

# TensorFlow에서 GPU 메모리 사용량을 관리하도록 설정
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 필요한 만큼만 메모리를 사용하도록 설정
session = tf.compat.v1.Session(config=config)

img = tf.random.uniform(shape=(1,512,512,3), minval=0, maxval=255, dtype=tf.float32)
model = deepfloorplanModel()
# model.summary()
# logits_cw, logits_r = model(img)
