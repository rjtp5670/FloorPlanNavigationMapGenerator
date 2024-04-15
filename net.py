import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import argparse

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

# 전역 정책으로 'mixed_float16'을 사용하도록 설정합니다
set_global_policy('mixed_float16')

def data_loader_bd_rm_from_tfrecord(batch_size=1): # TF Record 형식의 데이터셋에서 배치 단위로 데이터를 로드하는 함수.
	paths = open('/content/drive/MyDrive/Colab Notebooks/create_tfrecords/dataset/r3d_train_temp2.txt', 'r').read().splitlines()
	loader_dict = read_bd_rm_record('/content/drive/MyDrive/Colab Notebooks/create_tfrecords/dataset/newyork_train.tfrecords', batch_size=batch_size, size=512)
	num_batch = len(paths) // batch_size
	return loader_dict, num_batch

GPU_ID = '0'

from tensorflow.keras import layers, Sequential

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

from typing import List

def build_f_net(inputs) -> List[tf.Tensor]:
  # Feature Extraction (Encoder)
  conv1_1 = _conv2d(inputs, dim=64, act='relu', name='conv1')
  conv1_2 = _conv2d(conv1_1, dim=64, act='relu', name='conv2')
  pool1   = _max_pool2d(conv1_2)

  conv2_1 = _conv2d(pool1, dim=128, act='relu', name='conv2_1')
  conv2_2 = _conv2d(conv2_1, dim=128, act='relu', name='conv2_2')
  pool2   = _max_pool2d(conv2_2)

  conv3_1 = _conv2d(pool2, dim=256, act='relu', name='conv3_1')
  conv3_2 = _conv2d(conv3_1, dim=256, act='relu', name='conv3_2')
  conv3_3 = _conv2d(conv3_2, dim=256, act='relu', name='conv3_3')
  pool3   = _max_pool2d(conv3_3)

  conv4_1 = _conv2d(pool3, dim=512, act='relu', name='conv4_1')
  conv4_2 = _conv2d(conv4_1, dim=512, act='relu', name='conv4_2')
  conv4_3 = _conv2d(conv4_2, dim=512, act='relu', name='conv4_3')
  pool4   = _max_pool2d(conv4_3)

  conv5_1 = _conv2d(pool4, dim=512, act='relu', name='conv5_1')
  conv5_2 = _conv2d(conv5_1, dim=512, act='relu', name='conv5_2')
  conv5_3 = _conv2d(conv5_2, dim=512, act='relu', name='conv5_3')
  pool5   = _max_pool2d(conv5_3)

  pools = [pool1, pool2, pool3, pool4, pool5]

  return pools

def build_cw_net(pools):
  # Room Boundary (Decoder)
  pool1, pool2, pool3, pool4, pool5 = pools

  up2 = (_upconv2d(pool5, dim=256, act = 'linear', name= 'up2_1')
          + _conv2d(pool4, dim=256, act='linear', name='pool4_s'))
  up2_cw = _conv2d(up2, dim=256, act='linear',  name='up2_3')

  up4 = (_upconv2d(up2_cw, dim=128, act = 'linear', name= 'up4_1')
          + _conv2d(pool3, dim=128, act='linear', name='pool3_s'))
  up4_cw = _conv2d(up4, dim=128, act = 'linear', name= 'up4_3')

  up8 = (_upconv2d(up4_cw, dim=64, act = 'linear', name='up8_1')
          + _conv2d(pool2, dim=64, act='linear', name='pool2_s'))
  up8_cw = _conv2d(up8, dim=64, act = 'linear', name= 'up8_2')

  up16 = (_upconv2d(up8_cw, dim=32, act = 'linear', name= 'up16_1')
          + _conv2d(pool1, dim=32, act='linear', name='pool1_s'))
  up16_cw = _conv2d(up16, dim=32, act = 'linear', name= 'up16_2')

  return [up2_cw, up4_cw, up8_cw, up16_cw]

def build_r_net(pools, cw):
  # Room Types (Decoder)
  pool1, pool2, pool3, pool4, pool5 = pools
  up2_cw, up4_cw, up8_cw, up16_cw = cw

  up2 = (_upconv2d(pool5, dim=256, act = 'linear', name= 'up2_1r') +
        _conv2d(pool4, dim=256, act='linear', name='pool4_r'))
  up2 = _conv2d(up2, dim=256, act = 'relu', name='up2_2r')
  up2, _ = _non_local_context(up2_cw, up2, name='context_up2')

  up4 = (_upconv2d(up2, dim=128, act = 'linear', name= 'up4_1r') +
        _conv2d(pool3, dim=128, act='linear', name='pool3_r'))
  up4 = _conv2d(up4, dim=128, act = 'relu', name='up4_2r')
  up4, _ = _non_local_context(up4_cw, up4, name='context_up4')

  up8 = (_upconv2d(up4, dim=64, act = 'linear', name= 'up8_1r') +
  _conv2d(pool2, dim=64, act='linear', name='pool2_r'))
  up8 = _conv2d(up8, dim=64, act = 'relu', name='up8_2r')
  up8, _ = _non_local_context(up8_cw, up8, name='context_up8')

  up16 = (_upconv2d(up8, dim=32, act = 'linear', name= 'up16_1r') +
  _conv2d(pool1, dim=32, act='linear', name='pool1_r'))
  up16 = _conv2d(up16, dim=32, act = 'relu', name='up16_2r')
  up16_r, a = _non_local_context(up16_cw, up16, name='context_up16')

  return [up16_cw, up16_r]

def deepfloorplanModel(config: argparse.Namespace = None, dtype=tf.float32):
    input_shape=[512,512,3]
    inputs = layers.Input(shape=input_shape, name='input_layer')

    # Build Models
    pools: List[tf.Tensor] = build_f_net(inputs)
    cw: List[tf.Tensor] = build_cw_net(pools)
    up16_cw, up16_r = build_r_net(pools, cw)

    # Compute Logits
    logits_cw = _up_bilinear(up16_cw, dim=3, shape=(512,512), name='logits_cw')  # 3X3
    logits_r = _up_bilinear(up16_r, dim=4, shape=(512,512), name='logits_r')
    # logits_r = _up_bilinear(up16_r, dim=4, shape=(512,512), name='logits_r') # Free
    logits = [logits_cw, logits_r]

    return Model(inputs=inputs, outputs=logits, name="DEEP_NET")

# img = tf.random.uniform(shape=(1,512,512,3), minval=0, maxval=255, dtype=tf.float32)
# model = deepfloorplanModel()
# logits_cw, logits_r = model(img)
# # logits = deepfloorplanModel(img)

# # model.summary()

