
import argparse
import io
import os
import sys
from typing import List, Tuple
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from data import convert_one_hot_to_image, decodeAllRaw, loadDataset, preprocess
from net import deepfloorplanModel
from loss import balanced_entropy, cross_two_tasks_weight

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/home/david/david_ws/FloorPlanNavigationMapGenerator/utils'))))
from utils.settings import overwrite_args_with_toml

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

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def init(
    config: argparse.Namespace,
) -> Tuple[tf.data.Dataset, tf.keras.Model, tf.keras.optimizers.Optimizer]:
    dataset = loadDataset()
    # model = deepfloorplanFunc(config=config)
    model = deepfloorplanModel(config=config)
    os.system(f"mkdir -p {config.modeldir}")
    if config.weight:
        model.load_weights(config.weight)
    optim = tf.keras.optimizers.Adam(learning_rate=config.lr)
    return dataset, model, optim

def plot_to_image(figure: matplotlib.figure.Figure) -> tf.Tensor:
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def image_grid(
    img_origin: tf.Tensor,
    img_bound: tf.Tensor,
    img_room: tf.Tensor,
    logits_r: tf.Tensor,
    logits_cw: tf.Tensor,
    m_iou = None,
    option = None,
) -> matplotlib.figure.Figure:
    figure = plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(img_origin[0].numpy())
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(2, 3, 2)
    plt.imshow(img_bound[0].numpy())
    if option is not None:
      plt.text(0.5, -0.1, f'Mean IoU: {m_iou[0]:.4f}', transform=plt.gca().transAxes)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(2, 3, 3)
    plt.imshow(img_room[0].numpy())
    if option is not None:
      plt.text(0.5, -0.1, f'Mean IoU: {m_iou[1]:.4f}', transform=plt.gca().transAxes)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(2, 3, 5)
    plt.imshow(convert_one_hot_to_image(logits_cw)[0].numpy().squeeze())
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(2, 3, 6)
    plt.imshow(convert_one_hot_to_image(logits_r)[0].numpy().squeeze())
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    return figure

@tf.function
def train_step(
    model: tf.keras.Model,
    optim: tf.keras.optimizers.Optimizer,
    img: tf.Tensor,
    hr: tf.Tensor,
    hb: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        logits_cw, logits_r = model(img)
        loss1 = balanced_entropy(logits_r, hr)
        loss2 = balanced_entropy(logits_cw, hb)
        w1, w2 = cross_two_tasks_weight(hr, hb)
        loss = w1 * loss1 + w2 * loss2
        # print("\n loss =", loss)
    # backward
    grads = tape.gradient(loss, model.trainable_weights)
    optim.apply_gradients(zip(grads, model.trainable_weights))
    return logits_cw, logits_r, loss, loss1, loss2

from tensorflow.keras.metrics import MeanIoU

def create_overlay(true_mask, pred_mask) -> tf.Tensor:
  overlay = tf.zeros(true_mask.shape + (3,), dtype=np.uint8)
  # overlay[true_mask >= 1] = [0, 0, 255]
  # overlay[pred_mask >= 1] = [255, 0, 0]
  # overlay[(true_mask >= 1) & (pred_mask >= 1)] = [255, 0, 255]

  blue = tf.constant([0, 0, 255], dtype=tf.uint8)
  red = tf.constant([255, 0, 0], dtype=tf.uint8)
  purple = tf.constant([255, 0, 255], dtype=tf.uint8)

  blue_mask = tf.cast(true_mask >= 1, dtype=tf.uint8)
  red_mask = tf.cast(pred_mask >= 1, dtype=tf.uint8)
  purple_mask = tf.cast((true_mask >= 1) & (pred_mask >= 1), dtype=tf.uint8)

  overlay += blue_mask[..., tf.newaxis] * blue
  overlay += red_mask[..., tf.newaxis] * red
  overlay -= purple_mask[..., tf.newaxis] * blue
  overlay += purple_mask[..., tf.newaxis] * purple

  return overlay

def eval_iou(logits, gt, num_of_class):
  mean_iou = MeanIoU(num_classes=num_of_class)
  pred_mask = logits
  pred_mask = tf.argmax(pred_mask, axis = -1)
  true_mask = gt
  overlay = create_overlay(true_mask, pred_mask)
  mean_iou.update_state(true_mask, pred_mask)
  mean_iou_result = mean_iou.result().numpy()

  mean_iou.reset_state()

  return mean_iou_result, overlay

def main(config: argparse.Namespace):
    # initialization
    writer = tf.summary.create_file_writer(config.logdir)
    pltiter = 0
    dataset, model, optim = init(config)
    # model.compile(optim)
    model.summary()
    for epoch in range(config.epochs):
        print("[INFO] Epoch {}".format(epoch))
        for data in tqdm(list(dataset.shuffle(400).batch(config.batchsize))):
            img, bound, room = decodeAllRaw(data)
            img, bound, room, hb, hr = preprocess(img, bound, room)
            logits_cw, logits_r, loss, loss1, loss2 = train_step(
                model, optim, img, hr, hb
            )

            # plot progress
            if pltiter % config.save_tensor_interval == 0:
                m_iou_cw, overlay_cw = eval_iou(logits_cw, bound, 3)
                m_iou_r, overlay_r = eval_iou(logits_r, room, 4)

                f = image_grid(img, bound, room, logits_r, logits_cw)
                im = plot_to_image(f)

                m_iou = [m_iou_cw, m_iou_r]
                overlays = image_grid(img, overlay_cw, overlay_r, logits_r, logits_cw, m_iou, option="iou")
                ovs = plot_to_image(overlays)

                with writer.as_default():
                    tf.summary.scalar("Loss", loss.numpy(), step=pltiter)
                    tf.summary.scalar("LossR", loss1.numpy(), step=pltiter)
                    tf.summary.scalar("LossB", loss2.numpy(), step=pltiter)
                    tf.summary.image("Data", im, step=pltiter)
                    tf.summary.scalar("Mean IoU Wall", m_iou_r, step=pltiter)
                    tf.summary.scalar("Mean IoU Room", m_iou_cw, step=pltiter)
                    tf.summary.image("overlays", ovs, step=pltiter)
                    tf.summary.image("overlay_cw", overlay_cw, step=pltiter)
                    tf.summary.image("overlay_r", overlay_r, step=pltiter)
                writer.flush()
            pltiter += 1
        print(f"Mean IoU CW: {m_iou_cw}", f"Mean IoU Room: {m_iou_r}")

        # save model
        if epoch % config.save_model_interval == 0:
            model.save_weights(config.logdir + "/G")
            model.save(config.modeldir)
            print("[INFO] Saving Model ...")

from easydict import EasyDict

def parse_args(args: List[str]) -> EasyDict:
    args_dict = EasyDict()
    args_dict.tfmodel = "subclass"
    args_dict.batchsize = 2
    args_dict.lr = 1e-4
    args_dict.wd = 1e-5
    args_dict.epochs = 1
    args_dict.logdir = "log/store"
    args_dict.modeldir = "model/store"
    args_dict.weight = None
    args_dict.save_tensor_interval = 10
    args_dict.save_model_interval = 20
    args_dict.tomlfile = None
    # args_dict.feature_channels = [256, 128, 64, 32]
    args_dict.backbone = "vgg16"
    args_dict.feature_names = [
        "block1_pool",
        "block2_pool",
        "block3_pool",
        "block4_pool",
        "block5_pool",
    ]
    return args_dict

if __name__ == "__main__":
  args = parse_args(sys.argv[1:])
  args = overwrite_args_with_toml(args)
  print(args)
  main(args)

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.optimizers import Adam

# 이미지 로드 및 전처리
image_path = '/home/david/david_ws/FloorPlanNavigationMapGenerator/resources/[TEST].png'  # 실제 이미지 경로로 바꾸세요
image = Image.open(image_path)
image = image.resize((512, 512))  # 모델에 맞는 이미지 크기로 조정
image = np.array(image)  # 이미지를 넘파이 배열로 변환

# 이미지의 채널 차원 확인
if image.shape[-1] == 4:
    image = image[..., :3]  # 알파 채널 제거

image = image / 255.0  # 이미지를 0과 1 사이의 값으로 정규화

# plt.imshow(image)

print(image.shape)

image = np.expand_dims(image, axis=0)  # 배치 차원 추가

print(image.shape)

# 모델 불러오기
loaded_model = tf.keras.models.load_model(args.modeldir)

# 모델 컴파일
optimizer = Adam(learning_rate=0.001)  # 원하는 옵티마이저와 학습률을 지정하세요.
loss = 'sparse_categorical_crossentropy'  # 원하는 손실 함수를 지정하세요.
metrics = ['accuracy']  # 원하는 평가 메트릭을 지정하세요.

# loaded_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
loaded_model.compile(optimizer='adam', loss={'cw': 'binary_crossentropy', 'r': 'sparse_categorical_crossentropy'},
              loss_weights={'cw': 1.0, 'r': 1.0})

# 모델 평가
result = loaded_model.predict(image)

image_org = Image.open(image_path)
plt.imshow(np.array(image_org))

print(result[1].shape)
print(np.squeeze(result[0][:,:,:], axis=0)[:][:][:,:,0].shape)
print(np.squeeze(result[1], axis=0).shape)

plt.imshow(np.squeeze(result[0][:,:,:], axis=0)[:][:][:,:,])

from tensorboard import notebook

notebook.display(port=6009)
