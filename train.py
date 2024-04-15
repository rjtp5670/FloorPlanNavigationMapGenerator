
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

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/home/david/SIMPLE_DFPN/utils'))))
from utils.settings import overwrite_args_with_toml

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

# import seaborn as sns

# def plot_confusion_matrix(hist, class_names):

#     pass
#     # plt.figure(figsize=(10,8))

#     # sns.heatmap(hist, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

#     # plt.ylabel("True label")
#     # plt.xlabel("Predict Label")
#     # plt.title("Confustion Matrix")
#     # plt.show()

# def fast_hist(im, gt, n=4): # background, Free, wall, free, opening, wall line
#     """
#     n is num_of_classes
#     """
#     k = (gt >= 0) & (gt < n)
#     return np.bincount(n * gt[k].astype(int) + im[k], minlength=n**2).reshape(n, n)

# def eval_acc(boundary, room, logits_cw, logits_r, epoch, comp_epochs):
#     boundary_class_names = ("Back Ground", "Opening", "Wall")
#     room_class_names = ("Back Ground", "Free")

#     # hist = np.zeros((num_of_classes, num_of_classes))
#     bound_num_of_class = 3
#     bound_num_of_room = 4

#     # Confusion Matrix Place Holder
#     hist_bound = np.zeros((bound_num_of_class, bound_num_of_class))
#     hist_room = np.zeros((bound_num_of_room, bound_num_of_room))

#     # Convert logits to discrete values (e.g., using argmax)
#     predicted_cw = tf.argmax(logits_cw, axis=-1)
#     predicted_r = tf.argmax(logits_r, axis=-1)

#     # Flatten and calculate histogram
#     bound_flat = tf.reshape(boundary, [-1])
#     room_flat = tf.reshape(room, [-1])

#     predicted_cw_flat = tf.reshape(predicted_cw, [-1])
#     predicted_r_flat = tf.reshape(predicted_r, [-1])

#     hist_bound += fast_hist(predicted_cw_flat.numpy(), bound_flat.numpy(), bound_num_of_class)
#     hist_room  += fast_hist(predicted_r_flat.numpy(), room_flat.numpy(), bound_num_of_room)

#     overall_acc_bound = np.diag(hist_bound).sum() / hist_bound.sum()
#     mean_acc_bound = np.nanmean(np.diag(hist_bound) / (np.sum(hist_bound, axis=1) + 1e-6))

#     overall_acc_room = np.diag(hist_room).sum() / hist_room.sum()
#     mean_acc_room = np.nanmean(np.diag(hist_room) / (np.sum(hist_room, axis=1) + 1e-6))

#     overall_acc = [overall_acc_bound, overall_acc_room]
#     mean_acc = [mean_acc_bound, mean_acc_room]

#     if epoch == (comp_epochs - 1) :
#       plot_confusion_matrix(hist_bound, boundary_class_names)
#       plot_confusion_matrix(hist_room, room_class_names)

#     return overall_acc, mean_acc

# Evaluate IoU

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

# def image_grid_overlays(
#     img: tf.Tensor,
#     overlay_cw: tf.Tensor,
#     overlay_r: tf.Tensor,
#     logits_r: tf.Tensor,
#     logits_cw: tf.Tensor,
# ) -> matplotlib.figure.Figure:
#     figure = plt.figure()
#     plt.subplot(2, 3, 1)
#     plt.imshow(img[0].numpy())
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.subplot(2, 3, 2)
#     plt.imshow(overlay_cw[0].numpy())
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.subplot(2, 3, 3)
#     plt.imshow(overlay_r[0].numpy())
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.subplot(2, 3, 5)
#     plt.imshow(convert_one_hot_to_image(logits_cw)[0].numpy().squeeze())
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.subplot(2, 3, 6)
#     plt.imshow(convert_one_hot_to_image(logits_r)[0].numpy().squeeze())
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     return figure

# def checkpoints(epoch, optim, model):
#   checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0), optimizer=optim, model=model)
#   manager = tf.train.CheckpointManager(checkpoint, directory='./checkpoints', max_to_keep=5)
#   checkpoint.epoch.assign_add(1)
#   save_path = manager.save()
#   print("Saved checkpoint for epoch {}: {}".format(int(checkpoint.epoch), save_path))

#   if manager.latest_checkpoint:
#     checkpoint.restore(manager.latest_checkpoint)
#     print("Restored from {}".format(manager.latest_checkpoint))
#   else:
#     print("Initializing from scratch")

#   return int(checkpoint.epoch)

#   # for epoch in range(int(checkpoint.epoch), config.epochs)

def main(config: argparse.Namespace):
    # initialization
    writer = tf.summary.create_file_writer(config.logdir)
    pltiter = 0
    dataset, model, optim = init(config)
    # model.compile(optim)

    # # training loop
    # checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0), optimizer=optim, model=model)
    # manager = tf.train.CheckpointManager(checkpoint, directory='/home/david/SIMPLE_DFPN/checkpoints/', max_to_keep=5)
    # if manager.latest_checkpoint:
    #   checkpoint.restore(manager.latest_checkpoint)
    #   # checkpoint.restore('/home/david/SIMPLE_DFPN/checkpoints/ckpt-19')
    #   print("Restored from {}".format(manager.latest_checkpoint))
    # else:
    #   print("Initializing from scratch.")
    #   print("[Info] checkpoint.epoch", int(checkpoint.epoch))
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
        # checkpoint.epoch.assign_add(1)
        # save_path = manager.save()
        # print("Saved checkpoint for epoch {}: {}".format(int(checkpoint.epoch), save_path))
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
    args_dict.epochs = 100
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

# %cd /content/drive/MyDrive/Colab Notebooks/Deep Floor Plan Recognition/
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

# %cd /home/david/SIMPLE_DFPN/
# 이미지 로드 및 전처리
image_path = '/home/david/SIMPLE_DFPN/resources/[TEST].png'  # 실제 이미지 경로로 바꾸세요
image = Image.open(image_path)
image = image.resize((512, 512))  # 모델에 맞는 이미지 크기로 조정
image = np.array(image)  # 이미지를 넘파이 배열로 변환

# plt.imshow(image)

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

loaded_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# 모델 평가
result = loaded_model.predict(image)

image_org = Image.open(image_path)
plt.imshow(np.array(image_org))

print(result[1].shape)
print(np.squeeze(result[0][:,:,:], axis=0)[:][:][:,:,0].shape)
print(np.squeeze(result[1], axis=0).shape)

plt.imshow(np.squeeze(result[0][:,:,:], axis=0)[:][:][:,:,])

# Commented out IPython magic to ensure Python compatibility.
# %unload_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
from tensorboard import notebook
# %load_ext tensorboard
# %tensorboard --logdir=log/store

notebook.display(port=6009)

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext tensorboard

