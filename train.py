import argparse
import io
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gc

from typing import List, Tuple
from tqdm import tqdm
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.metrics import MeanIoU

from data import convert_one_hot_to_image, decodeAllRaw, loadDataset, preprocess
from net import deepfloorplanModel
from loss import balanced_entropy, cross_two_tasks_weight
from utils.settings import overwrite_args_with_toml
from easydict import EasyDict

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/home/david/david_ws/SIMPLE_DFPN/utils'))))

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
GPU_ID = '0'

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # 현재 프로그램이 필요할 때만 메모리를 할당
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

gc.collect()

# TensorFlow에서 GPU 메모리 사용량을 관리하도록 설정
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 필요한 만큼만 메모리를 사용하도록 설정
session = tf.compat.v1.Session(config=config)

from tensorflow.keras.optimizers.schedules import ExponentialDecay

def init(config: argparse.Namespace) -> Tuple[tf.data.Dataset, tf.keras.Model, tf.keras.optimizers.Optimizer]:
    dataset = loadDataset()
    model, masks = deepfloorplanModel(config=config)

    if config.weight:
        model.load_weights(config.weight)

    # 학습률 스케줄러 설정
    initial_learning_rate = config.lr  # config에서 초기 학습률 가져오기
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,  # 몇 스텝마다 학습률을 감소시킬지
        decay_rate=0.9,  # 학습률 감소율
        staircase=True    # 계단식 감소 여부
    )

    # Optimizer에 학습률 스케줄러 적용
    optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return dataset, model, optim, masks

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
    overlay_cw_prt = None,
    overlay_r_prt = None,
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
        plt.text(0.5, -0.1, f'Mean IoU: {m_iou[0]:.4f}\nMatch: {overlay_cw_prt:.2f}', transform=plt.gca().transAxes)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplot(2, 3, 3)
    plt.imshow(img_room[0].numpy())
    if option is not None:
        plt.text(0.5, -0.1, f'Mean IoU: {m_iou[1]:.4f}\nMatch: {overlay_r_prt:.2f}', transform=plt.gca().transAxes)
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
def train_step(model, optim, img, hr, hb, masks):
    with tf.GradientTape() as tape:
        
        logits_cw, logits_r = model(img)
        loss1 = balanced_entropy(logits_r, hr, masks)
        loss2 = balanced_entropy(logits_cw, hb, masks)
        w1, w2 = cross_two_tasks_weight(hr, hb)
        loss = w1 * loss1 + w2 * loss2
    # Backward pass
    grads = tape.gradient(loss, model.trainable_weights)
    optim.apply_gradients(zip(grads, model.trainable_weights))
    return logits_cw, logits_r, loss, loss1, loss2

def create_overlay(true_mask, pred_mask) -> tf.Tensor:
    overlay = tf.zeros(true_mask.shape + (3,), dtype=np.uint8)

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

from tensorflow.keras.metrics import MeanIoU

def create_overlay(true_mask, pred_mask) -> tf.Tensor:
    overlay = tf.zeros(true_mask.shape + (3,), dtype=np.uint8)

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
    pred_mask = tf.argmax(logits, axis=-1, output_type=tf.int32)
    true_mask = tf.cast(gt, tf.int32)

    # Create overlay image
    overlay = create_overlay(true_mask, pred_mask)

    # Calculate IoU
    mean_iou.update_state(true_mask, pred_mask)
    mean_iou_result = mean_iou.result().numpy()

    # Exclude black pixels (0 value) from comparison
    valid_mask = tf.cast(true_mask > 0, tf.float32)
    matching_pixels = tf.reduce_sum(tf.cast((true_mask == pred_mask), tf.float32) * valid_mask)
    total_valid_pixels = tf.reduce_sum(valid_mask)
    match_percentage = (matching_pixels / total_valid_pixels).numpy()

    mean_iou.reset_state()

    return mean_iou_result, overlay, match_percentage

def main(config: argparse.Namespace):
    writer = tf.summary.create_file_writer(config.logdir)
    pltiter = 0
    dataset, model, optim, mask = init(config)

    for epoch in range(config.epochs):
        print("[INFO] Epoch {}".format(epoch))
        for data in tqdm(list(dataset.shuffle(400).batch(config.batchsize))):
            img, bound, room = decodeAllRaw(data)
            img, bound, room, hb, hr = preprocess(img, bound, room)
            logits_cw, logits_r, loss, loss1, loss2 = train_step(
                model, optim, img, hr, hb, mask
            )

            if pltiter % config.save_tensor_interval == 0:
                m_iou_cw, overlay_cw, overlay_cw_prt = eval_iou(logits_cw, bound, 3)
                m_iou_r, overlay_r, overlay_r_prt = eval_iou(logits_r, room, 2)

                f = image_grid(img, bound, room, logits_r, logits_cw)
                im = plot_to_image(f)

                m_iou = [m_iou_cw, m_iou_r]
                overlays = image_grid(img, overlay_cw, overlay_r, logits_r, logits_cw, m_iou, overlay_cw_prt, overlay_r_prt, option="iou")
                ovs = plot_to_image(overlays)

                with writer.as_default():
                    tf.summary.scalar("Loss", loss.numpy(), step=pltiter)
                    tf.summary.image("Data", im, step=pltiter)
                    tf.summary.image("overlays", ovs, step=pltiter)
                writer.flush()
            pltiter += 1
        print(f"Mean IoU CW: {m_iou_cw}", f"Mean IoU Room: {m_iou_r}", f"LOSS: {loss.numpy()}")
        print(f"overlay_cw_prt: {overlay_cw_prt}", f"overlay_r_prt: {overlay_r_prt}")

        # save model
        if epoch % config.save_model_interval == 0:
            model.save_weights(config.logdir + "/G")
            model.save(config.modeldir)
            print("[INFO] Saving Model ...")

def parse_args(args: List[str]) -> EasyDict:
    args_dict = EasyDict()
    args_dict.mask = True
    args_dict.tfmodel = "subclass"
    args_dict.batchsize = 1
    args_dict.lr = 1e-4
    args_dict.wd = 1e-5
    args_dict.epochs = 10
    args_dict.logdir = "log/store"
    args_dict.modeldir = "model/store"
    args_dict.weight = None
    args_dict.tomlfile = None    
    args_dict.save_tensor_interval = 10
    args_dict.save_model_interval = 20
    return args_dict

if __name__ == "__main__":
  args = parse_args(sys.argv[1:])
  args = overwrite_args_with_toml(args)
  print(args)
  main(args)

# from tensorboard import notebook
# %load_ext tensorboard
# %tensorboard --logdir=log/store

# notebook.display(port=6009)