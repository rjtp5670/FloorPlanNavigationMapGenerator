import argparse
import gc
import os
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils.settings import overwrite_args_with_toml
from utils.util import fill_break_line, flood_fill, refine_room_region
from typing import List, Tuple
from easydict import EasyDict
from data import convert_one_hot_to_image
from net import deepfloorplanModel


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/home/david/david_ws/FloorPlanNavigationMapGenerator/utils'))))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

applied_size = 256

def init(
    config: argparse.Namespace,
) -> Tuple[tf.keras.Model, tf.Tensor, np.ndarray]:
    
    model, _ = deepfloorplanModel(config=config)
    if config.loadmethod == "log":
        model.load_weights(config.weight)
    img = mpimg.imread(config.image)[:, :, :3]
    shape = img.shape
    img = tf.convert_to_tensor(img, dtype=tf.uint8)
    img = tf.image.resize(img, [applied_size, applied_size])
    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [-1, applied_size, applied_size, 3])

    if tf.math.reduce_max(img) > 1.0:
        img /= 255
    if config.loadmethod == "tflite":
        return model, img, shape
    return model, img, shape

def predict(
    model: tf.keras.Model, img: tf.Tensor, shp: np.ndarray
) -> Tuple[tf.Tensor, tf.Tensor]:
    features = []
    feature = img
    for layer in model.vgg16.layers:
        feature = layer(feature)
        if layer.name.find("pool") != -1:
            features.append(feature)
    x = feature
    features = features[::-1]
    del model.vgg16
    gc.collect()

    featuresrbp = []
    for i in range(len(model.rbpups)):
        x = model.rbpups[i](x) + model.rbpcv1[i](features[i + 1])
        x = model.rbpcv2[i](x)
        featuresrbp.append(x)
    logits_cw = tf.keras.backend.resize_images(
        model.rbpfinal(x), 2, 2, "channels_last"
    )

    x = features.pop(0)
    nLays = len(model.rtpups)
    for i in range(nLays):
        rs = model.rtpups.pop(0)
        r1 = model.rtpcv1.pop(0)
        r2 = model.rtpcv2.pop(0)
        f = features.pop(0)
        x = rs(x) + r1(f)
        x = r2(x)
        a = featuresrbp.pop(0)
        x = model.non_local_context(a, x, i)

    del featuresrbp
    logits_r = tf.keras.backend.resize_images(
        model.rtpfinal(x), 2, 2, "channels_last"
    )

    del model.rtpfinal

    return logits_cw, logits_r

def post_process(
    rm_ind: np.ndarray, bd_ind: np.ndarray, shp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    print("post_process")
    hard_c = (bd_ind > 0).astype(np.uint8)
    # region from room prediction
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind > 0] = 1
    # region from close wall line
    cw_mask = hard_c
    # regine close wall mask by filling the gap between bright line
    cw_mask = fill_break_line(cw_mask)
    cw_mask = np.reshape(cw_mask, (*shp[:2], -1))
    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask >= 1] = 255

    # refine fuse mask by filling the hole
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask // 255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask, rm_ind)

    # ignore the background mislabeling
    new_rm_ind = fuse_mask.reshape(*shp[:2], -1) * new_rm_ind
    new_bd_ind = fill_break_line(bd_ind).squeeze()
    return new_rm_ind, new_bd_ind

from typing import Dict, List

import numpy as np

# use for index 2 rgb
floorplan_room_map = {
	0: [0, 0, 0], # background
	1: [255, 255, 255], # Free

}

# boundary label
floorplan_boundary_map = {
    0: [203, 203, 203],  # background
    1: [255, 255, 255],  # opening (door&window)
    2: [0,0,0],  # wall line
}

def rgb2ind(
    im: np.ndarray, color_map: Dict[int, List[int]] = floorplan_room_map
) -> np.ndarray:
    ind = np.zeros((im.shape[0], im.shape[1]))

    for i, rgb in color_map.items():
        ind[(im == rgb).all(2)] = i

    # return ind.astype(int) # int => int64
    return ind.astype(np.uint8)  # force to uint8

def ind2rgb(
    ind_im: np.ndarray, color_map: Dict[int, List[int]] = floorplan_room_map
) -> np.ndarray:
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for i, rgb in color_map.items():
        rgb_im[(ind_im == i)] = rgb
    return rgb_im.astype(int)

def colorize(r: np.ndarray, cw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print("colorize")
    cr = ind2rgb(r, color_map=floorplan_room_map)
    ccw = ind2rgb(cw, color_map=floorplan_boundary_map)
    return cr, ccw

def main(config: argparse.Namespace) -> np.ndarray:
    model, img, shape = init(config)
    logits_cw, logits_r = model(img)
    logits_cw = tf.image.resize(logits_cw, shape[:2])
    logits_r = tf.image.resize(logits_r, shape[:2])
    cw = convert_one_hot_to_image(logits_cw)[0].numpy()
    r = convert_one_hot_to_image(logits_r)[0].numpy()

    if not config.colorize and not config.postprocess:
        cw[cw == 1] = 9
        cw[cw == 2] = 10
        r[cw != 0] = 0
        return (r + cw).squeeze()
    elif config.colorize and not config.postprocess:
        r_color, cw_color = colorize(r.squeeze(), cw.squeeze())
        return r_color + cw_color

    newr, newcw = post_process(r, cw, shape)
    if not config.colorize and config.postprocess:
        newcw[newcw == 1] = 9
        newcw[newcw == 2] = 10
        newr[newcw != 0] = 0
        return newr.squeeze() + newcw
    newr_color, newcw_color = colorize(newr.squeeze(), newcw.squeeze())
    result = newr_color + newcw_color

    if config.save:
        mpimg.imsave(config.save, result.astype(np.uint8))

    return result

def deploy_plot_res(result: np.ndarray): 

    dpi = 100  # dots per inch, 해상도 설정
    height, width, _ = result.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    plt.imshow(result)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    file_path = "/home/david/david_ws/FloorPlanNavigationMapGenerator/resources/output.png"
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight', pad_inches=0)

def parse_args(args: List[str]) -> EasyDict:
    args_dict = EasyDict()
    args_dict.tfmodel = "subclass"
    args_dict.image = "/resources/floorplan_resized_noised3.png"
    args_dict.weight = "log/store/G"
    args_dict.postprocess = True
    args_dict.colorize = True
    args_dict.loadmethod = "log"
    args_dict.save = None
    args_dict.tomlfile = None
    return args_dict

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    args = overwrite_args_with_toml(args)
    result = main(args)
    deploy_plot_res(result)
    plt.show()