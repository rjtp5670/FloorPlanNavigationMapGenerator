import tensorflow as tf
import cv2
import numpy as np
import os
import sys
import glob
import time
import matplotlib.pyplot as plt

from PIL import Image
from typing import Dict, List
from imageio import imread

# use for index 2 rgb
floorplan_room_map = {
	0: [  0,  0,  0], # background
	1: [203, 203, 203], # Free
	# 2: [224,224,224], # not used
	# 3: [224,224,128]  # not used
}
# boundary label
floorplan_boundary_map = {
	0: [  0,  0,  0], # background
	1: [255,60,128],  # opening (door&window)
	2: [255,255,255]  # wall line
}

# boundary label for presentation
floorplan_boundary_map_figure = {
	0: [255,255,255], # background
	1: [255, 60,128],  # opening (door&window)
	2: [  0,  0,  0]  # wall line
}

# merge all label into one multi-class label
floorplan_fuse_map = {
	0: [  0,  0,  0], # background
	1: [203, 203, 203], # Free
	2: [224,224,224], # not used
	3: [224,224,128], # not used
	9: [255,60,128],  # extra label for opening (door&window)
	10: [255,255,255]  # extra label for wall line
}

# invert the color of wall line and background for presentation
floorplan_fuse_map_figure = {
	0: [255,255,255], # background
	1: [203, 203, 203], # Free
	2: [224,224,224], # not used
	3: [224,224,128], # not used
	9: [255,60,128],  # extra label for opening (door&window)
	10: [ 0, 0,  0]  # extra label for wall line
}

def rgb2ind(
    im: np.ndarray, color_map: Dict[int, List[int]] = floorplan_room_map
) -> np.ndarray:
    ind = np.zeros((im.shape[0], im.shape[1]))

    for i, rgb in color_map.items():
        ind[(im == rgb).all(2)] = i

    return ind.astype(np.uint8)  # force to uint8

def ind2rgb(
    ind_im: np.ndarray, color_map: Dict[int, List[int]] = floorplan_room_map
) -> np.ndarray:
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for i, rgb in color_map.items():
        rgb_im[(ind_im == i)] = rgb
    return rgb_im.astype(int)

def load_bd_rm_images(path):
    paths = path.split('\t')
    image = np.array(Image.open(paths[0]).convert('RGB'))
    close = np.array(Image.open(paths[2]).convert('L'))
    room = np.array(Image.open(paths[3]).convert('RGB'))
    close_wall = np.array(Image.open(paths[4]).convert('L'))

    # Resize the images
    new_size = (512, 512)

    image = np.array(Image.fromarray(image).resize(new_size))
    close = np.array(Image.fromarray(close).resize(new_size)) / 255.0
    close_wall = np.array(Image.fromarray(close_wall).resize(new_size)) / 255.0
    room = np.array(Image.fromarray(room).resize(new_size))

    room_ind = rgb2ind(room)

    # Merge result
    d_ind = (close > 0.5).astype(np.uint8)

    cw_ind = (close_wall > 0.5).astype(np.uint8)
    cw_ind[cw_ind == 1] = 2
    cw_ind[d_ind == 1] = 1

    # Make sure the dtype is uint8
    image = image.astype(np.uint8)
    room_ind = room_ind.astype(np.uint8)
    cw_ind = cw_ind.astype(np.uint8)

    return image, cw_ind, room_ind, d_ind

from tensorflow.io import TFRecordWriter # TensorFlow 2.X

def _bytes_feature(value): 
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_bd_rm_record(paths, name='dataset.tfrecords'):

  with TFRecordWriter(name) as writer:

    for i in range(len(paths)):
      # Load the image
      image, cw_ind, room_ind, d_ind = load_bd_rm_images(paths[i]) # Original, Close Wall, Room, Door

      #Create a feature
      feature = {'image': _bytes_feature(image.tobytes()),
            'boundary': _bytes_feature(cw_ind.tobytes()),
            'room': _bytes_feature(room_ind.tobytes()),
            'door': _bytes_feature(d_ind.tobytes())}

      # Create an example protocol buffer
      example = tf.train.Example(features=tf.train.Features(feature=feature))

      # Serialize to string and write on the file
      writer.write(example.SerializeToString())

tfrecord_file = '/create_tfrecords/dataset/2_240519_r3d.tfrecords'

def show_me_recorded_data():

    dataset = tf.data.TFRecordDataset(tfrecord_file)
   
    def _parse_function(example_proto):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'boundary': tf.io.FixedLenFeature([], tf.string),
            'room': tf.io.FixedLenFeature([], tf.string),
            'door': tf.io.FixedLenFeature([], tf.string)
        }
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = dataset.map(_parse_function)

    example_list = list(parsed_dataset)

    # Display images in a grid of 4 columns (image, boundary, room, door) for each example
    num_examples = 10
    num_columns = 4
    plt.figure(figsize=(num_columns * 4, num_examples * 3))  # Adjust the size as needed

    for i, example in enumerate(example_list[:num_examples]):
        image_data = tf.io.decode_raw(example['image'], tf.uint8)
        boundary_data = tf.io.decode_raw(example['boundary'], tf.uint8)
        room_data = tf.io.decode_raw(example['room'], tf.uint8)
        door_data = tf.io.decode_raw(example['door'], tf.uint8)

        image_shape = [512, 512, 3]
        bd_shape = [512, 512, 1]
        rm_shape = [512, 512, 1]
        dr_shape = [512, 512, 1]

        img_data = tf.reshape(image_data, image_shape)
        bd_data = tf.reshape(boundary_data, bd_shape)
        rm_data = tf.reshape(room_data, rm_shape)
        dr_data = tf.reshape(door_data, dr_shape)

        # Plot each component of the example in its subplot
        plt.subplot(num_examples, num_columns, i * num_columns + 1)
        plt.imshow(img_data.numpy())
        plt.title('Image')
        plt.axis('off')

        plt.subplot(num_examples, num_columns, i * num_columns + 2)
        plt.imshow(bd_data.numpy().squeeze())
        plt.title('Boundary')
        plt.axis('off')

        plt.subplot(num_examples, num_columns, i * num_columns + 3)
        plt.imshow(rm_data.numpy().squeeze())
        plt.title('Room')
        plt.axis('off')

        plt.subplot(num_examples, num_columns, i * num_columns + 4)
        plt.imshow(dr_data.numpy().squeeze())
        plt.title('Door')
        plt.axis('off')

    # Show all plots
    plt.tight_layout()
    plt.show()

# Commented out IPython magic to ensure Python compatibility.
if __name__ == '__main__':
	# write to TFRecord
    train_file = '/create_tfrecords/3_r3d_all.txt'    
    train_paths = open(train_file, 'r').read().splitlines()
    write_bd_rm_record(train_paths, name='../dataset/2_240519_r3d.tfrecords')
    
    # Enable interactive mode in Matplotlib
    plt.ion()
    show_me_recorded_data()