# -*- coding: utf-8 -*-
"""tf_record.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1je0xG8GNfewU9390_8FvB35eoHWwKLyY
"""

import numpy as np

# use for index 2 rgb
floorplan_room_map = {
	0: [  0,  0,  0], # background
	1: [203, 203, 203], # Free
	# 1: [192,192,224], # closet
	# 2: [192,255,255], # bathroom/washroom
	# 3: [224,255,192], # livingroom/kitchen/diningroom
	# 4: [255,224,128], # bedroom
	# 5: [255,160, 96], # hall
	# 6: [255,224,224], # balcony
	2: [224,224,224], # not used
	3: [224,224,128]  # not used
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
	# 1: [192,192,224], # closet
	# 2: [192,255,255], # batchroom/washroom
	# 3: [224,255,192], # livingroom/kitchen/dining room
	# 4: [255,224,128], # bedroom
	# 5: [255,160, 96], # hall
	# 6: [255,224,224], # balcony
	2: [224,224,224], # not used
	3: [224,224,128], # not used
	9: [255,60,128],  # extra label for opening (door&window)
	10: [255,255,255]  # extra label for wall line
}

# invert the color of wall line and background for presentation
floorplan_fuse_map_figure = {
	0: [255,255,255], # background
	1: [203, 203, 203], # Free
	# 1: [192,192,224], # closet
	# 2: [192,255,255], # batchroom/washroom
	# 3: [224,255,192], # livingroom/kitchen/dining room
	# 4: [255,224,128], # bedroom
	# 5: [255,160, 96], # hall
	# 6: [255,224,224], # balcony
	2: [224,224,224], # not used
	3: [224,224,128], # not used
	9: [255,60,128],  # extra label for opening (door&window)
	10: [ 0, 0,  0]  # extra label for wall line
}

# idx = 0 - 8

def rgb2ind(im, color_map=floorplan_room_map): # rgb2ind 함수는 이미지와 맵핑된 컬러맵을 전달 받음
	ind = np.zeros((im.shape[0], im.shape[1])) # 이미지와 동일한 shape의 행렬 생성

	for i, rgb in color_map.items(): # 맵핑된 컬러맵을 반복하여 index 값을 찾아냄
		ind[(im==rgb).all(2)] = i # 이미지와 맵핑된 rgb 값이 2번째 축 [.all(2)]을 기준으로 정확히 일치하면 index 지정

	# return ind.astype(int) # int => int64
	return ind.astype(np.uint8) # force to uint8

def ind2rgb(ind_im, color_map=floorplan_room_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb

	return rgb_im

def unscale_imsave(path, im, cmin=0, cmax=255):
	toimage(im, cmin=cmin, cmax=cmax).save(path)

import numpy as np

import tensorflow as tf

from imageio import imread
import cv2
import numpy as np

from matplotlib import pyplot as plt
# from rgb_ind_convertor import * # 그냥 위에다가 바로 실행 시켜버림

import os
import sys
import glob
import time

# # Test
# img_path = '/content/34_close.png'
# img = imread(img_path,  mode='RGB')

def load_raw_images(path):
  paths = path.split('\t')
  image = imread(paths[0], mode='RGB')
  wall  = imread(paths[1], mode='L')
  close = imread(paths[2], mode='L')
  room  = imread(paths[3], mode='RGB')
  close_wall = imread(paths[4], mode='L')

  # NOTE: imresize will rescale the image to range [0, 255], also cast data into uint8 or uint32
  image = imresize(image, (512, 512, 3))
  wall = imresize(wall, (512, 512))
  close = imresize(close, (512, 512))
  close_wall = imresize(close_wall, (512, 512))
  room = imresize(room, (512, 512, 3))

  room_ind = rgb2ind(room)

  # make sure the dtype is uint8
  image = image.astype(np.uint8)
  wall = wall.astype(np.uint8)
  close = close.astype(np.uint8)
  close_wall = close_wall.astype(np.uint8)
  room_ind = room_ind.astype(np.uint8)

  # debug
  # plt.subplot(231)
  # plt.imshow(image)
  # plt.subplot(233)
  # plt.imshow(wall, cmap='gray')
  # plt.subplot(234)
  # plt.imshow(close, cmap='gray')
  # plt.subplot(235)
  # plt.imshow(room_ind)
  # plt.subplot(236)
  # plt.imshow(close_wall, cmap='gray')
  # plt.show()

  return image, wall, close, room_ind, close_wall

def _int64_feature(value): # 정수 값을 tfrecords 형식으로 변환하기 위한 헬퍼함수.
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value): # TFRecord Feature 생성
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

from tensorflow.io import TFRecordWriter # TensorFlow 2.X

def write_record(paths, name='dataset.tfrecords'):

  with TFRecordWriter(name) as writer:
    for i in range(len(paths)):

      # Load the image
      image, wall, close, room_ind, close_wall = load_raw_images(paths[i])

      # Create a feature
      feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
            'wall': _bytes_feature(tf.compat.as_bytes(wall.tostring())),
            'close': _bytes_feature(tf.compat.as_bytes(close.tostring())),
            'room': _bytes_feature(tf.compat.as_bytes(room_ind.tostring())),
            'close_wall': _bytes_feature(tf.compat.as_bytes(close_wall.tostring()))}

      # Create an example protocol buffer
      example = tf.train.Example(features=tf.train.Features(feature=feature))

      # Serialize to string and write on the file
      writer.write(example.SerializeToString())

def read_record(data_path, batch_size=1, size=512):
	feature = {'image': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'wall': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'close': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'room': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'close_wall': tf.FixedLenFeature(shape=(), dtype=tf.string)}

	# Create a list of filenames and pass it to a queue
	filename_queue = tf.train.string_input_producer([data_path], num_epochs=None, shuffle=False, capacity=batch_size*128)

	# Define a reader and read the next record
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	# Decode the record read by the reader
	features = tf.parse_single_example(serialized_example, features=feature)

	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features['image'], tf.uint8)
	wall = tf.decode_raw(features['wall'], tf.uint8)
	close = tf.decode_raw(features['close'], tf.uint8)
	room = tf.decode_raw(features['room'], tf.uint8)
	close_wall = tf.decode_raw(features['close_wall'], tf.uint8)

	# Cast data
	image = tf.cast(image, dtype=tf.float32)
	wall = tf.cast(wall, dtype=tf.float32)
	close = tf.cast(close, dtype=tf.float32)
	# room = tf.cast(room, dtype=tf.float32)
	close_wall = tf.cast(close_wall, dtype=tf.float32)

	# Reshape image data into the original shape
	image = tf.reshape(image, [size, size, 3])
	wall = tf.reshape(wall, [size, size, 1])
	close = tf.reshape(close, [size, size, 1])
	room = tf.reshape(room, [size, size])
	close_wall = tf.reshape(close_wall, [size, size, 1])


	# Any preprocessing here ...
	# normalize
	image = tf.divide(image, tf.constant(255.0))
	wall = tf.divide(wall, tf.constant(255.0))
	close = tf.divide(close, tf.constant(255.0))
	close_wall = tf.divide(close_wall, tf.constant(255.0))

	# Genereate one hot room label
	room_one_hot = tf.one_hot(room, 9, axis=-1)

	# Creates batches by randomly shuffling tensors
	images, walls, closes, rooms, close_walls = tf.train.shuffle_batch([image, wall, close, room_one_hot, close_wall],
						batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)

	# images, walls = tf.train.shuffle_batch([image, wall],
						# batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)

	return {'images': images, 'walls': walls, 'closes': closes, 'rooms': rooms, 'close_walls': close_walls}

# ------------------------------------------------------------------------------------------------------------------------------------- *
# Following are only for segmentation task, merge all label into one

def load_seg_raw_images(path):
	paths = path.split('\t')

	image = imread(paths[0], mode='RGB')
	close = imread(paths[2], mode='L')
	room  = imread(paths[3], mode='RGB')
	close_wall = imread(paths[4], mode='L')

	# NOTE: imresize will rescale the image to range [0, 255], also cast data into uint8 or uint32
	image = imresize(image, (512, 512, 3))
	close = imresize(close, (512, 512)) / 255
	close_wall = imresize(close_wall, (512, 512)) / 255
	room = imresize(room, (512, 512, 3))

	room_ind = rgb2ind(room)

	# merge result
	d_ind = (close>0.5).astype(np.uint8)
	cw_ind = (close_wall>0.5).astype(np.uint8)
	room_ind[cw_ind==1] = 10
	room_ind[d_ind==1] = 9

	# make sure the dtype is uint8
	image = image.astype(np.uint8)
	room_ind = room_ind.astype(np.uint8)

	# debug
	# merge = ind2rgb(room_ind, color_map=floorplan_fuse_map)
	# plt.subplot(131)
	# plt.imshow(image)
	# plt.subplot(132)
	# plt.imshow(room_ind)
	# plt.subplot(133)
	# plt.imshow(merge/256.)
	# plt.show()

	return image, room_ind

def write_seg_record(paths, name='dataset.tfrecords'):
	writer = tf.python_io.TFRecordWriter(name)

	for i in range(len(paths)):
		# Load the image
		image, room_ind = load_seg_raw_images(paths[i])

		# Create a feature
		feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
					'label': _bytes_feature(tf.compat.as_bytes(room_ind.tostring()))}

		# Create an example protocol buffer
		example = tf.train.Example(features=tf.train.Features(feature=feature))

		# Serialize to string and write on the file
		writer.write(example.SerializeToString())

	writer.close()

def read_seg_record(data_path, batch_size=1, size=512):
	feature = {'image': tf.FixedLenFeature(shape=(), dtype=tf.string),
				'label': tf.FixedLenFeature(shape=(), dtype=tf.string)}

	# Create a list of filenames and pass it to a queue
	filename_queue = tf.train.string_input_producer([data_path], num_epochs=None, shuffle=False, capacity=batch_size*128)

	# Define a reader and read the next record
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	# Decode the record read by the reader
	features = tf.parse_single_example(serialized_example, features=feature)

	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features['image'], tf.uint8)
	label = tf.decode_raw(features['label'], tf.uint8)

	# Cast data
	image = tf.cast(image, dtype=tf.float32)

	# Reshape image data into the original shape
	image = tf.reshape(image, [size, size, 3])
	label = tf.reshape(label, [size, size])


	# Any preprocessing here ...
	# normalize
	image = tf.divide(image, tf.constant(255.0))

	# Genereate one hot room label
	label_one_hot = tf.one_hot(label, 11, axis=-1)

	# Creates batches by randomly shuffling tensors
	images, labels = tf.train.shuffle_batch([image, label_one_hot],
						batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)

	# images, walls = tf.train.shuffle_batch([image, wall],
						# batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)

	return {'images': images, 'labels': labels}

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- *
# ------------------------------------------------------------------------------------------------------------------------------------- *
# Following are only for multi-task network. Two labels(boundary and room.)

from PIL import Image

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

# def _bytes_feature(value):
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

# from google.colab import drive
# drive.mount('/content/drive')
# train_file = '/content/drive/MyDrive/Colab Notebooks/create_tfrecords/dataset/r3d_train_temp2.txt'
# len(open(train_file, 'r').read().splitlines())

def write_bd_rm_record(paths, name='dataset.tfrecords'):

  with TFRecordWriter(name) as writer:

    # len(paths) = 11 개 Line으로 구분 됨
    for i in range(len(paths)):
      # Load the image
      image, cw_ind, room_ind, d_ind = load_bd_rm_images(paths[i])

      # print(" tf.compat.as_bytes(image.tostring()) >> image.tobytes()")

      #Create a feature
      feature = {'image': _bytes_feature(image.tobytes()),
            'boundary': _bytes_feature(cw_ind.tobytes()),
            'room': _bytes_feature(room_ind.tobytes()),
            'door': _bytes_feature(d_ind.tobytes())}

      # encoded_image = tf.io.encode_jpeg(image_tensor).numpy()

      # Create an example protocol buffer
      example = tf.train.Example(features=tf.train.Features(feature=feature))

      # Serialize to string and write on the file
      writer.write(example.SerializeToString())
      # print("Writing bd_rm record is done")

"""# 아래는 테스트 용"""

# import tensorflow as tf
# import numpy as np

# # 가상의 이미지 데이터 생성
# image = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)

# # 방법 1: image.tobytes()
# bytes_result_1 = image.tobytes()

# # 방법 2: tf.compat.as_bytes(image.tostring())
# bytes_result_2 = tf.compat.as_bytes(image.tobytes())

# # 결과 비교
# print("Bytes result from image.tobytes():\n", bytes_result_1[:50])
# print("\nBytes result from tf.compat.as_bytes(image.tostring()):\n", bytes_result_2[:50])

# # 결과가 같은지 비교
# print("\nAre the results equal?", bytes_result_1 == bytes_result_2)


# data = {
#     'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'image_data'])),
#     'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
# }

# features = tf.train.Features(feature=data)

# example = tf.train.Example(features=features)

# serialized_example = example.SerializeToString

# print("Serialized Example:", serialized_example)

# # Open the tfrecord sesstion

# with TFRecordWriter("Serialized_Test") as writer:

#     writer.write(example.SerializeToString())

# raw_ds = tf.data.TFRecordDataset("Serialized_Test")

# # Print the reuslt

# for raw_record in raw_ds.take(1):
#   example = tf.train.Example()
#   example.ParseFromString(raw_record.numpy())
#   print(example)

def read_bd_rm_record(data_path, batch_size=1, size=512):
	feature = {'image': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
				'boundary': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
				'room': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
				'door': tf.io.FixedLenFeature(shape=(), dtype=tf.string)}

	# Create a list of filenames and pass it to a queue
	filename_queue = tf.compat.v1.train.string_input_producer([data_path], num_epochs=None, shuffle=False, capacity=batch_size*128)

	# Define a reader and read the next record
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	# Decode the record read by the reader
	features = tf.parse_single_example(serialized_example, features=feature)

	# Convert the image data from string back to the numbers
	image = tf.decode_raw(features['image'], tf.uint8)
	boundary = tf.decode_raw(features['boundary'], tf.uint8)
	room = tf.decode_raw(features['room'], tf.uint8)
	door = tf.decode_raw(features['door'], tf.uint8)

	# Cast data
	image = tf.cast(image, dtype=tf.float32)

	# Reshape image data into the original shape
	image = tf.reshape(image, [size, size, 3])
	boundary = tf.reshape(boundary, [size, size])
	room = tf.reshape(room, [size, size])
	door = tf.reshape(door, [size, size])

	# Any preprocessing here ...
	# normalize
	image = tf.divide(image, tf.constant(255.0))

	# Genereate one hot room label
	label_boundary = tf.one_hot(boundary, 3, axis=-1)
	label_room = tf.one_hot(room, 9, axis=-1)

	# Creates batches by randomly shuffling tensors
	images, label_boundaries, label_rooms, label_doors = tf.train.shuffle_batch([image, label_boundary, label_room, door],
						batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)

	# images, walls = tf.train.shuffle_batch([image, wall],
						# batch_size=batch_size, capacity=batch_size*128, num_threads=1, min_after_dequeue=batch_size*32)

	return {'images': images, 'label_boundaries': label_boundaries, 'label_rooms': label_rooms, 'label_doors': label_doors}



