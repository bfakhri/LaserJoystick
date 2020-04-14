#################################################
# Trains the Simple Object Detector (SOD) Model #
#################################################

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from sod_model import MSFE_Model, SOD_Model
import sys
import os
import datetime
# Hack to get it to work with RTX cards
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Params
batch_size = 16 
img_size = 640

# Load/configure dataset
(ds_train, ds_test), info = tfds.load(name='coco/2017', split=['train', 'test'], with_info=True)
def imgr(sample):
    img = sample['image']
    bb = sample['objects']['bbox']
    img = tf.image.pad_to_bounding_box(img, 0, 0, img_size, img_size)
    img = tf.cast(img, tf.float32)/255.0
    img_bb = tf.squeeze(tf.image.draw_bounding_boxes(tf.expand_dims(img, axis=0), tf.expand_dims(bb, axis=0), [(255,255,255,100)]))
    return (img, img_bb)
ds_train = ds_train.map(imgr, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# Make log directories
base_logdir = './logs/'
logdir = base_logdir + datetime.datetime.now().strftime('%d-%H%M%S')
writer = tf.summary.create_file_writer(logdir)


for step,data in enumerate(ds_train):
    print('Step: ', step)
    imgs = data[0]
    bbs = data[1]
    with writer.as_default():
        tf.summary.image('train/image', imgs, step=step)
        tf.summary.image('train/bb_img', bbs, step=step)
        tf.summary.scalar('step', step, step=step)
