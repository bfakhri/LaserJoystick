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
batch_size = 1
img_size = 640

# Load/configure dataset
(ds_train, ds_test), info = tfds.load(name='coco/2017', split=['train', 'test'], with_info=True)
print(info)

# Data preperation
def imgr(sample):
    # Extract image
    img = sample['image']
    # BB are proportions (x_min_prop, y_min_prop, x_max_prop, y_max_prop)
    bb = sample['objects']['bbox']
    # Resize the image by padding with zeros
    #img = tf.image.pad_to_bounding_box(img, 0, 0, img_size, img_size)
    img = tf.image.resize(img, (img_size, img_size))
    # Cast and map to (0,1)
    img = tf.cast(img, tf.float32)/255.0
    # Draw BBs
    img_bb = tf.squeeze(tf.image.draw_bounding_boxes(tf.expand_dims(img, axis=0), tf.expand_dims(bb, axis=0), [(255,255,255,100)]))
    #return (img, img_size, img_bb)
    return (img, img_bb, bb)

# Data pipeline definition
ds_train = ds_train.map(imgr, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(batch_size)
#ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# Make log directories
base_logdir = './logs/'
logdir = base_logdir + datetime.datetime.now().strftime('%d-%H%M%S')
writer = tf.summary.create_file_writer(logdir)


for step,data in enumerate(ds_train):
    print('Step: ', step)
    img = data[0]
    bb_img = data[1]
    bb = data[2]
    print(bb)
    with writer.as_default():
        tf.summary.image('train/image', img, step=step)
        tf.summary.image('train/bb_img', bb_img, step=step)
        tf.summary.scalar('step', step, step=step)
