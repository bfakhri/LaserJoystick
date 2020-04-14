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
    return (img, bb)
ds_train = ds_train.map(imgr, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# Make log directories
base_logdir = './logs/'
logdir = base_logdir + datetime.datetime.now().strftime('%d-%H%M%S')
writer = tf.summary.create_file_writer(logdir)


# Load feature extractor model, if it doesn't exist throw an error
msfe_path = './msfe_model/'
if(os.path.exists(msfe_path)):
    print('Loading Multi-Scale Feature Extractor')
    msfe = tf.keras.models.load_model(msfe_path)
else:
    print('Failed to find feature extractor model: '+msfe_path+'. Exiting....')
    print('Train the model by running \`python train_msfe_model.py\'')
    sys.exit()

# Instantiate and train object detection model
sod_model = SOD_Model()

for step,data in enumerate(ds_train):
    # Extract features
    features = msfe(data[0])
    with tf.GradientTape() as tape:
        # Do the object detection 
        detections = sod_model(features)
        # Calculate loss
        #loss = something...
        # Perform gradient update
    with writer.as_default():
        tf.summary.image('train/image', data[0], step=global_steps)
        tf.summary.scalar('step', step, step=global_steps)
