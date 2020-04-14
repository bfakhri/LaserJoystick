##############################################
# Trains the MSFE and Object Detection Model #
##############################################

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from sod_model import MSFE_Model
import os
# Hack to get it to work with RTX cards
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Params
batch_size = 16 
img_size = 640

# Load/configure dataset
(ds_train, ds_test), info = tfds.load(name='coco/2017', split=['train', 'test'], with_info=True)
print(info)
def imgr(sample):
    img = sample['image']
    img = tf.image.pad_to_bounding_box(img, 0, 0, img_size, img_size)
    img = tf.cast(img, tf.float32)/255.0
    return (img, img)
ds_train = ds_train.map(imgr, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


# Load feature extractor model, if it doesn't exist, trains it
model_path = './msfe_model/'
if(os.path.exists(model_path)):
    print('Loading Multi-Scale Feature Extractor')
    model = tf.keras.models.load_model(model_path)
else:
    print('Training Multi-Scale Feature Extractor from Scratch')
    model = MSFE_Model()
    model.compile(optimizer='adam', loss='mse')

    # Train model
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./logs/', update_freq='batch', histogram_freq=0, write_graph=False, write_images=True)
    model.fit(x=ds_train, epochs=3, callbacks=[tbCallBack])

    # Save model
    model.save(model_path)

# Instantiate and train object detection model

