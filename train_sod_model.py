import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from sod_model import MSFE_Model
# Hack to get it to work with RTX cards
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load/configure dataset
(ds_train, ds_test), info = tfds.load(name='celeb_a', split=['train', 'test'], with_info=True)
print(info)
def imgr(sample):
    img = tf.cast(sample['image'], tf.float32)/255.0
    return (img, img)
    #return img
ds_train = ds_train.map(imgr, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


# Load model
model = MSFE_Model()
model.compile(optimizer='adam', loss='mse')

# Train model
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./logs/', update_freq='batch', histogram_freq=0, write_graph=False, write_images=True)
model.fit(x=ds_train, epochs=3, callbacks=[tbCallBack])

# Save model
model.save('./msfe_model.tfmodel')
