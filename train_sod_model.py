import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from sod_model import MSFE_Model

# Load/configure dataset
(ds_train, ds_test), info = tfds.load(name='celeb_a', split=['train', 'test'], with_info=True)
print(info)
def imgr(sample):
    img = tf.cast(sample['image'], tf.float32)/255.0
    return (img, img)
    #return img
ds_train = ds_train.map(imgr, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
#ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


# Load model
model = MSFE_Model()
model.compile(optimizer='adam', loss='mse')

# Train model
#ds_input = Input(tensor=
#model.fit(x=ds['image'], y=ds['image'], epochs=1)
model.fit(x=ds_train, epochs=1)

# Save model
model.save('./msfe_model.tfmodel')
