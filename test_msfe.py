import tensorflow as tf
import numpy as np
import datetime
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
#ds_train = ds_train.cache()
#ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


# Load model
model = tf.keras.models.load_model('./msfe_model/')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/audit/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

for step,data in enumerate(ds_train):
    img = data[0]
    output = model(img, training=True)
    with train_summary_writer.as_default():
        tf.summary.image('AE_Orig', img, step=step)
        tf.summary.image('AE_Output', output, step=step)


