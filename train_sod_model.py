import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from sod_model import MSFE_Model

# Load dataset
ds, info = tfds.load(name='celeb_a', split=['train'], with_info=True)
print(info)

# Load model
model = MSFE_Model

# Train model
model.fit(x=ds, y=ds, epochs=1)

# Save model
model.save('./msfe_model.tfmodel')
