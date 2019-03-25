
# basedf on https://www.tensorflow.org/guide/keras

import tensorflow as tf
from tensorflow._api.v1.keras import layers

print(f"tensorflow version: {tf.VERSION}")
print(f"keras version: {tf.keras.__version__}")

model = tf.keras.Sequential()

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))





