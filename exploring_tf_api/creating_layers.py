import tensorflow as tf
import time
from util import tf_help

writer = tf_help.crete_file_writer(__file__)

x = tf.placeholder(tf.float32, shape=(None, 3))
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)


writer.add_graph(tf.get_default_graph())
time.sleep(1) #wait for writer writes events to disk
print("done")
