import tensorflow as tf
import time
from util import tf_help

writer = tf_help.crete_file_writer(__file__)
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=(None, 3), name="x")

linear_model = tf.layers.Dense(
    units=1,
    kernel_initializer=tf.initializers.constant([0.1, 0.2, 0.3]),
    bias_initializer=tf.initializers.constant(5),
    name="dense"
)


y = linear_model(x) # linear_model.call(...)
#or using shortcut function

y1 = tf.layers.dense(
    x,
    units=1,
    kernel_initializer=tf.initializers.constant([0.1, 0.2, 0.3]),
    bias_initializer=tf.initializers.constant(5),
    name="dense_using_shortcut_function"
)

init = tf.global_variables_initializer()
sess.run(init)

# 0.1*1 + 0.2*2 + 0.3*3 + 5 = 6.4
print(
    sess.run((linear_model.kernel, y), feed_dict={x: [[1, 2, 3]]})
)

x_input = [
    [0, 0, 0],
    [1, 2, 3],
    [16, 212, 312322],
]

print(
    sess.run((linear_model.kernel, y), feed_dict={
        x: x_input
    })
)

writer.add_graph(tf.get_default_graph())
time.sleep(1)  # wait for writer writes events to disk
print("done")
