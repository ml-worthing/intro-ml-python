import tensorflow as tf
import time
from util import tf_help
from collections import namedtuple

writer = tf_help.crete_file_writer(__file__)

# vec = tf.random_uniform(shape=(3, 2, 4))

vec = tf.placeholder(tf.float32, (3, 2), "vec")
#
out = vec + 1
out = out + 2
out = out + 2
# writer.add_graph(tf.get_default_graph())

sess = tf.Session()

print(
    sess.run(out, feed_dict={vec: [[1, 2], [3, 4], [5, 6]]})
)
print(
    out.eval(session=sess, feed_dict={vec: [[1, 2], [3, 4], [5, 6]]})
)


writer.add_graph(tf.get_default_graph())
time.sleep(1) #wait for writer writes events to disk
print("done")
