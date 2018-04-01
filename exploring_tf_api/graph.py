import tensorflow as tf
import time
from util import tf_help
from collections import namedtuple

writer = tf_help.crete_file_writer(__file__)

a = tf.constant(3.0, dtype=tf.float32, name="a")
b = tf.constant(4.0, name="b")  # type implicitly float32
c = tf.constant(6.0, dtype=tf.float32, name="c")

with tf.name_scope("addition"):
    total = a + b

with tf.name_scope("multiplication"):
    total = total * c

print(a, type(a))
print(b, type(b))
print(total, type(total))
# Tensor("Const:0", shape=(), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'>
# Tensor("Const_1:0", shape=(), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'>
# Tensor("add:0", shape=(), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'>

sess = tf.Session()

with sess.as_default():
    print(total.eval())

# other ways of computing
print(sess.run(total))
print(total.eval(session=sess))
print(sess.run((a, b, total)))
print(sess.run({'a': a, 'b': b, 'total': total}))
print(sess.run({'ab': (a, b), 'total': total}))
print(sess.run({'abtotal': (a, b, total)}))
Abtotal = namedtuple('Abtotal', ['a', 'b', 'total'], verbose=False)
print(sess.run(Abtotal(a, b, total)))

writer.add_graph(tf.get_default_graph())
time.sleep(1) #wait for writer writes events to disk
print("done")
