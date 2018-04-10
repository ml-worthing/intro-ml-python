from uu import test

import tensorflow as tf
import random
from util import tf_help
import time
import numpy as np


def func_to_approx(x):
    """Function to approximate"""
    if x < 0:
        return 0.0
    else:
        return 1.0


class P:
    """global props so intellij can auto-complete after p.<ctrl+space>"""
    steps = 5250
    summaries_on_step = 20
    min_x = -10
    max_x = 10
    test_xs_step = 1.0
    number_of_samples = 15
    learning_rate = 1e-3
    seed = 122

    @staticmethod
    def makeParamString():
        return "lr%s_nos%s" % (P.learning_rate, P.number_of_samples)


def makeData():
    """creates arrays of numbers representing features and corresponsing labels"""
    r = random.Random(P.seed + 1)
    features = [r.uniform(P.min_x, P.max_x) for _ in range(P.number_of_samples)]
    labels = [func_to_approx(x) for x in features]
    test_features = [x for x in np.arange(P.min_x, P.max_x, P.test_xs_step)]
    return features, labels, test_features


class G:
    """Graph related static ops"""

    print('[Start] Creating graph...')

    features = tf.placeholder(dtype=tf.float32, shape=[None], name="features")
    labels = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")

    with tf.name_scope("model"):
        features_reshaped = tf.reshape(features, shape=[-1, 1], name="features_r")
        predictions = tf.layers.dense(inputs=features_reshaped, units=1, name="predictions")

    with tf.name_scope("loss"):
        labels_reshaped = tf.reshape(labels, shape=[-1, 1], name="labels_r")
        loss = tf.losses.mean_squared_error(labels=labels_reshaped, predictions=predictions)
        tf.summary.scalar('loss', loss)

    with tf.name_scope("training"):
        optimiser = tf.train.GradientDescentOptimizer(P.learning_rate)
        minimise = optimiser.minimize(loss)

    with tf.name_scope("everything_else"):
        init = tf.global_variables_initializer()
        summaries = tf.summary.merge_all()

    print('[Done] Creating graph')


writer = tf_help.crete_file_writer(__file__, P.makeParamString())
writer.add_graph(tf.get_default_graph())

(xs, ys, test_xs) = makeData()

with tf.Session() as sess:
    sess.run(G.init)

    plot_data_summary = sess.run(tf_help.plot_summary('func_to_approx', xs, ys))
    writer.add_summary(plot_data_summary)
    for step in range(P.steps):
        if step % P.summaries_on_step == 0:
            _, loss, summaries = sess.run((G.minimise, G.loss, G.summaries), feed_dict={G.features: xs, G.labels: ys})
            writer.add_summary(summaries, step)

            predictions = sess.run(G.predictions, feed_dict={G.features: test_xs})
            plot_data_summary = sess.run(tf_help.plot_summary('predictions_step_%s' % step, test_xs, predictions))
            writer.add_summary(plot_data_summary)

            print("Step %d %s" % (step, loss))
        else:
            sess.run(G.minimise, feed_dict={G.features: xs, G.labels: ys})

time.sleep(2)  # wait for writer writes events to disk
print("Done")
