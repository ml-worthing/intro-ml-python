from uu import test

import tensorflow as tf
import random
from util import tf_help

import time
import numpy as np

class P:
    """global props so intellij can auto-complete after p.<ctrl+space>"""
    steps = 25000
    summaries_on_step = 50
    min_x = -10
    max_x = 10
    test_xs_step = 1.0
    number_of_samples = 15
    learning_rate = 1e-2
    seed = 122
    num_of_hidden_units = 1
    num_of_experiments = 10
    group_name = "beta"

    @staticmethod
    def make_param_string():
        return "lr%s_nos%s_hu%s_%s_" % (P.learning_rate, P.number_of_samples, P.num_of_hidden_units, P.group_name)


class Functions:
    """Example functions to approximate"""

    @staticmethod
    def f1(x):
        if x < 0:
            return 0.0
        else:
            return 1.0

    @staticmethod
    def f2(x):
        if x < 1.5:
            return 0.0
        else:
            return 1.0


def func_to_approx(x):
    """Function to approximate"""
    return Functions.f1(x)


def make_data():
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

        model = tf.layers.dense(inputs=features_reshaped,
                                activation=tf.sigmoid,
                                units=P.num_of_hidden_units,
                                name="in_layer")

        tf_help.summate_kernel_verbose("in_layer")
        tf_help.summate_bias_verbose("in_layer")

        model = tf.layers.dense(inputs=model,
                                activation=None,
                                units=1,
                                name="out_layer")
        tf_help.summate_kernel_verbose("out_layer")

        # with tf.variable_scope("predictions", reuse=True):
        #     bias = tf.get_variable("bias")
        #     # tf.summary.scalar('bias_s', tf.reshape(bias, []))
        #     tf.summary.histogram('bias_h', bias)
        #     kernel = tf.get_variable("kernel")
        #     for i in range(P.num_of_hidden_units):
        #         kernel_i = kernel[0, i]
        #         tf.summary.scalar('kernel_s_%s' % i, tf.reshape(kernel_i, []))
        #     tf.summary.histogram('kernel_h', kernel)
        #     tf.summary.tensor_summary('kernel_summary', kernel)

        # with tf.variable_scope("predictions_out", reuse=True):
        #     bias = tf.get_variable("bias")
        #     tf.summary.scalar('bias_out', tf.reshape(bias, []))
        #     kernel = tf.get_variable("kernel")
        #     # tf.summary.scalar('kernel_out', tf.reshape(kernel, []))

    with tf.name_scope("loss"):
        labels_reshaped = tf.reshape(labels, shape=[-1, 1], name="labels_r")
        loss = tf.losses.mean_squared_error(labels=labels_reshaped, predictions=model)
        tf.summary.scalar('loss', loss)

    with tf.name_scope("training"):
        optimiser = tf.train.GradientDescentOptimizer(P.learning_rate)
        minimise = optimiser.minimize(loss)

    with tf.name_scope("everything_else"):
        init = tf.global_variables_initializer()
        summaries = tf.summary.merge_all()

    print('[Done] Creating graph')


class S:
    (xs, ys, test_xs) = make_data()

    @staticmethod
    def run_experiment(experiment_no=None):

        writer = tf_help.crete_file_writer(__file__, P.make_param_string())
        writer.add_graph(tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(G.init)

            plot_data_summary = sess.run(tf_help.plot_summary('func_to_approx', S.xs, S.ys))
            writer.add_summary(plot_data_summary)
            for step in range(P.steps):
                if step % P.summaries_on_step == 0:
                    _, loss, summaries = sess.run((G.minimise, G.loss, G.summaries),
                                                  feed_dict={G.features: S.xs, G.labels: S.ys})
                    writer.add_summary(summaries, step)

                    predictions = sess.run(G.model, feed_dict={G.features: S.test_xs})
                    plot_data_summary = sess.run(
                        tf_help.plot_summary('predictions_step_%s' % step, S.test_xs, predictions))
                    writer.add_summary(plot_data_summary)

                    print("%s #%s/%s step:%d loss:%s" % (P.group_name, experiment_no, P.num_of_experiments, step, loss))
                else:
                    sess.run(G.minimise, feed_dict={G.features: S.xs, G.labels: S.ys})


P.group_name="zealot"
P.num_of_hidden_units=1

for experiment_no in range(P.num_of_experiments):
    S.run_experiment(experiment_no)


P.group_name="adept"
P.num_of_hidden_units=2

for experiment_no in range(P.num_of_experiments):
    S.run_experiment(experiment_no)

P.group_name="archon"
P.num_of_hidden_units=3

for experiment_no in range(P.num_of_experiments):
    S.run_experiment(experiment_no)

P.group_name="templar"
P.num_of_hidden_units=20

for experiment_no in range(P.num_of_experiments):
    S.run_experiment(experiment_no)

P.group_name="dragon"
P.num_of_hidden_units=100

for experiment_no in range(P.num_of_experiments):
    S.run_experiment(experiment_no)


time.sleep(2)  # wait for writer writes events to disk
print("Done")
