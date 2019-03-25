from uu import test

import tensorflow as tf

import random
import tf_help

import time
import numpy as np
import math

print(f"tensorflow version: {tf.VERSION}")
print(f"keras version: {tf.keras.__version__}")


class P:
    """global props so intellij can auto-complete after p.<ctrl+space>"""
    steps = 15000
    summaries_on_step = 100
    min_x = -10
    max_x = 10
    test_xs_step = 1.0
    number_of_samples = 150
    learning_rate = 1e-2
    seed = 122
    num_of_hidden_units = 1
    num_of_experiments = 1
    group_name = "beta"

    @staticmethod
    def make_param_string():
        return "lr%s_nos%s_hu%s_%s_" % (P.learning_rate, P.number_of_samples, P.num_of_hidden_units, P.group_name)


class Functions:
    """Example functions to approximate"""

    @staticmethod
    def f1(x):
        "signum function"
        if x < 0:
            return 0.0
        else:
            return 1.0

    @staticmethod
    def f2(x):
        "shifed f1"
        return Functions.f1(x + 1.2)

    @staticmethod
    def f3(x):
        "shifed and lifted f1"
        return Functions.f1(x + 1.2) + 0.4

    @staticmethod
    def sinlike(x):
        "increasing in amplitude sinus like"
        return math.sin(5.0 * x) * x + 0.7


def func_to_approx(x):
    """Function to approximate"""
    return Functions.f3(x)


def make_data():
    """creates arrays of numbers representing features and corresponsing labels"""
    r = random.Random(P.seed + 1)
    features = [r.uniform(P.min_x, P.max_x) for _ in range(P.number_of_samples)]
    labels = [func_to_approx(x) for x in features]
    test_features = [x for x in np.arange(P.min_x, P.max_x, P.test_xs_step)]
    return np.array([[x] for x in features]), np.array(labels), test_features


class M:
    """Create model"""

    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Dense(
            units=P.num_of_hidden_units,
            activation=tf.nn.tanh,  # https://www.tensorflow.org/api_guides/python/nn#Activation_Functions
            name="in_layer"
        )
    )

    model.add(
        tf.keras.layers.Dense(
            activation=None,
            units=1,
            name="out_layer"
        )
    )

    model.compile(
        optimizer=tf.train.AdamOptimizer(P.learning_rate),
        loss=tf.losses.mean_squared_error,
        metrics=['accuracy']
    )

#
# class G:
#     """Graph related static ops"""
#
#     print('[Start] Creating graph...')
#
#     features = tf.placeholder(dtype=tf.float32, shape=[None], name="features")
#     labels = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")
#
#     features_reshaped = tf.reshape(features, shape=[-1, 1], name="features_r")
#
#     with tf.name_scope("loss"):
#         labels_reshaped = tf.reshape(labels, shape=[-1, 1], name="labels_r")
#         loss = tf.losses.mean_squared_error(labels=labels_reshaped, predictions=model)
#         tf.summary.scalar('loss', loss)
#
#     with tf.name_scope("training"):
#         optimiser = tf.train.AdamOptimizer(P.learning_rate)
#         with tf.variable_scope("in_layer", reuse=True):
#             kernel = tf.get_variable("kernel")
#
#         grads = optimiser.compute_gradients(loss, kernel)
#         tf.summary.scalar("grads0", tf.reshape(grads[0][0], []))
#         tf.summary.scalar("grads1", tf.reshape(grads[0][1], []))
#
#     minimise = optimiser.minimize(loss)
#
#     with tf.name_scope("everything_else"):
#         init = tf.global_variables_initializer()
#         summaries = tf.summary.merge_all()
#
#     print('[Done] Creating graph')


class S:
    (xs, ys, test_xs) = make_data()

    @staticmethod
    def run_experiment(experiment_no=None):

        M.model.fit(
            x=S.xs,
            y=S.ys,
            batch_size=10,
            epochs=1
        )

        writer = tf_help.crete_file_writer(__file__, P.make_param_string())
        writer.add_graph(tf.get_default_graph())
        #
        # with tf.Session() as sess:
        #     sess.run(G.init)
        #
        #     for step in range(P.steps):
        #         if step % P.summaries_on_step == 0:
        #             _, loss, summaries = sess.run((G.minimise, G.loss, G.summaries),
        #                                           feed_dict={G.features: S.xs, G.labels: S.ys})
        #             writer.add_summary(summaries, step)
        #
        #             predictions = sess.run(G.model, feed_dict={G.features: S.test_xs})
        #             print("%s #%s/%s step:%d loss:%s" % (
        #                 P.make_param_string(), experiment_no, P.num_of_experiments, step, loss))
        #             if step % (2 * P.summaries_on_step) == 0:
        #                 plot_data_summary = sess.run(
        #                     tf_help.plot_summary(
        #                         summary_name='predictions_step_%s' % step,
        #                         xs=[S.xs, S.test_xs],
        #                         ys=[S.ys, predictions],
        #                         styles=['b.', 'r--'],
        #                         labels=['func_to_approx', 'approximation'],
        #                         loss=loss
        #                     ))
        #                 writer.add_summary(plot_data_summary)
        #
        #     else:
        #         sess.run(G.minimise, feed_dict={G.features: S.xs, G.labels: S.ys})


for experiment_no in range(P.num_of_experiments):
    S.run_experiment(experiment_no)

time.sleep(2)  # wait for writer writes events to disk
print("Done")
