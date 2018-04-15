import tensorflow as tf
import os
import datetime
import json

import matplotlib.pyplot as plt
import io
import errno


def crete_file_writer(file, hparam_string=""):
    """creates FileWriter with name `.tensorboard-<file>`"""

    parent_folder = ".tensorboard-" + os.path.basename(file)
    run_number = get_and_increase_run_number(parent_folder)
    child_file = "/rn%s_%s" % (run_number, hparam_string)
    writer = tf.summary.FileWriter(parent_folder + child_file)
    return writer


def plot_summary(summary_name, xs, ys, styles):
    """creates an image summary containing plots"""
    assert len(xs) == len(ys), "xs and ys are different lengths"
    plt.figure()
    for (x, y, style) in zip(xs, ys, styles):
        plt.plot(x, y, style)
        plt.title(summary_name)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)  # Convert PNG buffer to TF image
        image = tf.expand_dims(image, 0)  # Add the batch dimension
        summary = tf.summary.image(summary_name, image)
    return summary


def get_and_increase_run_number(parent_folder):
    """reads and parses 'run.json' file. It bumps up the 'run_number' value in it. """
    file_name = parent_folder + '/run.json'
    next_run_number_key = "next_run_number"

    def initialize():
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

        with open(file_name, 'w') as f:
            data = {next_run_number_key: 0}
            json.dump(data, f)

    if not os.path.exists(file_name):
        initialize()

    is_file_empty = os.stat(file_name).st_size == 0
    if is_file_empty:
        initialize()

    with open(file_name, 'r+') as f:
        json_data = f.read()
        data = json.loads(json_data)
        run_number = data[next_run_number_key]
        data[next_run_number_key] = run_number + 1
        f.seek(0)  # truncates file
        json.dump(data, f)

    return run_number


def summate_kernel_verbose(variable_scope):
    """Plot all kernel values from dense layer. As well plot some stats"""
    with tf.variable_scope(variable_scope, reuse=True):
        kernel = tf.get_variable("kernel")
        cols = kernel.shape[0].value
        rows = kernel.shape[1].value

        tf.summary.scalar("kernel_mean", tf.reduce_mean(kernel), family="%s_stats" % variable_scope)
        tf.summary.scalar("kernel_min", tf.reduce_min(kernel), family="%s_stats" % variable_scope)
        tf.summary.scalar("kernel_max", tf.reduce_max(kernel), family="%s_stats" % variable_scope)
        tf.summary.histogram("kernel", kernel, family="%s_stats" % variable_scope)

        for c in range(cols):
            for r in range(rows):
                kernel_i = kernel[c, r]
                tf.summary.scalar('kernel_[%s,%s]' % (c, r), tf.reshape(kernel_i, []),
                                  family="%s_elements" % variable_scope)

    return None


def summate_bias_verbose(variable_scope):
    """Plot all bias values from dense layer. As well plot some stats"""
    with tf.variable_scope(variable_scope, reuse=True):
        bias = tf.get_variable("bias")
        cols = bias.shape[0].value

        tf.summary.scalar("bias_mean", tf.reduce_mean(bias), family="%s_stats" % variable_scope)
        tf.summary.scalar("bias_min", tf.reduce_min(bias), family="%s_stats" % variable_scope)
        tf.summary.scalar("bias_max", tf.reduce_max(bias), family="%s_stats" % variable_scope)
        tf.summary.histogram("bias", bias, family="%s_stats" % variable_scope)

        for c in range(cols):
            bias_i = bias[c]
            tf.summary.scalar('bias_[%s]' % (c), tf.reshape(bias_i, []),
                              family="%s_elements" % variable_scope)

    return None
