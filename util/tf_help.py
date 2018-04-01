import tensorflow as tf
import os

def crete_file_writer(file):
    """creates FileWriter with name `.tensorboard-<file>`"""
    writer = tf.summary.FileWriter(".tensorboard-" + os.path.basename(file))
    return writer

