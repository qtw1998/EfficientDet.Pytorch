import tensorflow as tf
import os

strategy = tf.distribute.MirroredStrategy()
print(strategy.num_replicas_in_sync)
