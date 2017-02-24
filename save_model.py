from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import scipy
import scipy.misc as misc
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors

from datasets import dataset_factory
from nets import nets_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'model_name', 'pspnet_v1_50', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'output_dir', 'model', 'The directory where the protobuf be written')

tf.app.flags.DEFINE_string(
    'output_filename', 'pspnet_v1_50.pb', 'The filename of the protobuf')


FLAGS = tf.app.flags.FLAGS

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(2, num_channels, image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(2, channels)


def _mean_image_subtraction2(image):
  means = np.array([123.68, 116.78, 103.94])
  return image - means


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default() as g:
    tf_global_step = slim.get_or_create_global_step()

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=2,
        is_training=False)

    #####################################
    # Select the preprocessing function #
    #####################################
    image_X = tf.placeholder(tf.float32, (224, 224, 3), name='ph_input_x')
    image = _mean_image_subtraction(image_X, [_R_MEAN, _G_MEAN, _B_MEAN])
    images = tf.expand_dims(image, axis=[0])

    ####################
    # Define the model #
    ####################
    logits, ep = network_fn(images)
    variables_to_restore = slim.get_variables_to_restore()

    softmax = tf.nn.softmax(logits, name='softmax2')
    softmax = tf.reshape(softmax, shape=(-1,224,224,2), name='softmax')
    predictions = tf.argmax(softmax, 3, name='predictions')

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    sess = tf.Session()

    saver = tf.train.Saver(variables_to_restore)

    init_op = tf.group(
      tf.global_variables_initializer(),
      tf.local_variables_initializer())

    sess.run(init_op)
    saver.restore(sess, checkpoint_path)

    g2 = tf.graph_util.convert_variables_to_constants(sess, g.as_graph_def(),
                                                      ['softmax', 'predictions'])

    tf.train.write_graph(g2, FLAGS.output_dir, FLAGS.output_filename, as_text=False)


if __name__ == '__main__':
  tf.app.run()
