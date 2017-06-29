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
    'image', None, 'Test image')

FLAGS = tf.app.flags.FLAGS

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94



palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)

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

  channels = tf.split(image, num_channels, 2)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(channels, 2)


def _mean_image_subtraction2(image):
  means = np.array([123.68, 116.78, 103.94])
  return image - means


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=150,
        is_training=False)

    #####################################
    # Select the preprocessing function #
    #####################################
    input_image_ori = scipy.misc.imread(FLAGS.image)
    H, W = input_image_ori.shape[0], input_image_ori.shape[1]
    input_image = scipy.misc.imresize(input_image_ori, (224, 224))
    print('image.shape:', input_image.shape)

    image_X = tf.placeholder(tf.float32, input_image.shape)
    image = _mean_image_subtraction(image_X, [_R_MEAN, _G_MEAN, _B_MEAN])
    images = tf.expand_dims(image, axis=[0])
    #images = tf.expand_dims(image_X, axis=[0])

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)
    variables_to_restore = slim.get_variables_to_restore()

    for v in variables_to_restore:
      print(v)
    return

    # predictions = tf.argmax(logits, 1)
    predictions = tf.argmax(logits, 3)

    print('logits:', logits.get_shape())
    # print('predictions:', predictions.get_shape())

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

    logits = tf.image.resize_images(logits, (H, W)) 
    logit = sess.run(logits, feed_dict={image_X: input_image})[0]
    print(logit.shape)
    p = np.argmax(logit, axis=2)
    p = p.astype(np.uint8)
    print(np.unique(p))

    fig = plt.figure()
    ax = fig.add_subplot('121')
    ax.imshow(input_image_ori)
    ax = fig.add_subplot('122')
    ax.matshow(p, vmin=0, vmax=21, cmap=my_cmap)
    plt.show()

    for m in [3]: #np.unique(p):
        print(m)
        m = p == m
        m = m.astype(np.uint8)
        masked = input_image_ori * m[:,:,np.newaxis]

        fig = plt.figure()
        ax = fig.add_subplot('121')
        ax.imshow(input_image_ori)
        ax = fig.add_subplot('122')
        ax.imshow(masked) #, cmap='gray')
        plt.show()


if __name__ == '__main__':
  tf.app.run()
