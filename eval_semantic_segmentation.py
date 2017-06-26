from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'ade20k', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', 'pspnet_v1_50', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'classes', None,
    'The classes to classify.')

FLAGS = tf.app.flags.FLAGS


def _get_label_mapping_tensor(classes, num_classes):
  tbl = np.zeros(num_classes + 1)
  for i, c in enumerate(classes):
    tbl[c] = i + 1
  return tf.constant(tbl, dtype=tf.int32)


def _filter_classes(labels, mapping):
  if mapping is None:
    return labels

  return tf.gather(mapping, labels)


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    num_classes = dataset.num_classes
    class_map = None
    if FLAGS.classes is not None:
      classes = [int(c) for c in FLAGS.classes.split(',')]
      class_map = _get_label_mapping_tensor(classes, dataset.num_classes)
      num_classes = len(classes) + 1

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=num_classes,
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image, label = image_preprocessing_fn(image, eval_image_size, eval_image_size,
                                          label=label)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
    labels = _filter_classes(labels, class_map)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)
    variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 3)
    labels = tf.squeeze(labels)
    predictions = tf.squeeze(predictions)

    # Define the metrics:
    pixel_acc = tf.contrib.metrics.streaming_accuracy(predictions, labels)
    mean_iou = tf.contrib.metrics.streaming_mean_iou(predictions, labels, num_classes)
    names_to_values, names_to_updates =\
        tf.contrib.metrics.aggregate_metric_map({
          'Pixel ACC': pixel_acc,
          'IOU': mean_iou
        })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master='',
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
