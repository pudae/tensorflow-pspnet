from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import zipfile

import numpy as np
import tensorflow as tf
from six.moves import urllib

# The URL where the ADE20k data can be downloaded.
_DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'

# The number of shards per dataset split.
_NUM_SHARDS = 20


def download_and_uncompress_zip(zip_url, dataset_dir):
  """Downloads the `zip_url` and uncompresses it locally.
     From: https://github.com/tensorflow/models/blob/master/slim/datasets/dataset_utils.py

  Args:
    zip_url: The URL of a zip file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = zip_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()

  if tf.gfile.Exists(filepath):
    print('Zip file already exist. Skip download..', filepath)
  else:
    filepath, _ = urllib.request.urlretrieve(zip_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  with zipfile.ZipFile(filepath) as f:
    print('Extracting ', filepath)
    f.extractall(dataset_dir)
    print('Successfully extracted')


class ImageReader(object):
  def __init__(self):
    self._decode_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_data, channels=3)
    self._decode_png = tf.image.decode_png(self._decode_data)

  def read_jpeg_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape

  def read_png_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
    return image


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'ade20k_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _get_image_filenames(image_dir):
  return sorted(os.listdir(image_dir))


def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _to_tfexample(image_data, image_format, label_data, label_format, height, width):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_feature(image_data),
      'image/format': _bytes_feature(image_format),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'label/encoded': _bytes_feature(label_data),
      'label/format': _bytes_feature(label_format),
      'label/height': _int64_feature(height),
      'label/width': _int64_feature(width),
  }))


def _add_to_tfrecord(record_dir, image_dir, annotation_dir, split_name):
  """Loads image files and writes files to a TFRecord.
  """

  assert split_name in ['training', 'validation']

  image_dir = os.path.join(image_dir, split_name)
  annotation_dir = os.path.join(annotation_dir, split_name)

  filenames = zip(_get_image_filenames(image_dir),
                  _get_image_filenames(annotation_dir))
  # All matching files must have same name
  assert all([x[:-4] == y[:-4] for x, y in filenames])

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        record_filename = _get_dataset_filename(record_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d\n' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_filename, label_filename = filenames[i]
            image_filename = os.path.join(image_dir, image_filename)
            label_filename = os.path.join(annotation_dir, label_filename)

            image_data = tf.gfile.FastGFile(image_filename, 'r').read()
            label_data = tf.gfile.FastGFile(label_filename, 'r').read()
            height, width, depth = image_reader.read_jpeg_dims(sess, image_data)
            height, width, depth = image_reader.read_png_dims(sess, label_data)

            example = _to_tfexample(
                image_data, 'jpg', label_data, 'png', height, width)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  download_and_uncompress_zip(_DATA_URL, dataset_dir)

  image_dir      = os.path.join(dataset_dir, 'ADEChallengeData2016/images')
  annotation_dir = os.path.join(dataset_dir, 'ADEChallengeData2016/annotations')
  record_dir     = os.path.join(dataset_dir, 'records')

  if not tf.gfile.Exists(record_dir):
    tf.gfile.MakeDirs(record_dir)

  # process the training, validation data:
  _add_to_tfrecord(record_dir, image_dir, annotation_dir, 'training')
  _add_to_tfrecord(record_dir, image_dir, annotation_dir, 'validation')

  print('\nFinished converting the Ade20k dataset!')
