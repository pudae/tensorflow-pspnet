from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

slim = tf.contrib.slim

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """
  """

class Level(collections.namedtuple('Level', ['scope', 'fn', 'args'])):
  """
  """

def subsample(inputs, factor, scope=None):
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None):
  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):

        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          unit_depth, unit_depth_bottleneck, unit_stride, unit_rate = unit

          net = block.unit_fn(net,
                              depth=unit_depth,
                              depth_bottleneck=unit_depth_bottleneck,
                              stride=unit_stride,
                              rate=unit_rate)
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net


@slim.add_arg_scope
def pyramid_pooling_module(inputs, levels, outputs_collections=None):
  with tf.variable_scope('pyramid_pool_module', [inputs]) as sc:
    level_maps = [inputs]
    for level in reversed(levels):
      with tf.variable_scope(level.scope, 'level', [inputs]) as sc:
        level_size, level_depth = level.args
        level_map = level.fn(inputs, level_size, level_depth)
        level_maps.append(level_map)

    net = tf.concat(axis=3, values=level_maps)
    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net


def pspnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):

  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc

