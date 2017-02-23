
import numpy as np
import tensorflow as tf
import scipy
import scipy.misc as misc

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors

tf.app.flags.DEFINE_string(
    'image', None, 'Test image')

FLAGS = tf.app.flags.FLAGS

palette = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.0),
           (0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5), (0.5, 0.5, 0.5),
           (0.25, 0.0, 0.0), (0.75, 0.0, 0.0), (0.25, 0.5, 0.0), (0.75, 0.5, 0.0),
           (0.25, 0.0, 0.5), (0.75, 0.0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5),
           (0.0, 0.25, 0.0), (0.5, 0.25, 0.0), (0.0, 0.75, 0.0), (0.5, 0.75, 0.0),
           (0.0, 0.25, 0.5)]
my_cmap = mpl_colors.LinearSegmentedColormap.from_list('Custom cmap', palette, 21)

def main(_):

  g = tf.Graph()
  sess = tf.Session(graph=g)

  with sess.graph.as_default():
      graph_def = tf.GraphDef()
      with open('./train/skynet_v1_50_graph.pb', 'rb') as file:
          graph_def.ParseFromString(file.read())
      tf.import_graph_def(graph_def, name="")

  input_x = sess.graph.get_operation_by_name('ph_input_x').outputs[0]
  pred = sess.graph.get_operation_by_name('predictions').outputs[0]

  input_image_ori = scipy.misc.imread(FLAGS.image)
  H, W = input_image_ori.shape[0], input_image_ori.shape[1]

  input_image = scipy.misc.imresize(input_image_ori, (224, 224))

  import time
  before = time.time()
  p = sess.run(pred, feed_dict={input_x: input_image})[0]
  print(time.time() - before)

  before = time.time()
  p = sess.run(pred, feed_dict={input_x: input_image})[0]
  print(time.time() - before)

  fig = plt.figure()
  ax = fig.add_subplot('121')
  ax.imshow(input_image_ori)
  ax = fig.add_subplot('122')
  ax.imshow(p)
  plt.show()


if __name__ == '__main__':
  tf.app.run()
