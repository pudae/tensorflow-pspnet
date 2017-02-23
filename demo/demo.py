import os
import sys
from StringIO import StringIO

import configargparse
import cv2
import logging
import numpy as np
import scipy
import scipy.misc as misc
import tensorflow as tf
import tornado
import tornado.httpserver as httpserver
import tornado.web as web


app = None # pylint: disable=invalid-name


try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    CURRENT_DIR = os.getcwd()


def setup_log(dest=None):
    root = logging.getLogger("")
    formatter = logging.Formatter("%(asctime)s %(message)s", "%m/%d %H:%M:%ST%z")
    root.handlers = []

    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if dest:
        handlers.append(logging.FileHandler(dest))

    for handler in handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(logging.INFO)


# class PredictHandler(APIHandler):
#     @gen.coroutine
#     def post(self):
#         try:
#             input_ = escape.json_decode(self.request.body)
#         except ValueError:
#             self.finish()
#             self.write_error(http.client.BAD_REQUEST, msg="JSON Decode Failed") # pylint: disable=no-member
#             return
# 
#         if not input_:
#             self.write_error(http.client.BAD_REQUEST, msg="Empty Request") # pylint: disable=no-member
#             self.finish()
#             return
#         loop = asyncio.get_event_loop()
# 
#         input_size = len(input_)
#         pos = 0
# 
#         results = []
#         while pos < input_size:
#             result = yield from loop.run_in_executor(
#                 None,
#                 lambda: app.predict(input_[pos:pos+app.batch_size]))
#             results.extend(result)
#             pos += app.batch_size
# 
#         self.set_header('Content-Type', 'application/json')
#         self.write(json.dumps(results, ensure_ascii=False))
#         self.finish()
#         return


class EchoHandler(web.RequestHandler):
  def initialize(self, app):
    self._app = app

  def get(self, texts):
    self.write(self._app.echo(texts))


class TestUIHandler(web.RequestHandler):
  def initialize(self, app):
    self._app = app


  def get(self):
    self.render("template.html", title="My title")


  def post(self):
    print('post!!!!!!!!!!!!!!!!!!')
    req = self.request
    image = req.files['image'][0]['body']
    print(type(image))
    image = misc.imread(StringIO(image))
    r = self._app.predict(image)
    print(type(r))
    # self.write(r)
    self.write(image.tobytes())
    self.set_header("Content-type", req.files['image'][0]['content_type'])
    #self.set_header("Content-type", "image/png")



class App(object):
  def __init__(self, config):

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    self._graph = tf.Graph()
    self._sess = tf.Session(config=tf_config, graph=self._graph)

    with self._sess.graph.as_default():
        graph_def = tf.GraphDef()
        with open('./skynet_v1_50_graph.pb', 'rb') as file:
            graph_def.ParseFromString(file.read())
        tf.import_graph_def(graph_def, name="")

    self._input_x = self._sess.graph.get_operation_by_name('ph_input_x').outputs[0]
    self._pred = self._sess.graph.get_operation_by_name('predictions').outputs[0]

    self._http_app = web.Application(
        handlers=[
            web.URLSpec(r"/api/echo/(.*)", EchoHandler, dict(app=self)),
            web.URLSpec(r"/api/image", EchoHandler, dict(app=self)),
            web.URLSpec(r"/ui", TestUIHandler, dict(app=self))
        ],
        debug=config["debug"],
    )


  def http_app(self):
    return self._http_app


  def echo(self, texts):
      return texts


  def post(self, image):
      return texts


  def predict(self, image):
    H, W = image.shape[0], image.shape[1]
    input_image = scipy.misc.imresize(image, (224, 224))

    p = self._sess.run(self._pred, feed_dict={self._input_x: input_image})[0]
    p = misc.imresize(p, (H, W), interp='nearest')
    b = p.tobytes()
    # print(b)
    # print(type(b))
    # print(len(b))
    # results = misc.imread(StringIO(b), mode='L')
    #return results
    return b


def main():
  """main"""
  parser = configargparse.ArgParser(
      default_config_files=[
          os.path.join(CURRENT_DIR, "server.conf"),
          "server.conf",
          "/etc/skynet/server.conf"])

  parser.add("--debug", dest="debug", default=False, action="store_true")
  parser.add("--no-debug", dest="debug", action="store_false")
  parser.add("--log", dest="log", default="")

  parser.add("--host", dest="host", default=os.environ.get("BIND", "127.0.0.1"))
  parser.add("--port", dest="port", type=int, default=int(os.environ.get("PORT", 80)))

  # parser.add("--model", dest="model", required=True)
  # parser.add("--batch-size", dest="batch_size", type=int, required=True)

  # parser.add("--gpu", dest="gpu", default=True, action="store_true")
  # parser.add("--no-gpu", dest="gpu", action="store_false")
  # parser.add("--gpu-memory-fraction", type=float, default=0.95, dest="gpu_memory_fraction")

  config = vars(parser.parse_args())
  setup_log(config["log"])

  logging.info("config: %s", config)

  global app # pylint: disable=invalid-name
  app = App(config)

  server = httpserver.HTTPServer(app.http_app())
  server.bind(config["port"], address=config["host"])
  server.start()
  logging.info("Server Start! Listen: %s", [x.getsockname() for x in server._sockets.values()]) # pylint: disable=protected-access
  tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
