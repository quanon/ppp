import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from cnn import CNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('image_size', 48, 'Image size.')

TRAIN_DIR = './data/train'
LOG_DIR = './log'
dirs = os.listdir(TRAIN_DIR)
CLASSES = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
PIXEL_COUNT = FLAGS.image_size * FLAGS.image_size * 3


test_images = []

for i in range(1, len(sys.argv)):
  image = cv2.imread(sys.argv[i])
  image = cv2.resize(image, (FLAGS.image_size, FLAGS.image_size))
  test_images.append(image.flatten().astype(np.float32) / 255.0)

test_images = np.asarray(test_images)

x = tf.placeholder(tf.float32, [None, PIXEL_COUNT])
labels = tf.placeholder(tf.float32, [None, len(CLASSES)])
keep_prob = tf.placeholder(tf.float32)

cnn = CNN(image_size=FLAGS.image_size, class_count=len(CLASSES))
y = cnn.inference(x, keep_prob, softmax=True)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess, os.path.join(LOG_DIR, 'model.ckpt'))

for i in range(len(test_images)):
  softmax = sess.run(y, feed_dict={x: [test_images[i]], keep_prob: 1.0}).flatten()
  prediction = np.argmax(softmax)

  print(softmax)
  print(CLASSES[prediction])
