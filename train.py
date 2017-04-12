import os
import re
import numpy as np
import cv2
import tensorflow as tf
from cnn import CNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('image_size', 48, 'Image size.')
flags.DEFINE_integer('step_count', 50, 'Number of steps.')
flags.DEFINE_integer('batch_size', 50, 'Batch size.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

TRAIN_DIR = './data/train'
TEST_DIR = './data/test'
LOG_DIR = './log'
dirs = os.listdir(TRAIN_DIR)
CLASSES = [d for d in os.listdir(TRAIN_DIR)
           if os.path.isdir(os.path.join(TRAIN_DIR, d))]
PIXEL_COUNT = FLAGS.image_size * FLAGS.image_size * 3


def fetch_images_and_labels(dir):
  images = []
  labels = []

  for i, klass in enumerate(CLASSES):
    image_dir = os.path.join(dir, klass)
    files = [f for f in os.listdir(image_dir) if re.match(r'.*\.jpg', f)]

    for f in files:
      image = cv2.imread(os.path.join(dir, klass, f))
      image = cv2.resize(image, (FLAGS.image_size, FLAGS.image_size))
      image = image.flatten().astype(np.float32) / 255.0
      images.append(image)

      label = np.zeros(len(CLASSES))
      label[i] = 1
      labels.append(label)

  return np.asarray(images), np.asarray(labels)


def shaffle_images_and_labels(images, labels):
    assert len(images) == len(labels)
    permutation = np.random.permutation(len(images))

    return images[permutation], labels[permutation]

train_images, train_labels = fetch_images_and_labels(TRAIN_DIR)
train_images, train_labels = shaffle_images_and_labels(train_images,
                                                       train_labels)

test_images, test_labels = fetch_images_and_labels(TEST_DIR)
test_images, test_labels = shaffle_images_and_labels(test_images, test_labels)

cnn = CNN(image_size=FLAGS.image_size, class_count=len(CLASSES))

with tf.Graph().as_default():
  x = tf.placeholder(tf.float32, [None, PIXEL_COUNT])
  labels = tf.placeholder(tf.float32, [None, len(CLASSES)])
  keep_prob = tf.placeholder(tf.float32)

  y = cnn.inference(x, keep_prob)
  v = cnn.cross_entropy(y, labels)
  train_step = cnn.train_step(v, FLAGS.learning_rate)
  accuracy = cnn.accuracy(y, labels)

  saver = tf.train.Saver()
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    for i in range(FLAGS.step_count):
      for j in range(int(len(train_images) / FLAGS.batch_size)):
        batch = FLAGS.batch_size * j
        sess.run(train_step, feed_dict={
                 x: train_images[batch:batch + FLAGS.batch_size],
                 labels: train_labels[batch:batch + FLAGS.batch_size],
                 keep_prob: 0.5})

      train_accuracy = sess.run(accuracy, feed_dict={
                                x: train_images,
                                labels: train_labels,
                                keep_prob: 1.0})

      print('step %d: training accuracy %g' % (i, train_accuracy))

      summary = sess.run(summary_op, feed_dict={
                         x: train_images,
                         labels: train_labels,
                         keep_prob: 1.0})
      summary_writer.add_summary(summary, i)

    test_accuracy = sess.run(accuracy, feed_dict={
                             x: test_images,
                             labels: test_labels,
                             keep_prob: 1.0})

    print('test accuracy %g' % test_accuracy)

    save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
