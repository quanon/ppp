import os
import re
import numpy as np
import cv2
import tensorflow as tf

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
CLASSES = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
CLASS_COUNT = len(CLASSES)

COLOR_CHANNEL_COUNT = 3
PIXEL_COUNT = FLAGS.image_size * FLAGS.image_size * COLOR_CHANNEL_COUNT


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

      label = np.zeros(CLASS_COUNT)
      label[i] = 1
      labels.append(label)

  return np.asarray(images), np.asarray(labels)


def shaffle_images_and_labels(images, labels):
    assert len(images) == len(labels)
    permutation = np.random.permutation(len(images))

    return images[permutation], labels[permutation]


def inference(x, keep_prob):
  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  x_image = tf.reshape(x, [-1, FLAGS.image_size, FLAGS.image_size, COLOR_CHANNEL_COUNT])

  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, COLOR_CHANNEL_COUNT, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([int(FLAGS.image_size / 4) * int(FLAGS.image_size / 4) * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(
      h_pool2, [-1, int(FLAGS.image_size / 4) * int(FLAGS.image_size / 4) * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, CLASS_COUNT])
    b_fc2 = bias_variable([CLASS_COUNT])
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return y


def cross_entropy(y, labels):
  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels))
  tf.summary.scalar('cross_entropy', cross_entropy)

  return cross_entropy


def train_step(cross_entropy):
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

  return train_step


def accuracy(y, labels):
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  return accuracy


train_images, train_labels = shaffle_images_and_labels(
  *fetch_images_and_labels(TRAIN_DIR))
test_images, test_labels = shaffle_images_and_labels(
  *fetch_images_and_labels(TEST_DIR))

with tf.Graph().as_default():
  x = tf.placeholder(tf.float32, [None, PIXEL_COUNT])
  labels = tf.placeholder(tf.float32, [None, CLASS_COUNT])
  keep_prob = tf.placeholder(tf.float32)

  y = inference(x, keep_prob)
  v = cross_entropy(y, labels)
  train_step = train_step(v)
  accuracy = accuracy(y, labels)

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
