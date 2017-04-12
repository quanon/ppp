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
flags.DEFINE_string('cascade_path',
                    './tmp/lbpcascade_animeface.xml', 'Path to cascade file.')

TRAIN_DIR = './data/train'
LOG_DIR = './log'
dirs = os.listdir(TRAIN_DIR)
CLASSES = [d for d in os.listdir(TRAIN_DIR)
           if os.path.isdir(os.path.join(TRAIN_DIR, d))]
PIXEL_COUNT = FLAGS.image_size * FLAGS.image_size * 3
CV_AA = 16


def detect(image):
  if not os.path.isfile(FLAGS.cascade_path):
    raise RuntimeError('%s not found' % FLAGS.cascade_path)

  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image_gray = cv2.equalizeHist(image_gray)
  cascade = cv2.CascadeClassifier(FLAGS.cascade_path)
  face_rects = cascade.detectMultiScale(image_gray,
                                        scaleFactor=1.1,
                                        minNeighbors=3,
                                        minSize=(28, 28))

  return face_rects


def get_face_images(image, face_rects):
  face_images = []

  for (x, y, w, h) in face_rects:
    face_image = image[y:y + h, x:x + w]
    face_image = cv2.resize(face_image, (FLAGS.image_size, FLAGS.image_size))
    face_image = face_image.flatten().astype(np.float32) / 255.0
    face_images.append(face_image)

  face_images = np.asarray(face_images)

  return face_images


def draw_prediction(image, face_rect, prediction):
  x, y, w, h = face_rect

  cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 204), 2)
  cv2.putText(image, CLASSES[prediction], (x, y),
              cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 102), 1, CV_AA)

  return image

test_image = cv2.imread(sys.argv[1])
face_rects = detect(test_image)
face_images = get_face_images(test_image, face_rects)

x = tf.placeholder(tf.float32, [None, PIXEL_COUNT])
labels = tf.placeholder(tf.float32, [None, len(CLASSES)])
keep_prob = tf.placeholder(tf.float32)

cnn = CNN(image_size=FLAGS.image_size, class_count=len(CLASSES))
y = cnn.inference(x, keep_prob, softmax=True)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess, os.path.join(LOG_DIR, 'model.ckpt'))

out_image = test_image.copy()

for i in range(len(face_images)):
  face_image = face_images[i]
  softmax = sess.run(y, feed_dict={x: [face_image], keep_prob: 1.0}).flatten()
  prediction = np.argmax(softmax)

  face_rect = face_rects[i]

  draw_prediction(out_image, face_rect, prediction)

  print(CLASSES[prediction], [n * 100 for n in softmax])

cv2.imwrite('tmp/out.png', out_image)
