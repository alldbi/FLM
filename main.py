
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
from imutils import face_utils
from src.models import inception_resnet_v1
from utils.ThinPlateSpline2 import ThinPlateSpline2 as TPS
from utils.attack_util import purturb_GFLM



def parse_arguments(argv):
    parser = argparse.ArgumentParser()


    # face net args

    parser.add_argument('--pretrained_model', type=str,
                        default='/home/aldb/PycharmProjects/facenet-master/trained_model/b/20180408-102900/model-20180408-102900.ckpt-90',
                        help='Load a pretrained model before training starts.')

    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='src.models.inception_resnet_v1')

    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)

    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)

    parser.add_argument('--use_fixed_image_standardization',
                        help='Performs fixed standardization of images.', action='store_true')

    parser.add_argument('--prelogits_hist_max', type=float,
                        help='The max value for the prelogits histogram.', default=10.0)

    # dlib args
    parser.add_argument('--dlib_model', type=str,
                        default='/home/aldb/PycharmProjects/libs/dlib_pretrianed/shape_predictor_68_face_landmarks.dat',
                        help='Load the trained dlib model')



    # FLM and GFLM args

    parser.add_argument('--epsilon', type=float,
                        help='Coefficient of perturbation for each single step.', default=0.005)
    parser.add_argument('--img', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='/home/aldb/shared320/comparison/58_39s.pngorig.png')
    parser.add_argument('--label', type=int,
                        help='True label of the input image.',
                        default=58)
    parser.add_argument('--output_dir', type=str,
                        help='Directly where output files will be saved.',
                        default='./output/')
    parser.add_argument('--fixed_points', type=int,
                        help='Number of fixed points on each side of the image to force the edge condition on the transformation.', default=4)

    # public args
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=123)
    return parser.parse_args(argv)

def image_warping(img, lndA, lndB):
    CROP_SIZE = 182
    input_images_expanded = tf.reshape(img, [1, CROP_SIZE, CROP_SIZE, 3, 1])
    t_img, T, det = TPS(input_images_expanded, lndA, lndB, [CROP_SIZE, CROP_SIZE, 3])
    t_img = tf.reshape(t_img, [1, CROP_SIZE, CROP_SIZE, 3])
    return t_img, T

def main(args):


    # set random seed
    np.random.seed(seed=args.seed)

    # load dlib landmark detector model

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.dlib_model)

    # construct the model
    network = inception_resnet_v1
    y = tf.placeholder(tf.int32)

    # define the source and target landmark locations
    lnd_A = tf.placeholder(tf.float32, [None, 2])
    lnd_B = tf.placeholder(tf.float32, [None, 2])

    x = tf.placeholder(tf.float32, shape=[182, 182, 3])
    images = x
    images = tf.image.per_image_standardization(images)
    images = tf.reshape(images, [-1, 182, 182, 3])

    lnd_source = tf.expand_dims(lnd_A, axis=0)
    lnd_target = tf.expand_dims(lnd_B, axis=0)

    # deforme the input image using the transformation that maps source landmarks to the target landmarks
    images_deformed, T = image_warping(images, lnd_target, lnd_source)

    images_deformed = tf.image.per_image_standardization(images_deformed[0])
    images_deformed = tf.expand_dims(images_deformed, axis=0)

    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Trained model: %s' % pretrained_model)
    else:
        exit("A pretrained model should be provided!")

    tf.set_random_seed(args.seed)

    # Build the inference graph
    prelogits, cam_conv, _ = network.inference(images_deformed, 1.,
                                     phase_train=False, bottleneck_layer_size=512)
    logits = slim.fully_connected(prelogits, 10575, activation_fn=None,
                                  scope='Logits', reuse=False)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    softmax = tf.nn.softmax(logits, axis=1)
    grad = tf.gradients(loss, lnd_B)[0]*1.

    # Create a saver
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # load checkpoint
        if tf.train.checkpoint_exists(pretrained_model):
            print('Restoring pretrained model: %s' % pretrained_model)
            saver.restore(sess, pretrained_model)
        else:
            print ('There is no checkpoint to load!')

        # read input image

        img = cv2.imread(args.img)

        # convert color image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # extract landmarks
        rect = detector(gray, 1)[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # generate edge points to force the transformation to keep the boundary
        step = args.fixed_points
        new_w = 182
        steps = np.array(list(range(0, new_w, new_w//step)))
        b = list()
        for s in steps:
            b.append([0, s])
            b.append([s, 0])
            b.append([182, s])
            b.append([s, 182])
        b = np.array(b)
        b = b.reshape([-1, 2])
        shape = np.concatenate((shape, b), axis=0)
        lnd = np.copy(shape)

        # convert to rgb
        img = img[..., ::-1]

        # remove mean and scale pixel values
        img = (img-127.5)/128.

        # scale landmarks to [-1, 1]
        dp = np.copy((lnd/182.)*2.-1.)
        lnd = np.copy(dp)

        # initialize the landmark locations of the adversarial face image
        lnd_adv = np.copy(lnd)

        print ('True label:', args.label)
        for i in range(100):
            l, s, img_d, t, grad_ = sess.run([logits, softmax, images_deformed, T, grad], feed_dict={x: img, lnd_A:lnd, lnd_B:lnd_adv, y:[args.label]})

            print ("step: %02d, Predicted class: %05d, Pr(predicted class): %.4f, Pr(true class): %.4f" %
                   (i, np.argmax(l), s.max(), s[0, args.label]))


            if np.argmax(l) != args.label:
                break
            epsilon = args.epsilon

            lnd_adv = purturb_GFLM(lnd_adv, grad=grad_, epsilon=epsilon)


        img_d = img_d.reshape([182, 182, 3])

        def mapback(x):
            return (x + 1.) * 182. / 2.

        plt.subplot(131)
        plt.imshow((img - img.min()) / (img.max() - img.min()))
        plt.scatter(mapback(lnd[:, 0]), mapback(lnd[:, 1]))
        plt.subplot(132)
        plt.imshow((img_d - img_d.min()) / (img_d.max() - img_d.min()))
        plt.show()
        scipy.misc.imsave('adv_group.jpg', (img_d - img_d.min()) / (img_d.max() - img_d.min()))

        soa = np.array([[0, 0, 3, 2], [0, 0, 1, 1], [0, 0, 9, 9]])
        plt.figure()
        X, Y, U, V = zip(*soa)
        X = (lnd[0:68, 0] + 1.) * (91. + 20.) - 40.
        Y = (lnd[0:68, 1] + 1.) * (91. + 20.) - 40.
        U = (lnd_adv[0:68, 0] - lnd[0:68, 0]) * 620.
        V = (lnd_adv[0:68, 1] - lnd[0:68, 1]) * 620.
        ax = plt.gca()

        ax.imshow(np.ones([182, 182, 3]))
        # ax.scatter((lnd[:, 0]+1.)*91., (lnd[:, 1]+1.)*91., marker='.', c='blue')
        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='r',
                  width=0.005)  # , headwidth=1, headlength=1)
        # ax.imshow((img-img.min())/(img.max()-img.min()))
        ax.axis('off')
        plt.draw()
        plt.savefig('quiver_group.png', dpi=300, bbox_inches='tight')
        plt.show()

        exit()





if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
