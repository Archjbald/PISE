'''
From https://github.com/tsc2017/Frechet-Inception-Distance
Code derived from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
Usage:
    Call get_fid(images1, images2)
Args:
    images1, images2: Numpy arrays with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary.
    dtype of the images is recommended to be np.uint8 to save CPU memory.
Returns:
    Frechet Inception Distance between the two image distributions.
'''

import tensorflow as tf

import os
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
# pip install tensorflow-gan
import tensorflow_gan as tfgan

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 64

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_FINAL_POOL = 'pool_3'


def inception_activations(images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    activations = tf.map_fn(
        fn=tfgan.eval.classifier_fn_from_tfhub(INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations


def get_inception_activations(inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    act = np.zeros([inps.shape[0], 2048], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] / 255. * 2 - 1
        logits = inception_activations(images=inp)
        act[i * BATCH_SIZE: i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = logits
    return act


def activations2distance(act1, act2):
    return tfgan.eval.frechet_classifier_distance_from_activations(act1, act2)


def get_fid(images1, images2):
    assert (type(images1) == np.ndarray)
    assert (len(images1.shape) == 4)
    if not images1.shape[1] == 3:
        images1 = np.moveaxis(np.array(images1), -1, 1)

    assert (images1.shape[1] == 3)
    assert (np.min(images1[0]) >= 0 and np.max(images1[0]) > 10), 'Image values should be in the range [0, 255]'
    assert (type(images2) == np.ndarray)
    assert (len(images2.shape) == 4)
    if not images2.shape[1] == 3:
        images2 = np.moveaxis(np.array(images2), -1, 1)

    assert (images2.shape[1] == 3)
    assert (np.min(images2[0]) >= 0 and np.max(images2[0]) > 10), 'Image values should be in the range [0, 255]'
    assert (images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
    print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(images1)
    act2 = get_inception_activations(images2)
    fid = activations2distance(act1, act2)
    print('FID calculation time: %f s' % (time.time() - start_time))
    return fid
