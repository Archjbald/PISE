'''
From https://github.com/tsc2017/Inception-Score
Code derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

Usage:
    Call get_inception_score(images, splits=10)
Args:
    images: A numpy array with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary. A dtype of np.uint8 is recommended to save CPU memory.
    splits: The number of splits of the images, default is 10.
Returns:
    Mean and standard deviation of the Inception Score across the splits.
'''

import sys
import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.python.ops import array_ops

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

import skimage.io
import six
import tensorflow_hub as tfhub

# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 64
INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'


def classifier_fn_from_tfhub(tfhub_module, output_fields, return_tensor=False):
    """Returns a function that can be as a classifier function.

    Wrapping the TF-Hub module in another function defers loading the module until
    use, which is useful for mocking and not computing heavy default arguments.

    Args:
      tfhub_module: A string handle for a TF-Hub module.
      output_fields: A string, list, or `None`. If present, assume the module
        outputs a dictionary, and select this field.
      return_tensor: If `True`, return a single tensor instead of a dictionary.

    Returns:
      A one-argument function that takes an image Tensor and returns outputs.
    """
    if isinstance(output_fields, six.string_types):
        output_fields = [output_fields]

    def _classifier_fn(images):
        output = tfhub.load(tfhub_module)(images)
        if output_fields is not None:
            output = {x: output[x] for x in output_fields}
        if return_tensor:
            assert len(output) == 1
            output = list(output.values())[0]
        return tf.nest.map_structure(tf.keras.layers.Flatten, output)

    return _classifier_fn


class ClassifierTFHub:
    def __init__(self, return_tensor=True):
        self.module = tfhub.load(INCEPTION_TFHUB)
        self.return_tensor = return_tensor

    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], np.float32)])
    def __call__(self, images):
        return tf.py_function(self.classify, [images], Tout=np.float32, name='classify')

    def classify(self, images):
        output = self.module(images)
        output = output['logits']
        output = tf.nest.map_structure(tf.keras.layers.Flatten(), output)
        return output


# classifier = ClassifierTFHub()
# tf.compat.v1.disable_eager_execution()
classifier = tfhub.load(INCEPTION_TFHUB)


@tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], np.float32)])
def classfy_fn(images):
    return tf.py_function(classify, [images], Tout=np.float32, name='classify')


def classify(images):
    output = classifier(images)
    output = output['logits']
    output = tf.nest.map_structure(tf.keras.layers.Flatten(), output)
    return output


def inception_logits(images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    logits = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(
            fn=classify,
            elems=array_ops.stack(generated_images_list),
            parallel_iterations=8,
            swap_memory=True,
            name='RunClassifier'))

    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits


def get_inception_probs(inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    preds = np.zeros([inps.shape[0], 1000], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255. * 2 - 1
        logits = inception_logits(images=inp)[:, :1000]
        preds[i * BATCH_SIZE: i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = logits
        drawProgressBar((i + 1) / n_batches)
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds


def drawProgressBar(percent, barLen=20):
    # percent float from 0 to 1.
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(barLen * percent), barLen, percent * 100))
    sys.stdout.flush()
    if percent > 1:
        sys.stdout.write('\n')
        sys.stdout.flush()


def preds2score(preds, splits=10):
    scores = []

    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
        drawProgressBar((i + 1) / splits)
    return np.mean(scores), np.std(scores)


def get_inception_score(images, splits=10):
    if not images.shape[1] == 3:
        images = np.moveaxis(np.array(images), -1, 1)

    assert (type(images) == np.ndarray)
    assert (len(images.shape) == 4)
    assert (images.shape[1] == 3)
    assert (np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'

    print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))
    start_time = time.time()
    preds = get_inception_probs(images)
    mean, std = preds2score(preds, splits)
    print('\nInception Score calculation time: %f s' % (time.time() - start_time))

    return mean, std  # Reference values: 11.38 for 50000 CIFAR-10 training set images, or mean=11.31, std=0.10 if in 10 splits.


if __name__ == '__main__':
    from metrics_benchmark import load_generated_images, get_args

    _, _, full_images, _ = load_generated_images(*get_args())

    IS = get_inception_score(full_images)
    print(IS)
