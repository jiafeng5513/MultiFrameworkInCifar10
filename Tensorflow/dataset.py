from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

#tf.enable_eager_execution()

# mnist_train = tfds.load(name="cifar10", split=tfds.Split.TRAIN)
# assert isinstance(mnist_train, tf.data.Dataset)
#
# mnist_example, = mnist_train.take(1)
# image, label = mnist_example["image"], mnist_example["label"]
#
# plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
# plt.show()
# print("Label: %d" % label)


cifar_train = tfds.load(name="cifar10", split=tfds.Split.TRAIN)
cifar_train = cifar_train.repeat(1).shuffle(50000).batch(32)
cifar_train = cifar_train.prefetch(tf.data.experimental.AUTOTUNE)
p = 0
for batch in tfds.as_numpy(cifar_train):
    images, labels = batch["image"], batch["label"]
    mean, std = images.mean(), images.std()
    images = (images - mean) / std
    i=0
    p=p+1
    for lab in labels:
        if lab ==1:
            img=images[i]
            plt.imshow(img[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
            plt.show()
        i=i+1
    if p==20:
        break