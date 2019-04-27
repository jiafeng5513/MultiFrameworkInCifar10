import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

mnist_train = tfds.load(name="cifar10", split=tfds.Split.TRAIN)
mnist_example, = mnist_train.take(1)
image, label = mnist_example["image"], mnist_example["label"]

plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
plt.show()
print("Label: %d" % label.numpy())

mnist_train = mnist_train.repeat().shuffle(1024).batch(32)

# prefetch will enable the input pipeline to asynchronously fetch batches while
# your model is training.
mnist_train = mnist_train.prefetch(tf.data.experimental.AUTOTUNE)

for batch in mnist_train:
    images, labels = batch["image"], batch["label"]
    print(images.shape)