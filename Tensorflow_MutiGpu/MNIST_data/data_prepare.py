import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# 生成整数的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# MNIST数据集
mnist = input_data.read_data_sets('./RAW/', dtype=tf.uint8, one_hot=True)

if not os.path.exists('./prepared/'):
    os.makedirs('./prepared/')

filename_train = './prepared/train.tfrecords'
writer_train = tf.python_io.TFRecordWriter(filename_train)

# 将每张图片都转为一个Example
for i in range(mnist.train.num_examples):
    image_raw = mnist.train.images[i].tostring()  # 将图像转为字符串
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'pixels': _int64_feature(mnist.train.images.shape[1]),# 训练图像的分辨率，作为Example的属性
            'label': _int64_feature(np.argmax(mnist.train.labels[i])),
            'image_raw': _bytes_feature(image_raw)
        }))
    writer_train.write(example.SerializeToString())  # 将Example写入TFRecord文件

print('train data processing success')
writer_train.close()

filename_validation = './prepared/validation.tfrecords'
writer_validation = tf.python_io.TFRecordWriter(filename_validation)

# 将每张图片都转为一个Example
for i in range(mnist.validation.num_examples):
    image_raw = mnist.validation.images[i].tostring()  # 将图像转为字符串
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'pixels': _int64_feature(mnist.validation.images.shape[1]),# 训练图像的分辨率，作为Example的属性
            'label': _int64_feature(np.argmax(mnist.validation.labels[i])),
            'image_raw': _bytes_feature(image_raw)
        }))
    writer_validation.write(example.SerializeToString())  # 将Example写入TFRecord文件

print('validation data processing success')
writer_validation.close()

