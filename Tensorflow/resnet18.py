# -*- coding: UTF-8 -*-
import tensorflow as tf
import utils
import numpy as np
import scipy.io as sio
import cv2
import cifar10_input
import time
import math
import os
from datetime import datetime


log_frequency = 50
batch_size = 32  # define for loggerHook


class Resnet18:
    # 初始化
    def __init__(self, data_dir, boundaries, learing_rates, batch_size=32, epochs=200,
                 number_train_samples=50000, number_test_samples=10000, checkpointdir="./checkpoint"):
        self.global_step = 0
        self.boundaries = boundaries
        self.learing_rates = learing_rates
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.epochs = epochs
        self.number_train_samples = number_train_samples
        self.number_test_samples = number_test_samples
        self.checkpointdir = checkpointdir
        self.max_step = self.number_train_samples * self.epochs
        self.MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.

    # 定义卷积
    def Convolution(self, input_, output_dim, kernel_height=1, kernel_width=1,
                    stride_height=1, stride_width=1, with_blas=False, name="w"):
        """
        :param input_:       输入变量
        :param output_dim:   输出维数
        :param kernel_height:卷积核尺寸h
        :param kernel_width: 卷积核尺寸w
        :param stride_height: 步长h
        :param stride_width:  步长w
        :param name:         name
        :param with_blas:    是否使用blas
        :return:
        """
        # 权重随机初始化的标准差
        stddev = np.sqrt(2.0 / (kernel_height * kernel_width * input_.get_shape().as_list()[-1] * output_dim))
        # 权重
        w = tf.get_variable(name, [kernel_height, kernel_width, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        # 卷积
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_height, stride_width, 1], padding="SAME")
        # blas
        if with_blas:
            bias = tf.get_variable([output_dim], initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(conv, bias)
        else:
            return conv

    # BatchNormalization
    def BatchNormalization(self, input, is_training):
        """
        BatchNormalization
        :param input:输入变量
        :param is_training:是否在训练
        :return:
        """
        conv_bn = tf.layers.batch_normalization(input, training=is_training, epsilon=1e-5)
        return conv_bn

    # 定义Residual Block V1
    def ResidualBlockV1(self, input, output_dim, is_training=True, kernel_height=3, kernel_width=3, strides=1, name="res"):
        """
        Residual Block V1,一路上是两个串联的卷积核BN,另一路是直通
        :param input:
        :param output_dim:
        :param is_training:
        :param kernel_height:
        :param kernel_width:
        :param strides:
        :param name:
        :return:
        """
        with tf.variable_scope(name+"_A"):
            c1 = self.Convolution(input, output_dim, kernel_height, kernel_width, strides, strides, name=name+"_A")
            bn1 = self.BatchNormalization(c1, is_training)
            r1 = tf.nn.relu(bn1)
        with tf.variable_scope(name+"_B"):
            c2 = self.Convolution(r1, output_dim, kernel_height, kernel_width, strides, strides, name=name+"_B")
            bn2 = self.BatchNormalization(c2, is_training)
            plus = tf.add_n([input, bn2])
        return tf.nn.relu(plus)

    # 定义Residual Block V2
    def ResidualBlockV2(self, input, output_dim, is_training=True, kernel_height=3, kernel_width=3, strides=1, name="res"):
        """
        Residual Block V2,一路上是两个串联的卷积核BN,另一路是一个卷积和BN
        :param input:           输入
        :param output_dim:      输出维数
        :param is_training:     是否在训练
        :param kernel_height:   卷积核尺寸h
        :param kernel_width:    卷积核尺寸w
        :param strides:         步长,两个方向相等
        :param name: 
        :return:
        """
        with tf.variable_scope(name+"_A"):
            c1 = self.Convolution(input, output_dim, kernel_height, kernel_width, strides, strides, name=name+"_A")
            bn1 = self.BatchNormalization(c1, is_training)
            r1 = tf.nn.relu(bn1)
        with tf.variable_scope(name+"_B"):
            c2 = self.Convolution(r1, output_dim, kernel_height, kernel_width, strides, strides, name=name+"_B")
            bn2 = self.BatchNormalization(c2, is_training)
        with tf.variable_scope(name+"_C"):
            c3 = self.Convolution(input, output_dim, kernel_height, kernel_width, strides, strides, name=name+"_C")
            bn3 = self.BatchNormalization(c3, is_training)
            plus = tf.add_n([bn3, bn2])

        return tf.nn.relu(plus)

    # 定义网络
    def inference(self, x, istraining):
        """
        构造整个网络
        :param x: 输入
        :return:  输出
        """
        with tf.name_scope("Conv1"):
            c1 = self.Convolution(x, 16, 3, 3, 1, 1, with_blas=False, name="conv1")
            bn1 = self.BatchNormalization(c1, istraining)
            conv1 = tf.nn.relu(bn1)
        with tf.name_scope("res1"):
            res1 = self.ResidualBlockV1(conv1, 16, istraining, name="res1")
        with tf.name_scope("res2"):
            res2 = self.ResidualBlockV1(res1, 16, istraining, name="res2")
        with tf.name_scope("res3"):
            res3 = self.ResidualBlockV1(res2, 16, istraining, name="res3")
        with tf.name_scope("res4"):
            res4 = self.ResidualBlockV2(res3, 32, istraining, name="res4")
        with tf.name_scope("res5"):
            res5 = self.ResidualBlockV1(res4, 32, istraining, name="res5")
        with tf.name_scope("res6"):
            res6 = self.ResidualBlockV1(res5, 32, istraining, name="res6")
        with tf.name_scope("res7"):
            res7 = self.ResidualBlockV2(res6, 64, istraining, name="res7")
        with tf.name_scope("res8"):
            res8 = self.ResidualBlockV1(res7, 64, istraining, name="res8")
        with tf.name_scope("res9"):
            res9 = self.ResidualBlockV1(res8, 64, istraining, name="res10")
        with tf.name_scope("out"):
            act = tf.nn.relu(res9)
            avp = tf.nn.avg_pool(act, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME", name="AveragePooling")
            fln = tf.layers.flatten(avp, name="Flatten")
            output = tf.layers.dense(fln, activation='softmax', units=10, use_bias=True, kernel_initializer='he_normal',
                                 name="output")
        return output

    # 计算学习率
    def get_Learning_rate(self, step):
        if step < self.boundaries[0]:
            return self.learing_rates[0]
        elif step >= self.boundaries[0] and step < self.boundaries[1]:
            return self.learing_rates[1]
        elif step >= self.boundaries[1] and step < self.boundaries[2]:
            return self.learing_rates[2]
        elif step >= self.boundaries[2] and step < self.boundaries[3]:
            return self.learing_rates[3]
        elif step >= self.boundaries[3]:
            return self.learing_rates[4]

    # 定义损失函数
    def loss(self, logits, labels):
        '''
        计算CNN的loss
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
        # tf.reduce_mean对cross entropy计算均值
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        # tf.add_to_collection:把cross entropy的loss添加到整体losses的collection中
        tf.add_to_collection('losses', cross_entropy_mean)
        # tf.add_n将整体losses的collection中的全部loss求和得到最终的loss
        return tf.add_n(tf.get_collection('losses'), name='total_loss')
        #return cross_entropy_mean

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in CIFAR-10 model.

        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.

        Args:
          total_loss: Total loss from loss().
        Returns:
          loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op

    # 定义Trainer
    def trainer(self, total_loss, global_step):
        """Train CIFAR-10 model.

          Create an optimizer and apply to all trainable variables. Add moving
          average for all trainable variables.

          Args:
            total_loss: Total loss from loss().
            global_step: Integer Variable counting the number of training steps
              processed.
          Returns:
            train_op: op for training.
          """
        # 一个epoch含有多少batch
        num_batches_per_epoch = int(self.number_train_samples / self.batch_size)
        #decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        """
        self.boundaries中含有学习率衰减的epoch界限
        tf的常数衰减需要给出衰减的step界限
        x个epoch含有多少个step呢?
        x * 每个epoch的batch数 *batch_size
        """
        boundaries_in_steps=[item * num_batches_per_epoch * self.batch_size for item in self.boundaries]
        lr = tf.train.piecewise_constant(global_step, boundaries=boundaries_in_steps, values=self.learing_rates)

        tf.summary.scalar('learning_rate', lr)

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self._add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            self.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op
        pass

    # 训练
    def train(self):
        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()
            with tf.device('/cpu:0'):
                images_train, labels_train = cifar10_input.distorted_inputs(data_dir=self.data_dir,
                                                                            batch_size=self.batch_size)
            logits = self.inference(images_train, True)
            loss = self.loss(logits=logits, labels=labels_train)
            train_op = self.trainer(loss, global_step)

            class _LoggerHook(tf.train.SessionRunHook):
                """Logs loss and runtime."""

                def begin(self):
                    self._step = -1
                    self._start_time = time.time()

                def before_run(self, run_context):
                    self._step += 1
                    return tf.train.SessionRunArgs(loss)  # Asks for loss value.

                def after_run(self, run_context, run_values):
                    if self._step % log_frequency == 0:
                        current_time = time.time()
                        duration = current_time - self._start_time
                        self._start_time = current_time

                        loss_value = run_values.results
                        examples_per_sec = log_frequency * batch_size / duration
                        sec_per_batch = float(duration / log_frequency)

                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now(), self._step, loss_value,
                                            examples_per_sec, sec_per_batch))

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=self.checkpointdir,
                    hooks=[tf.train.StopAtStepHook(last_step=self.max_step),
                           tf.train.NanTensorHook(loss),
                           _LoggerHook()],
                    config=tf.ConfigProto(
                        log_device_placement=False)) as mon_sess:
                while not mon_sess.should_stop():
                    mon_sess.run(train_op)


"""Class Resnet18 is over"""


def main(argv=None):
    datadir="./tmp/cifar10_data/cifar-10-batches-bin"
    checkpointdir= "./checkpoints/"
    boundaries = [125000, 187500, 250000, 281250]
    learing_rates = [3e-2, 3e-3, 3e-4, 3e-4, 5e-5]
    batch_size = 32
    epochs = 200
    number_train_samples = 50000
    number_test_samples = 10000

    model = Resnet18(datadir, boundaries, learing_rates, batch_size, epochs,
                     number_train_samples, number_test_samples, checkpointdir)

    #cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(checkpointdir):
        tf.gfile.DeleteRecursively(checkpointdir)
    tf.gfile.MakeDirs(checkpointdir)
    model.train()



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    tf.app.run()
