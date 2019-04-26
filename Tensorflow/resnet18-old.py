import tensorflow as tf
import utils
import numpy as np
import scipy.io as sio
import cv2
import cifar10_input
import time
import math
from tensorflow.python.framework import graph_util

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.

class Resnet18:
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
        self.max_step = int(self.number_train_samples / self.batch_size * self.epochs)
        self.MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.

    # 定义卷积
    def Convolution(self, input_, output_dim, kernel_height=1, kernel_width=1,
                    stride_height=1, stride_width=1, with_blas=False, name="w", padding="SAME"):
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
        conv = tf.nn.conv2d(input_, w, strides=[1, stride_height, stride_width, 1], padding=padding)
        # blas
        if with_blas:
            bias = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
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
    def ResidualBlockV1(self, input, output_dim, is_training=True, kernel_height=3, kernel_width=3, strides=1,
                        name="res"):
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
        with tf.variable_scope(name + "_A"):
            c1 = self.Convolution(input, output_dim, kernel_height, kernel_width, strides, strides, name=name + "_A")
            bn1 = self.BatchNormalization(c1, is_training)
            r1 = tf.nn.relu(bn1)
        with tf.variable_scope(name + "_B"):
            c2 = self.Convolution(r1, output_dim, kernel_height, kernel_width, strides, strides, name=name + "_B")
            bn2 = self.BatchNormalization(c2, is_training)
            plus = tf.add_n([input, bn2])
        return tf.nn.relu(plus)

    # 定义Residual Block V2
    def ResidualBlockV2(self, input, output_dim1, is_training=True, kernel_height=3, kernel_width=3, strides=1,
                        name="res"):
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
        with tf.variable_scope(name + "_A"):
            c1 = self.Convolution(input, output_dim1, 3, 3, 2, 2, name=name + "_A")
            bn1 = self.BatchNormalization(c1, is_training)
            r1 = tf.nn.relu(bn1)
        with tf.variable_scope(name + "_B"):
            c2 = self.Convolution(r1, output_dim1, 3, 3, 1, 1, name=name + "_B")
            bn2 = self.BatchNormalization(c2, is_training)
        with tf.variable_scope(name + "_C"):
            c3 = self.Convolution(input, output_dim1, 1, 1, 2, 2, name=name + "_C", padding="VALID")
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

    # 定义损失函数
    def loss(self, logits, labels):
        '''
        计算CNN的loss
        '''
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            labels = tf.cast(labels, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross_entropy_per_example')
            # tf.reduce_mean对cross entropy计算均值
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            # tf.add_to_collection:把cross entropy的loss添加到整体losses的collection中
            tf.add_to_collection('losses', cross_entropy_mean)
            # tf.add_n将整体losses的collection中的全部loss求和得到最终的loss
            return tf.add_n(tf.get_collection('losses'), name='total_loss')
            # return cross_entropy_mean

    # 定义准确率
    # 定义准确率
    def acc(self,logits,labels):
        """
        计算一个batch的准确率
        :param logits: 一个batch的输出,[batchsize,10]
        :param labels: [batchsize,10]
        :return: 准确率
        """
        true_count = 0
        for i in range(0,self.batch_size):
            Logit_item = logits[i]
            label = labels[i]
            if(np.argmax(Logit_item)==np.argmax(label)):
                true_count=true_count+1
        return str(true_count/self.batch_size)

    def train(self):
        """
        训练和验证
        :return:
        """
        # 定义网络模型的输入输出和损失
        with tf.Graph().as_default():
            batch_size = 32
            image_holder = tf.placeholder(tf.float32, [batch_size, 32, 32, 3], name="image_in")
            label_holder = tf.placeholder(tf.int32, [batch_size], name="label_in")
            is_training = tf.placeholder(tf.bool, name="is_training")

            data_dir="./tmp/cifar10_data/cifar-10-batches-bin"
            with tf.device('/cpu:0'):
                images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
                images_test, labels_test = cifar10_input.inputs(data_dir=data_dir,eval_data=True, batch_size=batch_size)

            logits = self.inference(image_holder, is_training)
            loss = self.loss(logits=logits, labels=label_holder)
            accuracy = self.acc(logits, label_holder)


            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                self.global_step = 0
                self.learing_rate = self.get_Learning_rate(self.global_step)
                train_op = tf.train.AdagradOptimizer(self.learing_rate).minimize(loss)  # Or AdamOptimizer
                top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

                init = tf.global_variables_initializer()
                sess.run(init)
                coord = tf.train.Coordinator()
                thread = tf.train.start_queue_runners(sess, coord)
                start = time.clock()
                
                print('******Training start********')
                for step in range(312500):
                    start_time = time.time()
                    image_batch, label_batch = sess.run([images_train, labels_train])
                    self.global_step=step
                    self.learing_rate = self.get_Learning_rate(self.global_step)
                    _, loss_value,acc_value = sess.run([train_op, loss,accuracy],
                                             feed_dict={image_holder: image_batch,
                                                        label_holder: label_batch,
                                                        is_training: True})
                    duration = time.time() - start_time
                    if step % 10 == 0:
                        # 每秒能训练的数量
                        examples_per_sec = batch_size / duration
                        # 一个batch数据所花费的时间
                        sec_per_batch = float(duration)
                        format_str = ('step %d, loss=%s, acc=%.4f, learing_rate:%.8f,(%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % ( self.global_step, loss_value, acc_value,
                                             self.learing_rate, examples_per_sec, sec_per_batch))
    
    
                num_examples = 10000
                num_iter = int(math.ceil(num_examples / batch_size))
                true_count = 0
                total_sample_count = num_iter * batch_size
                step = 0
                while step < num_iter:
                    # 获取images-test labels_test的batch
                    image_batch_test, label_batch_test = sess.run([images_test, labels_test])
                    # 计算这个batch的top 5上预测正确的样本数
                    preditcions = sess.run([top_k_op], feed_dict={image_holder: image_batch_test,
                                                                  label_holder: label_batch_test,
                                                                  is_training: False})
                    # 全部测试样本中预测正确的数量
                    true_count += np.sum(preditcions)
                    step += 1
                # 准确率
                precision = true_count / total_sample_count
                print('precision @ 1 = %.3f' % precision)
                elapsed = (time.clock() - start)
                print("Time used (s):", elapsed)
                tf.train.write_graph(sess.graph_def, "./model", 'expert-graph.pb', as_text=False)
        pass

    def test(self, img):
        pass



def main():
    datadir = "./tmp/cifar10_data/cifar-10-batches-bin"
    checkpointdir = "./checkpoints/"
    batch_size = 32
    epochs = 200
    number_train_samples = 50000
    number_test_samples = 10000
    # 一个step是一个batch
    # 80epoch =80* 50000/32
    boundaries_epoch = [80, 120, 160, 180]
    boundaries = [item * int(50000 / 32) for item in boundaries_epoch]
    learing_rates = [3e-2, 3e-3, 3e-4, 3e-4, 5e-5]

    model = Resnet18(datadir, boundaries, learing_rates, batch_size, epochs,
                     number_train_samples, number_test_samples, checkpointdir)

    # cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(checkpointdir):
        tf.gfile.DeleteRecursively(checkpointdir)
    tf.gfile.MakeDirs(checkpointdir)
    model.train()


if __name__ == "__main__":
    main()
