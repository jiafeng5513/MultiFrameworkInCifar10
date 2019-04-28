# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import time
import os
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.layers import normalization
from onnx_tf.frontend import tensorflow_graph_to_onnx_model
from sklearn import preprocessing
import keras.layers
# define for loggerHook,DO NOT USE IN OTHER PLACE!
log_frequency = 10
batch_size = 32


class Resnet20:
    # 初始化
    def __init__(self, boundaries, learing_rates, batch_size=32, epochs=200,
                 number_train_samples=50000, number_test_samples=10000):
        self.boundaries = boundaries
        self.learing_rates = learing_rates
        self.batch_size = batch_size
        self.epochs = epochs
        self.number_train_samples = number_train_samples
        self.number_test_samples = number_test_samples


    # 定义卷积
    def Convolution(self, input_, output_dim, kernel_height=1, kernel_width=1,
                    stride_height=1, stride_width=1, with_blas=False, padding="SAME"):
        # input_channels =(int)(input_.get_shape()[-1])
        # initial = tf.truncated_normal([kernel_height, kernel_width, input_channels, output_dim], stddev=0.1)
        # weight = tf.Variable(initial)
        # conv = tf.nn.conv2d(input_, weight, strides=[1, stride_height, stride_width, 1], padding=padding)

        conv = tf.keras.layers.Conv2D(filters=output_dim,kernel_size=(kernel_height,kernel_width),
                               strides=(stride_height,stride_width),use_bias=with_blas,padding="same")(input_)
        # blas
        # if with_blas:
        #     bias = tf.get_variable("b",[output_dim], initializer=tf.constant_initializer(0.0))
        #     return tf.nn.bias_add(conv, bias)
        # else:
        return conv

    # 定义Residual Block V1
    def ResidualBlockV1(self, input, output_dim,  name="res"):
        with tf.variable_scope(name):
            c1 = self.Convolution(input, output_dim, 3, 3, 1, 1)
            #bn1 = self.BatchNormalization(c1, is_training)
            bn1=tf.keras.layers.BatchNormalization()(c1)
            r1 = tf.keras.layers.ReLU()(bn1)
        #with tf.variable_scope(name+"_B"):
            c2 = self.Convolution(r1, output_dim, 3, 3, 1, 1)
            #bn2 = self.BatchNormalization(c2, is_training)
            bn2 = tf.keras.layers.BatchNormalization()(c2)
            plus = tf.keras.layers.add([input, bn2])
        return tf.keras.layers.ReLU()(plus)

    # 定义Residual Block V2
    def ResidualBlockV2(self, input, output_dim1, name="res"):
        with tf.variable_scope(name):
            c1 = self.Convolution(input, output_dim1, 3, 3, 2, 2)
            #bn1 = self.BatchNormalization(c1, is_training)
            bn1 = tf.keras.layers.BatchNormalization()(c1)
            r1 = tf.keras.layers.ReLU()(bn1)
        #with tf.variable_scope(name+"_B"):
            c2 = self.Convolution(r1, output_dim1, 3, 3, 1, 1)
            #bn2 = self.BatchNormalization(c2, is_training)
            bn2 = tf.keras.layers.BatchNormalization()(c2)
        #with tf.variable_scope(name+"_C"):
            c3 = self.Convolution(input, output_dim1, 1, 1, 2, 2)
            #bn3 = self.BatchNormalization(c3, is_training)
            bn3 = tf.keras.layers.BatchNormalization()(c3)
            plus = tf.keras.layers.add([bn3, bn2])

        return tf.keras.layers.ReLU()(plus)

    # 定义网络
    def inference(self, x):
        """
        构造整个网络
        :param x: 输入
        :return:  输出
        """
        with tf.name_scope("Conv1"):
            c1 = self.Convolution(x, 16, 3, 3, 1, 1, with_blas=False)
            bn1 = tf.keras.layers.BatchNormalization()(c1)
            conv1 = tf.keras.layers.ReLU()(bn1)
        res1 = self.ResidualBlockV1(conv1, 16,  name="res1")
        res2 = self.ResidualBlockV1(res1, 16,  name="res2")
        res3 = self.ResidualBlockV1(res2, 16,  name="res3")

        res4 = self.ResidualBlockV2(res3, 32,  name="res4")
        res5 = self.ResidualBlockV1(res4, 32,  name="res5")
        res6 = self.ResidualBlockV1(res5, 32,  name="res6")

        res7 = self.ResidualBlockV2(res6, 64,  name="res7")
        res8 = self.ResidualBlockV1(res7, 64,  name="res8")
        res9 = self.ResidualBlockV1(res8, 64,  name="res10")


        #avp = tf.nn.avg_pool(res9, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="VALID", name="AveragePooling")

        avp = tf.keras.layers.AvgPool2D((8,8),(8,8))(res9)
        #fln = keras.layers.Flatten()(avp)
        fln = tf.keras.layers.Flatten()(avp)
        #keras.layers.Dense(units=10,activation='softmax',use_bias=True)()
        output = tf.keras.layers.Dense(units=10,activation='softmax',kernel_initializer='he_normal')(fln)
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
        labels = tf.cast(tf.one_hot(labels,10,1,0), tf.float32)
        loss=-tf.reduce_sum(labels*tf.log(logits))

        # cce = tf.keras.losses.CategoricalCrossentropy()
        # la=tf.one_hot(labels,10,1,0)
        # loss = cce(la,logits)
        #print('Loss: ', loss.numpy())  # Loss: 0.3239


        '''计算CNN的loss
        tf.nn.sparse_softmax_cross_entropy_with_logits作用：
        把softmax计算和cross_entropy_loss计算合在一起'''
        #labels = tf.cast(labels, tf.int64)
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=logits, labels=labels, name='cross_entropy_per_example')
        # # tf.reduce_mean对cross entropy计算均值
        # cross_entropy_mean = tf.reduce_mean(cross_entropy,
        #                                     name='cross_entropy')
        # # tf.add_to_collection:把cross entropy的loss添加到整体losses的collection中
        # cross_entropy_mean_res=tf.add_to_collection('losses', cross_entropy_mean)
        # tf.add_n将整体losses的collection中的全部loss求和得到最终的loss
        #return tf.add_n(tf.get_collection('losses'), name='total_loss')
        return tf.reduce_mean(loss)


    def evaluation(self,logits, labels):
        with tf.variable_scope("accuracy") as scope:
            correct = tf.nn.in_top_k(logits, labels, 1)
            correct = tf.cast(correct, tf.float16)
            accuracy = tf.reduce_mean(correct)
            #tf.summary.scalar(scope.name + "accuracy", accuracy)
        return accuracy

    # 标准化
    def normalixe(self,data):
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # data is [n,32,32,3]
        x_train = data.astype('float32') / 255
        mean=np.mean(x_train, axis=0)
        return x_train

    # 训练
    def train(self):
        # 模型准备
        image_holder = tf.placeholder(tf.float32, [None, 32, 32, 3], name="input")#
        label_holder = tf.placeholder(tf.int64, [None])

        logits = self.inference(image_holder)
        loss = self.loss(logits, label_holder)
        train_acc = self.evaluation(logits, label_holder)

        global_step = tf.Variable(1, trainable=False,dtype=tf.int64)
        learning_rate = tf.train.piecewise_constant(x=global_step, boundaries=self.boundaries, values=self.learing_rates)

        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          epsilon=0.01).minimize(loss=loss,global_step=global_step)

        # 数据准备
        cifar_train = tfds.load(name="cifar10", split=tfds.Split.TRAIN)
        cifar_train = cifar_train.repeat(self.epochs).shuffle(self.number_train_samples).batch(self.batch_size)
        cifar_train = cifar_train.prefetch(tf.data.experimental.AUTOTUNE)



        #训练准备
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        num_batches_per_epoch = int(self.number_train_samples / self.batch_size)
        #训练循环
        start_time=time.clock()
        for batch in tfds.as_numpy(cifar_train):
            images, labels = batch["image"], batch["label"]
            images_n = self.normalixe(images) #images.astype('float32') / 255self.batch_size
            lr = sess.run(learning_rate)
            step = sess.run(global_step)
            #log=sess.run(logits,feed_dict={image_holder: images})
            _, loss_value,acc = sess.run([train_op, loss, train_acc],
                                     feed_dict={image_holder: images_n,
                                                label_holder: labels})
            if step % 20 == 0:
                epoch=step/num_batches_per_epoch+1
                format_str = ('step %d, epoch=%d, lr=%f, loss=%.4f, acc=%.4f')
                print(format_str % (step, epoch, lr, loss_value, acc))
        # end of for

        time_cost=time.clock()-start_time
        var=sess.graph_def
        # 保存pb文件
        if os.path.exists("ResNet20-cifar10-tf.pb"):
            os.remove("ResNet20-cifar10-tf.pb")
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["dense/Softmax"])
        with tf.gfile.FastGFile("ResNet20-cifar10-tf.pb", mode='wb') as f:
            f.write(constant_graph.SerializeToString())

        # 测试
        top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
        cifar_test = tfds.load(name="cifar10", split=tfds.Split.TEST)
        cifar_test = cifar_test.repeat(1).shuffle(self.number_test_samples).batch(self.batch_size)
        cifar_test = cifar_test.prefetch(tf.data.experimental.AUTOTUNE)
        true_count = 0
        total_number = 0
        for batch in tfds.as_numpy(cifar_test):
            images, labels = batch["image"], batch["label"]
            images_n = self.normalixe(images)
            # 计算这个batch的top 1上预测正确的样本数
            if np.shape(images)[0] >= 0:  # self.batch_size
                preditcions, logits_out = sess.run([top_k_op, logits], feed_dict={image_holder: images_n,
                                                                                  label_holder: labels})
                true_count += np.sum(preditcions)
                total_number += np.shape(images)[0]
                print('total %d samples,Num of correct is %d' % (total_number, true_count))

        # 准确率
        precision = true_count / total_number
        print('acc = %.4f' % precision)
        return precision,time_cost


    # 测试
    def test(self):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open("ResNet20-cifar10-tf.pb", "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")
                with tf.Session() as sess:
                    init = tf.global_variables_initializer()

                    image_holder = sess.graph.get_tensor_by_name("input:0")
                    logits = sess.graph.get_tensor_by_name("dense/Softmax:0")
                    #is_training_holder = sess.graph.get_tensor_by_name("is_training:0")
                    label_holder = tf.placeholder(tf.int64, [None])
                    top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

                    cifar_test = tfds.load(name="cifar10", split=tfds.Split.TEST)
                    cifar_test = cifar_test.repeat(1).shuffle(self.number_test_samples).batch(self.batch_size)
                    cifar_test = cifar_test.prefetch(tf.data.experimental.AUTOTUNE)
                    sess.run(init)
                    true_count = 0
                    total_number = 0
                    for batch in tfds.as_numpy(cifar_test):
                        images, labels = batch["image"], batch["label"]
                        images_n = self.normalixe(images)
                        # 计算这个batch的top 1上预测正确的样本数
                        if np.shape(images)[0] >=0:# self.batch_size
                            preditcions,logits_out = sess.run([top_k_op,logits], feed_dict={image_holder: images_n,
                                                                          label_holder: labels})
                            true_count += np.sum(preditcions)
                            total_number += np.shape(images)[0]
                            print('total %d samples,Num of correct is %d'%(total_number, true_count))

                    # 准确率
                    precision = true_count / total_number
                    print('acc = %.4f' % precision)
                    return precision


def main(argv=None):
    batch_size = 32
    epochs = 1
    number_train_samples = 50000
    number_test_samples = 10000
    boundaries_epoch=[80, 120, 160, 180]  # 学习率变化阈值,epoch到达阈值后学习率发生变化
    boundaries = [item * int(number_train_samples/batch_size) for item in boundaries_epoch]
    learing_rates = [3e-2, 3e-3, 3e-4, 3e-4, 5e-5]  # 学习率取值

    model = Resnet20( boundaries, learing_rates, batch_size, epochs,
                     number_train_samples, number_test_samples)

    precision,time_cost=model.train()
    #precision=model.test()
    # 输出训练日志
    i = 1
    while os.path.exists("log_" + str(i) + ".txt"):
        i = i + 1
    output = open("log_" + str(i) + ".txt", 'w')
    output.write("Test accuracy=%f,Time used %f s" % (precision, time_cost))

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    tf.app.run(main=main)


    #sess.close()