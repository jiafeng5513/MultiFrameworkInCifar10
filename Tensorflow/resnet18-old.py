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
    def __init__(self, gpu=0, cls=2, checkpoint_dir='./model', model_name='test', lr=1e-5):
        """
        初始化
        :param gpu:             使用几号GPU
        :param cls:             输出多少个分类结果
        :param checkpoint_dir:  保存ckpt的路径
        :param model_name:      模型名字
        :param lr:              学习率
        """
        self.cls = cls
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.lr = lr
        self.resnet50_path = 'weights\\imgnet_resnet18.npz'
        self.boundaries = [125000, 187500, 250000, 281250]
        self.learing_rates = [1e-3, 1e-4, 1e-5, 1e-6,5e-7]

        # if gpu:
        #     config = tf.ConfigProto()
        #     config.gpu_options.allow_growth = True
        #     self.sess = tf.Session(config=config)
        # else:
        #     self.sess = tf.Session()
        # self.build()

    def branch1(self, x, numOut, s, istraining):
        '''
        串联的两个卷积
        :param x:       输入
        :param numOut:  输出数量
        :param s:       stride,strides in conv2d is [1,s,s,1]
        :return:        输出
        '''
        with tf.variable_scope("conv1"):
            conv1 = utils.relu(utils.Bn(utils.conv2d(x, numOut, 3, 3, d_h=s, d_w=s), istraining))
        with tf.variable_scope("conv2"):
            conv2 = utils.Bn(utils.conv2d(conv1, numOut, 3, 3), istraining)
        return conv2

    def branch2(self, x, numOut, s, istraining):
        """
        一个卷积
        :param x:       输入
        :param numOut:  输出数量
        :param s:       stride,strides in conv2d is [1,s,s,1]
        :return:        输出
        """
        with tf.variable_scope("convshortcut"):
            return utils.Bn(utils.conv2d(x, numOut, d_h=s, d_w=s), istraining)

    def residual(self, x, numOut, istraining, stride=1, branch=False, name='res'):
        """
        一个残差块
        :param x:       输入
        :param numOut:  输出数量
        :param stride:  stride,strides in conv2d is [1,s,s,1]
        :param branch:  用于决定是否在直连侧添加一个卷积层
        :param name:    名字
        :return:
        """
        with tf.variable_scope(name):
            block = self.branch1(x, numOut, stride,istraining)
            if branch:
                skip = self.branch2(x, numOut, stride,istraining)
                return utils.relu(tf.add_n([block, skip]))
            else:
                return utils.relu(tf.add_n([x, block]))

    def inference(self, x, istraining):
        """
        构造整个网络
        :param x: 输入
        :return:  输出
        """
        with tf.variable_scope("conv"):
            conv1 = utils.relu(utils.Bn(utils.conv2d(x, 16, 3, 3, 1, 1, bias=False), training=istraining))

        with tf.variable_scope("Res"):
            res1 = self.residual(conv1, 16, istraining, branch=False, name='res1')
            res2 = self.residual(res1,  16, istraining, branch=False, name='res2')
            res3 = self.residual(res2,  16, istraining, branch=False, name='res3')
            res4 = self.residual(res3,  32, istraining, branch=True, name='res4')
            res5 = self.residual(res4,  32, istraining, branch=False, name='res5')
            res6 = self.residual(res5,  32, istraining, branch=False, name='res6')
            res7 = self.residual(res6,  64, istraining, branch=True, name='res7')
            res8 = self.residual(res7,  64, istraining, branch=False, name='res8')
            res9 = self.residual(res8,  64, istraining, branch=False, name='res9')

        with tf.variable_scope("out"):
            act = utils.relu(res9)
            avp = utils.avg_pool(act,8,8,8,8,"AveragePooling")
            fln = tf.layers.flatten(avp,name="Flatten")
            output=tf.layers.dense(fln,activation='softmax', units=10,use_bias=True, kernel_initializer='he_normal',name="output")

        return output

    def load_weight(self):
        param = dict(np.load(self.resnet50_path))
        vars = tf.global_variables(scope="inference")
        for v in vars:
            nameEnd = v.name.split('/')[-1]
            if nameEnd == "moving_mean:0":
                name =  v.name[10:-13]+"mean/EMA"
            elif nameEnd == "moving_variance:0":
                name = v.name[10:-17]+"variance/EMA"
            else:
                name = v.name[10:-2]
            if name == 'linear/W':
                param[name] = param['linear/W'].reshape(512, 1000)
            self.sess.run(v.assign(param[name]))
            print("Copy weights: " + name + "---->"+ v.name)

    def build(self):
        self.inputs = tf.placeholder(tf.float32, [None, 32, 32, 3], "inputs")
        self.labels = tf.placeholder(tf.int32, [None], "labels")
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        with tf.variable_scope("inference"):
            self.outs = self.inference(self.inputs,self.is_training)
        # 定义损失函数

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outs, labels=self.labels))
        with tf.variable_scope('minimizer'):
            adam = tf.train.AdamOptimizer(self.lr)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optm = adam.minimize(self.loss)
        # 模型保存
        self.saver = tf.train.Saver(max_to_keep=2)
        # 初始化
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # 加载预训练权重
        # self.load_weight()

    def get_Learning_rate(self,step):
        if step<self.boundaries[0]:
            return self.learing_rates[0]
        elif step>=self.boundaries[0] and step<self.boundaries[1]:
            return self.learing_rates[1]
        elif step>=self.boundaries[1] and step <self.boundaries[2]:
            return self.learing_rates[2]
        elif step>=self.boundaries[2] and step <self.boundaries[3]:
            return self.learing_rates[3]
        elif step>=self.boundaries[3] :
            return self.learing_rates[4]

    def loss(self, logits, labels):
        '''计算CNN的loss
        tf.nn.sparse_softmax_cross_entropy_with_logits作用：
        把softmax计算和cross_entropy_loss计算合在一起'''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
        # tf.reduce_mean对cross entropy计算均值
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='cross_entropy')
        # tf.add_to_collection:把cross entropy的loss添加到整体losses的collection中
        tf.add_to_collection('losses', cross_entropy_mean)
        # tf.add_n将整体losses的collection中的全部loss求和得到最终的loss
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def train(self):
        """
        训练和验证
        :return:
        """
        # 定义网络模型的输入输出和损失
        with tf.Graph().as_default():
            batch_size = 32
            image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3],name="image_in")
            label_holder = tf.placeholder(tf.int32, [batch_size],name="label_in")
            is_training = tf.placeholder(tf.bool, name="is_training")

            data_dir="./tmp/cifar10_data/cifar-10-batches-bin"
            with tf.device('/cpu:0'):
                images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
                images_test, labels_test = cifar10_input.inputs(data_dir=data_dir,eval_data=True, batch_size=batch_size)

            logits = self.inference(image_holder, is_training)
            loss = self.loss(logits=logits, labels=label_holder)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                self.global_step = 0
                self.learing_rate = self.get_Learning_rate(self.global_step)
                train_op = tf.train.AdamOptimizer(self.learing_rate).minimize(loss)  # Or AdamOptimizer
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
                    _, loss_value = sess.run([train_op, loss],
                                             feed_dict={image_holder: image_batch, label_holder: label_batch,is_training: True})
                    duration = time.time() - start_time
                    if step % 10 == 0:
                        # 每秒能训练的数量
                        examples_per_sec = batch_size / duration
                        # 一个batch数据所花费的时间
                        sec_per_batch = float(duration)
                        format_str = ('step %d, loss=%.2f ,learing_rate:%.8f,(%.1f examples/sec; %.3f sec/batch)')
                        print(format_str % ( self.global_step, loss_value,self.learing_rate, examples_per_sec, sec_per_batch))
    
    
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
    model = Resnet18(gpu=0,cls=10,model_name='test', lr=1e-5)
    model.train()


if __name__ == "__main__":
    main()
