import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time
import math

max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'


def variable_with_weight_loss(shape, stddev, w1):
    '''定义初始化weight函数,使用tf.truncated_normal截断的正态分布，但加上L2的loss，相当于做了一个L2的正则化处理'''
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    '''w1:控制L2 loss的大小，tf.nn.l2_loss函数计算weight的L2 loss'''
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        '''tf.add_to_collection:把weight losses统一存到一个collection，名为losses'''
        tf.add_to_collection('losses', weight_loss)

    return var


# 使用cifar10类下载数据集并解压展开到默认位置
cifar10.maybe_download_and_extract()

'''distored_inputs函数产生训练需要使用的数据，包括特征和其对应的label,
返回已经封装好的tensor，每次执行都会生成一个batch_size的数量的样本'''
images_train, labels_train = cifar10_input.distored_inputs(data_dir=data_dir,
                                                           batch_size=batch_size)

images_test, labels_test = cifar10_input.inputs(eval_data=True,
                                                data_dir=data_dir,
                                                batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

'''第一个卷积层：使用variable_with_weight_loss函数创建卷积核的参数并进行初始化。
第一个卷积层卷积核大小：5x5 3：颜色通道 64：卷积核数目
weight1初始化函数的标准差为0.05，不进行正则wl(weight loss)设为0'''
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
# tf.nn.conv2d函数对输入image_holder进行卷积操作
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')

bias1 = tf.Variable(tf.constant(0.0, shape=[64]))

conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
# 最大池化层尺寸为3x3,步长为2x2
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])
# LRN层模仿生物神经系统的'侧抑制'机制
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

'''第二个卷积层：'''
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
# bias2初始化为0.1
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))

conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全连接层
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 全连接层，隐含层节点数下降了一半
weight4 = variable_with_weight_loss(shape=[384, 182], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

'''正态分布标准差设为上一个隐含层节点数的倒数，且不计入L2的正则'''
weight5 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)


def loss(logits, labels):
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


# 将logits节点和label_holder传入loss计算得到最终loss
loss = loss(logits, label_holder)

train_op = tf.trian.AdamOptimizer(1e-3).minimize(loss)
# 求输出结果中top k的准确率，默认使用top 1(输出分类最高的那一类的准确率)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.trian.start_queue_runners()

for step in range(max_steps):
    '''training:'''
    start_time = time.time()
    # 获得一个batch的训练数据
    image_batch, label_batch = sess.run([images_train, labels_train])
    # 将batch的数据传入train_op和loss的计算
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={image_holder: image_batch, label_holder: label_batch})

    duration = time.time() - start_time
    if step % 10 == 0:
        # 每秒能训练的数量
        examples_per_sec = batch_size / duration
        # 一个batch数据所花费的时间
        sec_per_batch = float(duration)

        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
# 样本数
num_examples = 10000
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    # 获取images-test labels_test的batch
    image_batch, label_batch = sess.run([images_test, labels_test])
    # 计算这个batch的top 1上预测正确的样本数
    preditcions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                  label_holder: label_batch
                                                  })
    # 全部测试样本中预测正确的数量
    true_count += np.sum(preditcions)
    step += 1
# 准确率
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)




	train, train_label = read_data.get_files(train_dir)

    train_batch, train_label_batch = read_data.get_batch(train,
                                                             train_label,
                                                             IMG_W,
                                                             IMG_H,
                                                             BATCH_SIZE,
                                                             CAPACITY)
    with tf.Session(config=config) as sess:
    # with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:
        # init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        sess.run(init)
     	coord = tf.train.Coordinator()
		thread = tf.train.start_queue_runners(sess, coord)
  
        writer = tf.summary.FileWriter(save_dir, sess.graph)
        sample_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

        print("start_sess")

        for step in range(MaxStep):

            batch_images, batch_labels = sess.run([train_batch, train_label_batch])
     
            batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
          

            # 更新 D 的参数
            _, summary_str = sess.run([d_optim, d_sum],
                                      feed_dict={images: batch_images,
                                                 z: batch_z,
                                                 y: batch_labels})
