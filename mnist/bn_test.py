import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    mnist = input_data.read_data_sets('mnist', one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    image = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(image, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             name='conv1')
    bn1 = tf.layers.batch_normalization(conv1, training=False, name='bn1')
    pool1 = tf.layers.max_pooling2d(bn1, pool_size=[2, 2], strides=[2, 2], padding='same', name='pool1')
    conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             name='conv2')
    bn2 = tf.layers.batch_normalization(conv2, training=False, name='bn2')
    pool2 = tf.layers.max_pooling2d(bn2, pool_size=[2, 2], strides=[2, 2], padding='same', name='pool2')

    flatten_layer = tf.contrib.layers.flatten(pool2, 'flatten_layer')
    weights = tf.get_variable(shape=[flatten_layer.shape[-1], 10], dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.1), name='fc_weights')
    biases = tf.get_variable(shape=[10], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0), name='fc_biases')
    logit_output = tf.nn.bias_add(tf.matmul(flatten_layer, weights), biases, name='logit_output')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logit_output))
    pred_label = tf.argmax(logit_output, 1)
    label = tf.argmax(y_, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_label, label), tf.float32))

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    sess = tf.InteractiveSession(config=tf_config)
    saver = tf.train.Saver()
    if tf.train.latest_checkpoint('ckpts') is not None:
        saver.restore(sess, tf.train.latest_checkpoint('ckpts'))
    else:
        assert 'can not find checkpoint folder path!'

    loss, acc = sess.run([cross_entropy,accuracy],feed_dict={x: mnist.test.images,y_: mnist.test.labels})
    log_str = 'loss:%.6f \t acc:%.6f' % (loss, acc)
    tf.logging.info(log_str)
    sess.close()
