# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import os
import re
import sys
import urllib
import tarfile
from datetime import datetime
import time
import math
import cifar10_input

log_frequency = 10
batch_size = 32
NUM_CLASSES = 10

TOWER_NAME = 'tower'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
dest_directory = '../CIFAR-10'
checkpoint_dir = './checkpoints'
eval_dir = './eval'
log_device_placement = False

# 一个训练的epoch含有多少样本 50000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
# 一个测试的epoch含有多少样本 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
# 一个epoch有多少个batch
NUM_BATCHES_PRE_RPOCH_FOE_TRAIN = (int)(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/batch_size)
# 一共训练多少个Epoch
MAX_EPOCH = 3
# 训练次数控制,一个step是一个batch
max_steps = NUM_BATCHES_PRE_RPOCH_FOE_TRAIN*MAX_EPOCH
# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999
# 学习率变化阈值,epoch到达阈值后学习率发生变化
boundaries_epoch = [80, 120, 160, 180]
# 学习率取值
learing_rates = [3e-2, 3e-3, 3e-4, 3e-4, 5e-5]

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""

  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """

  data_dir = os.path.join(dest_directory, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=batch_size)

  return images, labels

def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """

  data_dir = os.path.join(dest_directory, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=batch_size)

  return images, labels

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss):
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

def resV1(input,k,name,is_training=True):
    with tf.variable_scope(name) as scope:
        kernel_1 = _variable_with_weight_decay('weights_1',
                                               shape=[3, 3, (int)(input.get_shape()[-1]), k],
                                               stddev=5e-2,
                                               wd=0.0)
        conv_1 = tf.nn.conv2d(input, kernel_1, [1, 1, 1, 1], padding='SAME')
        bn1=tf.layers.batch_normalization(conv_1, training=is_training, name=scope.name+'-bn1')
        re = tf.nn.relu(bn1, name=scope.name+'-1')
        kernel_2 = _variable_with_weight_decay('weights_2',
                                               shape=[3, 3, (int)(input.get_shape()[-1]), k],
                                               stddev=5e-2,
                                               wd=0.0)
        conv_2 = tf.nn.conv2d(re, kernel_2, [1, 1, 1, 1], padding='SAME')
        bn2=tf.layers.batch_normalization(conv_2, training=is_training, name=scope.name+'-bn2')
        res1 = tf.nn.relu(tf.add_n([input, bn2]), name=scope.name+'-2')
        _activation_summary(res1)
        return res1

def resV2(input,k,name,is_training=True):
    with tf.variable_scope(name) as scope:
        kernel_1 = _variable_with_weight_decay('weights_1',
                                               shape=[3, 3, (int)(input.get_shape()[-1]), k],
                                               stddev=5e-2,
                                               wd=0.0)
        conv_1 = tf.nn.conv2d(input, kernel_1, [1, 2, 2, 1], padding='SAME')
        bn1 = tf.layers.batch_normalization(conv_1, training=is_training, name=scope.name + '-bn1')
        re = tf.nn.relu(bn1, name=scope.name+'-1')
        kernel_2 = _variable_with_weight_decay('weights_2',
                                               shape=[3, 3, (int)(re.get_shape()[-1]), k],
                                               stddev=5e-2,
                                               wd=0.0)
        conv_2 = tf.nn.conv2d(re, kernel_2, [1, 1, 1, 1], padding='SAME')
        bn2 = tf.layers.batch_normalization(conv_2, training=is_training, name=scope.name + '-bn2')

        kernel_3 = _variable_with_weight_decay('weights_3',
                                               shape=[1, 1, (int)(input.get_shape()[-1]), k],
                                               stddev=5e-2,
                                               wd=0.0)
        conv_3 = tf.nn.conv2d(input, kernel_3, [1, 2, 2, 1], padding='SAME')
        bn3 = tf.layers.batch_normalization(conv_3, training=is_training, name=scope.name + '-bn3')

        res1 = tf.nn.relu(tf.add_n([bn3, bn2]), name=scope.name+'-2')
        _activation_summary(res1)
        return res1

def inference_res20(images,is_training=True):
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 16],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    bn1 = tf.layers.batch_normalization(conv, training=is_training, name=scope.name + '-bn1')
    conv1 = tf.nn.relu(bn1, name=scope.name)
    _activation_summary(conv1)

  # res
  res1 = resV1(conv1, 16, "res1",is_training)
  res2 = resV1(res1, 16, "res2",is_training)
  res3 = resV1(res2, 16, "res3",is_training)

  res4 = resV2(res3, 32, "res4",is_training)
  res5 = resV1(res4, 32, "res5",is_training)
  res6 = resV1(res5, 32, "res6",is_training)

  res7 = resV2(res6, 64, "res7",is_training)
  res8 = resV1(res7, 64, "res8",is_training)
  res9 = resV1(res8, 64, "res9",is_training)


  with tf.variable_scope('softmax_linear') as scope:
    pool = tf.nn.avg_pool(res9, ksize=[1, 8, 8, 1],
                            strides=[1, 8, 8, 1], padding='SAME', name='pool')
    weights = _variable_with_weight_decay('weights', [64, NUM_CLASSES],
                                          stddev=1/64.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(tf.layers.Flatten()(pool), weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def loss_res20(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def getTrainOp(total_loss, global_step):
  boundaries = [item * NUM_BATCHES_PRE_RPOCH_FOE_TRAIN for item in boundaries_epoch]
  learning_rate = tf.train.piecewise_constant(x=global_step, boundaries=boundaries, values=learing_rates)
  tf.summary.scalar('learning_rate', learning_rate)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(learning_rate)
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
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')
    train_op = tf.group([train_op, update_ops])
  return train_op

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = distorted_inputs()

    # Build a Graph that computes the logits predictions from the inference model.
    logits = inference_res20(images,is_training=True)

    # Calculate loss.
    loss = loss_res20(logits, labels)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = getTrainOp(loss, global_step)

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
          epoch=(int)(self._step/NUM_BATCHES_PRE_RPOCH_FOE_TRAIN)+1

          format_str = ('%s: step %d, epoch = %d/%d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, epoch,MAX_EPOCH, loss_value,
                               examples_per_sec, sec_per_batch))

    start_time = time.clock()
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=checkpoint_dir,
        hooks=[tf.train.StopAtStepHook(last_step=max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
    return time.clock() - start_time

def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    return precision

def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.

    images, labels = inputs(eval_data=True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = inference_res20(images,is_training=False)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(eval_dir, g)
    return eval_once(saver, summary_writer, top_k_op, summary_op)

if __name__ == "__main__":
    maybe_download_and_extract()
    if tf.gfile.Exists(checkpoint_dir):
        tf.gfile.DeleteRecursively(checkpoint_dir)
    tf.gfile.MakeDirs(checkpoint_dir)
    time_used = train()

    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)
    acc = evaluate()

    print ("total time used: %fs, acc=%f, write in log file!" % (time_used,acc))
    # 输出训练日志
    i = 1
    while os.path.exists("log_" + str(i) + ".txt"):
        i = i + 1
    output = open("log_" + str(i) + ".txt", 'w')
    output.write("Test accuracy= %f ,Time used %f s" % (acc, time_used))
