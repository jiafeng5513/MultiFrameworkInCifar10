from __future__ import print_function
import os
import argparse
import cntk as C
import numpy as np

from cntk import cross_entropy_with_softmax, classification_error, reduce_mean
from cntk import Trainer, cntk_py
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk.learners import momentum_sgd, learning_parameter_schedule, momentum_schedule,adam
from cntk.debugging import *
from cntk.logging import *

import cntk.io.transforms as xforms
from cntk.initializer import he_normal, normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu
# Paths relative to current python file.
abs_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(abs_path, "..", "CIFAR-10")

# model dimensions
image_height = 32
image_width = 32
num_channels = 3  # RGB
num_classes = 10


# Define the reader for both training and evaluation action.
def create_image_mb_source(map_file, mean_file, train, total_number_of_samples):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError(
            "File '%s' or '%s' does not exist. Please run install_cifar10.py from DataSets/CIFAR-10 to fetch them" %
            (map_file, mean_file))

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    # if train:
        # transforms += [
        #     xforms.crop(crop_type='randomside', side_ratio=(0.8, 1.0), jitter_type='uniratio')  # train uses jitter
        # ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=transforms),  # first column in map file is referred to as 'image'
        labels=StreamDef(field='label', shape=num_classes))),  # and second as 'label'
                           randomize=train,
                           max_samples=total_number_of_samples,
                           multithreaded_deserializer=True)


#
# assembly components
#
def conv_bn(input, filter_size, num_filters, strides=(1, 1), init=he_normal(), bn_init_scale=1):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False, init_scale=bn_init_scale, disable_regularization=True)(c)
    return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1, 1), init=he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init, 1)
    return relu(r)

#
# ResNet components
#
def resnet_basic(input, num_filters):
    c1 = conv_bn_relu(input, (3, 3), num_filters)
    c2 = conv_bn(c1, (3, 3), num_filters, bn_init_scale=1)
    p = c2 + input
    return relu(p)

def resnet_basic_inc(input, num_filters, strides=(2, 2)):
    c1 = conv_bn_relu(input, (3, 3), num_filters, strides)
    c2 = conv_bn(c1, (3, 3), num_filters, bn_init_scale=1)
    s = conv_bn(input, (1, 1), num_filters, strides) # Shortcut
    p = c2 + s
    return relu(p)

def resnet_basic_stack(input, num_stack_layers, num_filters):
    assert(num_stack_layers >= 0)
    l = input
    for _ in range(num_stack_layers):
        l = resnet_basic(l, num_filters)
    return l

def create_cifar10_model(input, num_stack_layers, num_classes):
    c_map = [16, 32, 64]

    conv = conv_bn_relu(input, (3, 3), c_map[0])
    r1 = resnet_basic_stack(conv, num_stack_layers, c_map[0])

    r2_1 = resnet_basic_inc(r1, c_map[1])
    r2_2 = resnet_basic_stack(r2_1, num_stack_layers-1, c_map[1])

    r3_1 = resnet_basic_inc(r2_2, c_map[2])
    r3_2 = resnet_basic_stack(r3_1, num_stack_layers-1, c_map[2])

    # Global average pooling and output
    pool = AveragePooling(filter_shape=(8, 8), name='final_avg_pooling')(r3_2)
    z = Dense(num_classes, init=normal(0.01))(pool)
    return z

# Train and evaluate the network.
def train_and_evaluate(reader_train, reader_test, network_name, epoch_size, max_epochs, minibatch_size,
                       model_dir=None, log_dir=None, tensorboard_logdir=None, gen_heartbeat=False, fp16=False):
    """

    :param reader_train:
    :param reader_test:
    :param network_name:
    :param epoch_size:    一个epoch有多少样本
    :param max_epochs:    训练多少个epoch
    :param model_dir:
    :param log_dir:
    :param tensorboard_logdir:
    :param gen_heartbeat:
    :param fp16:
    :return:准确率,用时
    """
    set_computation_network_trace_level(0)

    # Input variables denoting the features and label data
    input_var = C.input_variable((num_channels, image_height, image_width), name='features')
    label_var = C.input_variable((num_classes))

    with C.default_options(dtype=np.float32):
        # create model, and configure learning parameters
        model = create_cifar10_model(input_var, 3, num_classes)
        # loss and metric
        loss = cross_entropy_with_softmax(model, label_var)
        error_rate = classification_error(model, label_var)


    # shared training parameters

    # Set learning parameters
    lr_per_sample = []
    check_point=[80,120,160,180]
    lrs=[3e-2,3e-3,3e-4,3e-4,5e-5]
    for i in range(max_epochs+1):
        if i in range(0,check_point[0]):
            lr_per_sample.append(lrs[0])
        if i in range(check_point[0], check_point[1]):
            lr_per_sample.append(lrs[1])
        if i in range(check_point[1], check_point[2]):
            lr_per_sample.append(lrs[2])
        if i in range(check_point[2], check_point[3]):
            lr_per_sample.append(lrs[3])
        if i>check_point[3]:
            lr_per_sample.append(lrs[4])

    lr_schedule = learning_parameter_schedule(lr_per_sample, minibatch_size=minibatch_size, epoch_size=epoch_size)
    mm_schedule = momentum_schedule(0.9, minibatch_size)  #动量

    # progress writers
    progress_writers = [
        ProgressPrinter(tag='Training', num_epochs=max_epochs, gen_heartbeat=gen_heartbeat)]
    tensorboard_writer = None
    if tensorboard_logdir is not None:
        tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=tensorboard_logdir, model=model)
        progress_writers.append(tensorboard_writer)

    # trainer object
    l2_reg_weight = 0.0001
    learner = adam(model.parameters, lr_schedule, mm_schedule,l2_regularization_weight=l2_reg_weight)
    trainer = Trainer(model, (loss, error_rate), learner, progress_writers)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    log_number_of_parameters(model);
    print("*********Training Start*********")
    start = time.clock()
    for epoch in range(max_epochs):  # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count),
                                               input_map=input_map)  # fetch minibatch.
            trainer.train_minibatch(data)  # update model with it
            sample_count += trainer.previous_minibatch_sample_count  # count samples processed so far

        trainer.summarize_training_progress()

        # Log mean of each parameter tensor, so that we can confirm that the parameters change indeed.
        if tensorboard_writer:
            for parameter in model.parameters:
                tensorboard_writer.write_value(parameter.uid + "/mean", reduce_mean(parameter).eval(), epoch)

        if model_dir:
            model.save(os.path.join(model_dir, network_name + "_{}.dnn".format(epoch)))
        enable_profiler()  # begin to collect profiler data after first epoch


    # Evaluation parameters
    test_epoch_size = 10000
    minibatch_size = 32

    # process minibatches and evaluate the model
    metric_numer = 0
    metric_denom = 0
    sample_count = 0

    while sample_count < test_epoch_size:
        current_minibatch = min(minibatch_size, test_epoch_size - sample_count)
        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)
        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch
        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples

    print("")
    trainer.summarize_test_progress()
    print("")
    elapsed = (time.clock() - start)
    return 1-metric_numer / metric_denom, elapsed


if __name__ == '__main__':
    epochs = 200
    epoch_size = 50000
    network_name = "resnet20"
    model_dir = os.path.join(abs_path, "Models")
    log_dir = None # None for Console Out,File path for log file
    tensorboard_logdir = "./tfboard_log/"
    minibatch_size = 32

    reader_train = create_image_mb_source(os.path.join(data_path, 'train_map.txt'),
                                          os.path.join(data_path, 'CIFAR-10_mean.xml'),
                                          True, total_number_of_samples=epochs * epoch_size)

    reader_test = create_image_mb_source(os.path.join(data_path, 'test_map.txt'),
                                         os.path.join(data_path, 'CIFAR-10_mean.xml'),
                                         False, total_number_of_samples=C.io.FULL_DATA_SWEEP)

    accuracy, time = train_and_evaluate(reader_train, reader_test, network_name, epoch_size, epochs, minibatch_size,
                        model_dir, log_dir, tensorboard_logdir, False, False)

    print("acc=%f,Time used=%f s" % (accuracy, time))
    i=1
    while os.path.exists("log_"+str(i)+".txt"):
        i=i+1
    output = open("log_"+str(i)+".txt", 'w')
    output.write("acc=%f,Time used=%f s" % (accuracy, time))