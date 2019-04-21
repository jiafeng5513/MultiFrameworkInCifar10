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
from resnet_models import *
import cntk.io.transforms as xforms

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
    if train:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=(0.8, 1.0), jitter_type='uniratio')  # train uses jitter
        ]
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

    dtype = np.float16 if fp16 else np.float32
    if fp16:
        graph_input = C.cast(input_var, dtype=np.float16)
        graph_label = C.cast(label_var, dtype=np.float16)
    else:
        graph_input = input_var
        graph_label = label_var

    with C.default_options(dtype=dtype):
        # create model, and configure learning parameters
        if network_name == 'resnet20':
            z = create_cifar10_model(graph_input, 3, num_classes)
            lr_per_mb = [1.0] * 80 + [0.1] * 40 + [0.01]
        elif network_name == 'resnet110':
            z = create_cifar10_model(graph_input, 18, num_classes)
            lr_per_mb = [0.1] * 1 + [1.0] * 80 + [0.1] * 40 + [0.01]
        else:
            raise RuntimeError("Unknown model name!")

        # loss and metric
        ce = cross_entropy_with_softmax(z, graph_label)
        pe = classification_error(z, graph_label)

    if fp16:
        ce = C.cast(ce, dtype=np.float32)
        pe = C.cast(pe, dtype=np.float32)

    # shared training parameters


    # Set learning parameters
    lr_per_sample = []
    check_point=[80,120,160,180]
    #check_point = [5, 10, 15, 20]
    lrs=[1e-3,1e-4,1e-5,1e-6,5e-7]
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
        ProgressPrinter(tag='Training', log_to_file=log_dir, num_epochs=max_epochs, gen_heartbeat=gen_heartbeat)]
    tensorboard_writer = None
    if tensorboard_logdir is not None:
        tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=tensorboard_logdir, model=z)
        progress_writers.append(tensorboard_writer)

    # trainer object
    l2_reg_weight = 0.0001
    learner = adam(z.parameters, lr=lr_schedule, momentum=mm_schedule)
    trainer = Trainer(z, (ce, pe), learner, progress_writers)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    log_number_of_parameters(z);
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
            for parameter in z.parameters:
                tensorboard_writer.write_value(parameter.uid + "/mean", reduce_mean(parameter).eval(), epoch)

        if model_dir:
            z.save(os.path.join(model_dir, network_name + "_{}.dnn".format(epoch)))
        enable_profiler()  # begin to collect profiler data after first epoch


    # Evaluation parameters
    test_epoch_size = 10000
    minibatch_size = 16

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
    return metric_numer / metric_denom, elapsed


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

    print("error rate=%f,Time used=%f s" % (accuracy, time))
    i=1
    while os.path.exists("log_"+str(i)+".txt"):
        i=i+1
    output = open("log_"+str(i)+".txt", 'w')
    output.write("error rate=%f,Time used=%f s" % (accuracy, time))