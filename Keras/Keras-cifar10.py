from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
import time

"""
epoch:全部样本训练的次数
iteration：1个iteration等于使用batchsize个样本训练一次
"""

batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
num_classes = 10
subtract_pixel_mean = False
# n = 3
# version = 1
# depth = n * 6 + 2
# model_type = 'ResNet%dv%d' % (depth, version)
# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_test_mean = np.mean(x_test, axis=0)
    x_train -= x_train_mean
    x_test -= x_test_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print('y_train shape:', y_train.shape)
print(x_test.shape[0], 'test samples')


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

check_point = [80, 120, 160, 180]
lrs = [3e-2, 3e-3, 3e-4, 3e-4, 5e-5]


def lr_schedule(epoch):
    """
    学习率
    """
    if epoch in range(0, check_point[0]):
        K.set_value(model.optimizer.lr, lrs[0])
    if epoch in range(check_point[0], check_point[1]):
        K.set_value(model.optimizer.lr, lrs[1])
    if epoch in range(check_point[1], check_point[2]):
        K.set_value(model.optimizer.lr, lrs[2])
    if epoch in range(check_point[2], check_point[3]):
        K.set_value(model.optimizer.lr, lrs[3])
    if epoch >= check_point[3]:
        K.set_value(model.optimizer.lr, lrs[4])
    print("learning rate = %f" % K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)

def ResBlock_v1(input,num_filters):
    conv1 = Conv2D(filters=num_filters, kernel_size=3, strides=1, padding='same')(input)
    bn1=BatchNormalization()(conv1)
    r1=Activation('relu')(bn1)
    conv2 = Conv2D(filters=num_filters, kernel_size=3, strides=1, padding='same')(r1)
    bn2=BatchNormalization()(conv2)
    out=Activation('relu')(keras.layers.add([input, bn2]))
    return out

def ResBlock_v2(input,num_filters):
    conv1 = Conv2D(filters=num_filters, kernel_size=3, strides=2, padding='same')(input)
    bn1=BatchNormalization()(conv1)
    r1=Activation('relu')(bn1)
    conv2 = Conv2D(filters=num_filters, kernel_size=3, strides=1, padding='same')(r1)
    bn2=BatchNormalization()(conv2)

    conv3= Conv2D(filters=num_filters, kernel_size=1, strides=2, padding='same')(input)
    bn3=BatchNormalization()(conv3)
    out = Activation('relu')(keras.layers.add([bn3, bn2]))
    return out

def resnet_v1(input_shape, num_classes=10):
    input = Input(shape=input_shape)
    conv1 = Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(input)
    bn1 = BatchNormalization()(conv1)
    ac1 = Activation('relu')(bn1)

    rb1 = ResBlock_v1(ac1, 16)
    rb2 = ResBlock_v1(rb1, 16)
    rb3 = ResBlock_v1(rb2, 16)

    rb4 = ResBlock_v2(rb3, 32)
    rb5 = ResBlock_v1(rb4, 32)
    rb6 = ResBlock_v1(rb5, 32)

    rb7 = ResBlock_v2(rb6, 64)
    rb8 = ResBlock_v1(rb7, 64)
    rb9 = ResBlock_v1(rb8, 64)

    x = AveragePooling2D(pool_size=8)(rb9)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax',kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=input, outputs=outputs)
    return model


'''训练和测试开始'''
model = resnet_v1(input_shape=input_shape)
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()

model_type='resnet20'
# Prepare model model saving directory.
save_dir = './saved_models'
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)
change_lr = LearningRateScheduler(lr_schedule)
# Run training
start = time.clock()
""""""
print('******Training start********')
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),
          shuffle=True, callbacks=[checkpoint, change_lr])

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
elapsed = (time.clock() - start)
print("Time used (s):", elapsed)

i = 1
while os.path.exists("log_" + str(i) + ".txt"):
    i = i + 1
output = open("log_" + str(i) + ".txt", 'w')
output.write("Test loss:=%f,Test accuracy=%f,Time used %f s" % (scores[0], scores[1], elapsed))
