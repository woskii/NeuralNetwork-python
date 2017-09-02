import numpy as np

import layers
import layer

global global_step   #全局的训练步长，需要初始化
global count
count = 0

def Sequence():
    global count
    count+=1
    return layers.layers("model"+str(count-1))

def fc(input_size, s, l2_loss, is_bias):
    return layer.fc(input_size, s, l2_loss, is_bias)

def relu():
    return layer.relu()

def reshape(t_shape):
    return layer.reshape(t_shape)

def sigmoid():
    return layer.sigmoid()

def tanh():
    return layer.tanh()

def convolution(stride, in_channel, kernelsize_h, kernelsize_w, kernel_num, l2_loss, is_bias, padding):
    return layer.convolution(stride, in_channel, kernelsize_h, kernelsize_w, kernel_num, l2_loss, is_bias, padding)

def avg_pool(height, width, stride):
    return layer.avg_pool(height, width, stride)

def max_pool(height, width, stride):
    return layer.max_pool(height, width, stride)

def dropout(level):
    return layer.dropout(level)

def load_model(layers, file_path, type):
    if type == "sgd":
        f = np.load("global_sgd.npz")
    elif type == "adam":
        f = np.load("global_adam.npz")
    global global_step
    global_step = f["globalstep"]
    layers.load(file_path)

def save_model(layers, file_path, type):
    global global_step
    if type == "sgd":
        np.savez("global_sgd.npz", globalstep = global_step)
    elif type == "adam":
        np.savez("global_adam.npz", globalstep=global_step)
    layers.save(file_path)

def xavier_initializer():
    return lambda w: w + np.random.randn(np.shape(w)[0], np.shape(w)[1]) / np.sqrt(np.shape(w)[1])

def constant_initializer(constant):
    return lambda x: x+constant

def gaussian_initializer():
    return lambda w: w + np.random.randn(np.shape(w)[0], np.shape(w)[1], np.shape(w)[2], np.shape(w)[3])

def init_model(ls, initialier_dict, globalstep = np.array(0)):
    global global_step
    global_step = globalstep
    ls.init_model(initialier_dict)

def get_global_step():
    global global_step
    return global_step






