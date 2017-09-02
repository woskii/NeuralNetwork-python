import numpy as np

import nn

global batch_count
batch_count = 0

global training_data
global training_target
training_data = []
training_target = []

global data_batches
global target_batches
data_batches = []
target_batches = []

def block(inputChannels, outputChannels, l2_loss, is_bias, is_relu):
    fc = nn.Sequence()
    fc.add(nn.fc(inputChannels, outputChannels, l2_loss, is_bias))
    if is_relu:
        fc.add(nn.relu())
    return fc

def get_max_index(x):
    re = np.where(x == np.max(x))
    return re[0][0]

def test_model(ls):
    n = 20
    test_data = []
    test_target = []
    d1 = np.load("img01_2000_test.npz")
    for i in range(0, n):
        test_data.append(d1["arr_0"][i])
    d2 = np.load("label01_2000_test.npz")
    for i in range(0, n):
        test_target.append(d2["arr_0"][i])
    logit = ls.infer(test_data, test = True)
    correct = 0
    for i in range(0, len(logit)):
        #print("test_logit", get_max_index(logit[i]))
        #print("test_target", get_max_index(test_target[i]))
        #print(np.max(logit[i]))
        if get_max_index(logit[i]) == get_max_index(test_target[i]) and np.max(logit[i]) > 0.7:
            correct +=1
    return correct/n

def load_training_data():  #将数据分成batchsize大小
    batch_size = 10
    n = 100
    epoch = n/batch_size
    global training_data
    global training_target
    f1 = np.load("img01_8000_train.npz")
    #print("data loading...")
    for i in range(0, n):
        training_data.append(f1["arr_0"][i])
        #print("到2   ",i)
    #print("data loading complete")
    f2 = np.load("label01_8000_train.npz")
    for i in range(0, n):
        training_target.append(f2["arr_0"][i])
    global data_batches
    global target_batches
    data_batches = [training_data[k:k + batch_size]
            for k in range(0, n, batch_size)]
    target_batches = [training_target[k:k + batch_size]
                    for k in range(0, n, batch_size)]

def get_training_data():
    global batch_count
    global data_batches
    global target_batches
    batch_count += 1
    #batch_count = 1
    n = 100
    #print("batch_count", batch_count)
    return data_batches[batch_count-1], target_batches[batch_count-1], n

def get_wd():  #待修改
    return 100.0

def learning_rate(step):
    lr = 0.1
    return lr*(0.1**(step//30))

def training_accur(logit, target, n):
    correct = 0
    for i in range(0, len(logit)):
        #print("test_logit", get_max_index(logit[i]))
        #print("test_target", get_max_index(test_target[i]))
        if get_max_index(logit[i]) == get_max_index(target[i]):
            correct += 1
    return correct / n

""" createModel """
'''
mid = 7*7*2
t_shape1 = (1, 28, 28)
t_shape2 = (mid, 1)
model = nn.Sequence()
model.add(nn.reshape(t_shape1))
model.add(nn.convolution(1, 1, 5, 5, 2, False, True, 2))  #stride, in_channel, kernelsize_h, kernelsize_w, kernel_num, l2_loss, is_bias, padding
model.add(nn.relu())
model.add(nn.max_pool(2, 2, 2))
model.add(nn.convolution(1, 2, 5, 5, 2, False, True, 2))
model.add(nn.relu())
model.add(nn.max_pool(2, 2, 2))
model.add(nn.reshape(t_shape2))
#model.add(block(mid, 1024, True, True, True))
model.add(nn.fc(mid, 2, False, False))
model.add(nn.sigmoid())
'''
t_shape1 = (1, 28, 28)
t_shape = (27*27*2, 1)
model = nn.Sequence()
model.add(nn.reshape(t_shape1))
model.add(nn.convolution(1, 1, 3, 3, 2, False, True, 1))  #stride, in_channel, kernelsize_h, kernelsize_w, kernel_num, l2_loss, is_bias, padding
model.add(nn.tanh())
model.add(nn.dropout(0.5))
model.add(nn.max_pool(2, 2, 1))
model.add(nn.reshape(t_shape))
sizes = [50, 30]
pre_sizes = [t_shape[0], 50]
for s, ps in zip(sizes, pre_sizes):
    model.add(block(ps, s, True, False, True))

model.add(nn.fc(30, 2, False, False))
model.add(nn.sigmoid())


""" parameter init """

initor = {
    'fc': {'weight':nn.xavier_initializer(), 'bias':nn.constant_initializer(0)},
    'bn': {'alpha':nn.constant_initializer(1), 'beta':nn.constant_initializer(0)},
    'conv': {'weight': nn.constant_initializer(1), 'bias': nn.constant_initializer(0)}
}


choice = input("please input:0-sgd,1-adam")
if choice == "0":
    #nn.init_model(model, initialier_dict = initor)      #如果没有已经训练好的模型，把这行注释取消进行初始化
    nn.load_model(model, "model_sgd.npz", "sgd")         #如果已经训练好模型，通过这一行进行加载模型数据
    max_steps = 10
    global_step = nn.get_global_step()
    f = open('record_sgd.txt', 'a')
    '''
    load_training_data()
    #print("initial set accuracy = ", test_model(model))
    for i in range(0, max_steps):
        data, target, n = get_training_data()    #n是整个训练集的大小
        #print("target0",target[0])
        wd = get_wd()
        lr = learning_rate(global_step)
        loss,logit = model.train(data, target, lr, wd, n)
        global_step+=1
        print(global_step, ":", "loss = ", loss, "training set accuracy = ", training_accur(logit, target, len(target)))
        f.write(str(global_step)+":"+"loss = "+str(loss)+"\n")
    nn.save_model(model, "model_sgd.npz", "sgd")
    '''
    print("test set accuracy = ", test_model(model))
    f.write("test set accuracy = "+str(test_model(model))+"\n")
    f.close()

elif choice == "1":
    #nn.init_model(model, initialier_dict = initor)
    nn.load_model(model, "model_adam.npz", "adam")
    max_steps = 10
    global_step = nn.get_global_step()
    f = open('record_adam.txt', 'a')
    load_training_data()
    #print("initial set accuracy = ", test_model(model))
    for i in range(0, max_steps):
        data, target, n = get_training_data()    #n是整个训练集的大小
        #print("target0",target[0])
        wd = get_wd()
        lr = learning_rate(global_step)
        loss,logit = model.train(data, target, lr, wd, n, method = "adam")
        global_step+=1
        print(global_step, ":", "loss = ", loss, "training set accuracy = ", training_accur(logit, target, len(target)))
        f.write(str(global_step)+":"+"loss = "+str(loss)+"\n")
    nn.save_model(model, "model_adam.npz", "adam")
    print("test set accuracy = ", test_model(model))
    f.write("test set accuracy = "+str(test_model(model))+"\n")
    f.close()