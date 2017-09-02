'''''
    使用python解析二进制文件
'''
import numpy as np
import struct


def loadImageSet(filename):
    binfile = open(filename, 'rb')  # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height])  # reshape为[60000,784]型数组

    return imgs, head


def loadLabelSet(filename):
    binfile = open(filename, 'rb')  # 读二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)  # 取label文件前2个整形数

    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    numString = '>' + str(labelNum) + "B"  # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset)  # 取label数据

    binfile.close()
    labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)

    return labels, head

def vectorized_result(j):  #向量化
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

if __name__ == "__main__":
    file1 = 'E:/intelligence center/task_3/mnist data/train-images.idx3-ubyte'
    file2 = 'E:/intelligence center/task_3/mnist data/train-labels.idx1-ubyte'
    number = 60000
    imgs1, data_head = loadImageSet(file1)
    print('data_head:', data_head)   #魔数（可以理解为文件id），图片数，图片纵向像素数，图片横向像素数
    #print(type(imgs1))
    print('imgs1_array:', imgs1)
    #print(np.reshape(imgs[1, :], [28, 28]))  # 取出其中一张图片的像素，转型为28*28，大致就能从图像上看出是几啦
    #print(np.reshape(imgs[0], [28, 28]))   #5
    imgs = []
    #for i in range(0, number):
        #imgs.append(imgs1[i]/512)


    #print('看看', np.reshape(imgs[0], [28, 28]))

    print('----------我是分割线-----------')

    labels, labels_head = loadLabelSet(file2)
    print('labels_head:', labels_head)
    print(type(labels))
    print(labels[0])   #5
    mark = []   #记录位置
    label_list = []
    count0 = 0
    count1 = 0
    for i in range(0, number):
        #label_list.append(vectorized_result(labels[i]))
        if labels[i] == 0 :
            count0 +=1
            imgs.append(np.reshape(imgs1[i] / 512, [28, 28]))
            label_list.append(vectorized_result(labels[i]))
            mark.append(i)
        elif labels[i] == 1:
            imgs.append(imgs1[i] / 512)
            label_list.append(vectorized_result(labels[i]))
            mark.append(i)
    np.savez("img01_8000_train.npz",np.array(imgs[:8000]))    #(8000,28,28)
    np.savez("label01_8000_train.npz", np.array(label_list[:8000]))   #(8000,2,1)
    np.savez("img01_2000_test.npz", np.array(imgs[8000:10000]))     #(2000,28,28)
    np.savez("label01_2000_test.npz", np.array(label_list[8000:10000]))    #(2000,2,1)

    #print("0:",count0)
    #print("1:", count1)
    #print("location:", mark[0],mark[1],mark[2],mark[3],mark[4])
    #print(np.reshape(imgs1[1], [28, 28]))    #0
    #print(np.reshape(imgs1[6], [28, 28]))   #1
    #print(label_list[6])