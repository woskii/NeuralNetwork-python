import numpy as np
import unittest
import copy

import layer
import l2_loss_vars

class TestFcLayer(unittest.TestCase):
    def assertNumpyArraysEqual(self, this, that):  # 比较np.array
        if this.shape != that.shape:
            raise AssertionError("Shapes don't match")
        if not np.allclose(this, that):
            raise AssertionError("Elements don't match!")

    def setUp(self):
        self.fclayer1 = layer.fc(3, 4, True, True)  # l2_loss = True is_bias = True的情况
        self.fclayer2 = layer.fc(3, 4, False, False)  # l2_loss = False is_bias = False的情况
        self.initor = {
            'fc': {'weight': self.xavier_initializer(), 'bias': self.constant_initializer(0)},
            'bn': {'alpha': self.constant_initializer(1), 'beta': self.constant_initializer(0)}
        }

    def test_init(self):
        self.assertEqual(self.fclayer1.input_size, 3)  #input_size
        self.assertEqual(self.fclayer1.size, 4)        #size
        self.assertEqual(self.fclayer1.l2_loss, True)  # l2_loss
        self.assertEqual(self.fclayer2.is_bias, False)  # is_bias
        self.assertEqual(self.fclayer2.module_name, "fc")

    def xavier_initializer(self):
        return lambda w: w + np.random.randn(np.shape(w)[0], np.shape(w)[1]) / np.sqrt(np.shape(w)[1])

    def constant_initializer(self, constant):
        return lambda x: x + constant

    def test_initial(self):
        self.fclayer1.initial(self.initor)
        self.fclayer2.initial(self.initor)
        #print(l2_loss_vars.l2_loss_vars)
        self.assertEqual(np.shape(self.fclayer1.weight), (4, 3))  # weight初始化
        self.assertNumpyArraysEqual(self.fclayer1.bias, np.zeros((4, 1)))    # bias初始化
        self.assertNumpyArraysEqual(self.fclayer1.ms, np.zeros((4, 3)))
        self.assertNumpyArraysEqual(self.fclayer1.vs, np.zeros((4, 3)))
        self.assertNumpyArraysEqual(self.fclayer1.mb, np.zeros((4, 1)))
        self.assertNumpyArraysEqual(self.fclayer1.vb, np.zeros((4, 1)))
        #self.assertEqual(np.shape(l2_loss_vars.l2_loss_vars[0]), (4, 3))  #正则化参数成功添加 (卷积也测试了)
        #self.assertEqual(len(l2_loss_vars.l2_loss_vars), 1)  # 正则化参数长度

    def test_forward_prop(self):
        self.fclayer1.initial(self.initor)   #若未初始化报错
        input = [np.arange(3).reshape(3, 1), np.arange(3).reshape(3, 1)]
        self.fclayer1.forward_prop(input)
        self.assertEqual(np.shape(self.fclayer1.z[0]), (4, 1))
        self.assertEqual(len(self.fclayer1.z), 2)

    def test_backward_prop(self):    #sgd
        self.fclayer1.initial(self.initor)
        input = [np.arange(3).reshape(3, 1), np.arange(3).reshape(3, 1)]
        self.fclayer1.forward_prop(input)
        old_l2 = copy.deepcopy(l2_loss_vars.l2_loss_vars[0])
        #print(old_l2)
        #print(self.fclayer1.weight)
        old_id = id(self.fclayer1.weight)
        old_w = copy.deepcopy(self.fclayer1.weight)
        old_b = copy.deepcopy(self.fclayer1.bias)
        output = self.fclayer1.backward_prop([np.arange(4).reshape(4, 1),np.arange(4).reshape(4, 1)], 2, 2, 2)
        self.assertEqual(len(output), 2)
        self.assertEqual(np.shape(output[0]), (3, 1))
        #self.assertNumpyArraysEqual(self.fclayer1.weight, old_w)   #不相同说明w更新了
        #self.assertNumpyArraysEqual(self.fclayer1.bias, old_b)     #不相同说明b更新了
        #self.assertNumpyArraysEqual(l2_loss_vars.l2_loss_vars[0], old_l2)  #不相同说明l2_loss_vars更新了 现在下标应该是1
        self.assertEqual(old_id, id(self.fclayer1.weight))
        #self.assertEqual(id(l2_loss_vars.l2_loss_vars[3]), id(self.fclayer1.weight))

    def test_backward_prop(self):     #adam
        self.fclayer1.initial(self.initor)
        input = [np.arange(3).reshape(3, 1), np.arange(3).reshape(3, 1)]
        self.fclayer1.forward_prop(input)
        old_l2 = copy.deepcopy(l2_loss_vars.l2_loss_vars[0])
        # print(old_l2)
        # print(self.fclayer1.weight)
        old_id = id(self.fclayer1.weight)
        old_w = copy.deepcopy(self.fclayer1.weight)
        old_b = copy.deepcopy(self.fclayer1.bias)
        old_ms = copy.deepcopy(self.fclayer1.ms)
        old_vs = copy.deepcopy(self.fclayer1.vs)
        old_mb = copy.deepcopy(self.fclayer1.mb)
        old_vb = copy.deepcopy(self.fclayer1.vb)
        output = self.fclayer1.backward_prop([np.arange(4).reshape(4, 1), np.arange(4).reshape(4, 1)], 2, 2, 2, method = "adam")
        self.assertEqual(len(output), 2)
        self.assertEqual(np.shape(output[0]), (3, 1))
        #self.assertNumpyArraysEqual(self.fclayer1.ms, old_ms)  # 不相同说明ms更新了
        #self.assertNumpyArraysEqual(self.fclayer1.vs, old_vs)  # 不相同说明vs更新了
        #self.assertNumpyArraysEqual(self.fclayer1.mb, old_mb)  # 不相同说明mb更新了
        #self.assertNumpyArraysEqual(self.fclayer1.vb, old_vb)  # 不相同说明vb更新了
        #self.assertNumpyArraysEqual(self.fclayer1.weight, old_w)   #不相同说明w更新了
        #self.assertNumpyArraysEqual(self.fclayer1.bias, old_b)     #不相同说明b更新了
        #self.assertNumpyArraysEqual(l2_loss_vars.l2_loss_vars[0], old_l2)  #不相同说明l2_loss_vars更新了 现在下标应该是1
        self.assertEqual(old_id, id(self.fclayer1.weight))
        # self.assertEqual(id(l2_loss_vars.l2_loss_vars[3]), id(self.fclayer1.weight))

    def test_load_para(self):
        self.fclayer1.instance_name = "fc0"
        np.savez("saved_test.npz", fc0weight = np.zeros((4, 3)), fc0bias = np.arange(4).reshape(4, 1), fc0ms = np.zeros((4, 3)),
                 fc0vs = np.zeros((4, 3)), fc0mb = np.zeros((4, 1)), fc0vb = np.zeros((4, 1)))
        d = np.load("saved_test.npz")
        self.fclayer1.load_para(d)
        self.assertNumpyArraysEqual(self.fclayer1.weight, np.zeros((4, 3)))
        self.assertNumpyArraysEqual(self.fclayer1.bias, np.arange(4).reshape(4, 1))
        self.assertNumpyArraysEqual(self.fclayer1.ms, np.zeros((4, 3)))
        self.assertNumpyArraysEqual(self.fclayer1.vs, np.zeros((4, 3)))
        self.assertNumpyArraysEqual(self.fclayer1.mb, np.zeros((4, 1)))
        self.assertNumpyArraysEqual(self.fclayer1.vb, np.zeros((4, 1)))

    def test_save_para(self):
        self.fclayer1.instance_name = "fc0"
        self.fclayer1.weight = np.arange(12).reshape(4, 3)
        self.fclayer1.bias = np.arange(4).reshape(4, 1)
        self.fclayer1.ms = np.arange(12).reshape(4, 3)
        self.fclayer1.vs = np.arange(12).reshape(4, 3)
        self.fclayer1.mb = np.arange(4).reshape(4, 1)
        self.fclayer1.vb = np.arange(4).reshape(4, 1)
        self.fclayer1.save_para("st.npz")
        self.fclayer2.instance_name = "fc1"
        self.fclayer2.weight = np.zeros((4, 3))
        self.fclayer2.ms = np.arange(12).reshape(4, 3)
        self.fclayer2.vs = np.arange(12).reshape(4, 3)
        self.fclayer2.save_para("st.npz")
        f = np.load("st.npz")
        self.assertNumpyArraysEqual(self.fclayer2.weight, f["fc1weight"])
        self.assertNumpyArraysEqual(self.fclayer2.ms, f["fc1ms"])
        self.assertNumpyArraysEqual(self.fclayer2.vs, f["fc1vs"])
        self.assertNumpyArraysEqual(self.fclayer1.weight, f["fc0weight"])
        self.assertNumpyArraysEqual(self.fclayer1.bias, f["fc0bias"])
        self.assertNumpyArraysEqual(self.fclayer1.ms, f["fc0ms"])
        self.assertNumpyArraysEqual(self.fclayer1.vs, f["fc0vs"])
        self.assertNumpyArraysEqual(self.fclayer1.mb, f["fc0mb"])
        self.assertNumpyArraysEqual(self.fclayer1.vb, f["fc0vb"])

class TestReshape(unittest.TestCase):
    def setUp(self):
        self.re_layer1 = layer.reshape((4, 1))

    def assertNumpyArraysEqual(self, this, that):  # 比较np.array
        if this.shape != that.shape:
            raise AssertionError("Shapes don't match")
        if not np.allclose(this, that):
            raise AssertionError("Elements don't match!")

    def test_forward_prop(self):
        s_input = [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])]
        output = [np.array([[1], [2], [3], [4]]), np.array([[1], [2], [3], [4]])]
        self.assertEqual(len(self.re_layer1.forward_prop(s_input)), 2)
        self.assertNumpyArraysEqual(self.re_layer1.forward_prop(s_input)[0], output[0])

    def test_backward_prop(self):
        s_input = [np.array([[[1], [2]], [[3], [4]]]), np.array([[[1], [2]], [[3], [4]]])]
        output = [np.array([[1], [2], [3], [4]]), np.array([[1], [2], [3], [4]])]
        self.re_layer1.forward_prop(s_input)
        backout = self.re_layer1.backward_prop(s_input, 1, 1, 1)
        self.assertEqual(len(backout), 2)
        self.assertNumpyArraysEqual(backout[0], s_input[0])

class TestRelu(unittest.TestCase):
    def setUp(self):
        self.relu_layer1 = layer.relu()

    def assertNumpyArraysEqual(self, this, that):  # 比较np.array
        if this.shape != that.shape:
            raise AssertionError("Shapes don't match")
        if not np.allclose(this, that):
            raise AssertionError("Elements don't match!")

    def test_forward_prop(self):
        s_input = [np.array([[1], [-2], [0], [4], [-3]])]
        output = np.array([[1], [0], [0], [4], [0]])
        self.assertNumpyArraysEqual(self.relu_layer1.forward_prop(s_input)[0], output)

    def test_backward_prop(self):
        s_input = [np.array([[1], [-2], [0], [4], [-3]])]
        self.relu_layer1.forward_prop(s_input)
        output = np.array([[1], [0], [0], [1], [0]])
        self.assertNumpyArraysEqual(self.relu_layer1.backward_prop(s_input, 1, 1, 1)[0], np.array([[1], [0], [0], [4], [0]]))

class TestSigmoid(unittest.TestCase):
	def setUp(self):
		self.sig_layer1 = layer.sigmoid()

	def assertNumpyArraysEqual(self,this,that):   #比较np.array
		if this.shape != that.shape:
			raise AssertionError("Shapes don't match")
		if not np.allclose(this,that):
			raise AssertionError("Elements don't match!")

	def test_forward_prop(self):
		s_input = [np.array([[0], [0], [0], [0], [0]])]
		output = [np.array([[0.5], [0.5], [0.5], [0.5], [0.5]])]
		self.assertNumpyArraysEqual(self.sig_layer1.forward_prop(s_input)[0], output[0])

	def test_backward_prop(self):
		s_input = [np.array([[0], [0], [0], [0], [0]])]
		output = np.array([[0.25], [0.25], [0.25], [0.25], [0.25]])
		self.sig_layer1.forward_prop(s_input)
		self.assertNumpyArraysEqual(self.sig_layer1.backward_prop([np.arange(5).reshape(5, 1)], 1, 1, 1)[0], np.array([[0], [0.25], [0.5], [0.75], [1]]))

class TestTanh(unittest.TestCase):
    def setUp(self):
        self.tanh_layer1 = layer.tanh()

    def assertNumpyArraysEqual(self, this, that):  # 比较np.array
        if this.shape != that.shape:
            raise AssertionError("Shapes don't match")
        if not np.allclose(this, that):
            raise AssertionError("Elements don't match!")

    def test_forward_prop(self):
        s_input = [np.array([[0], [0], [0], [0], [0]])]
        output = np.tanh(s_input[0])
        self.assertNumpyArraysEqual(self.tanh_layer1.forward_prop(s_input)[0], output)

    def test_backward_prop(self):
        s_input = [np.array([[0], [0], [0], [0], [0]])]
        self.tanh_layer1.forward_prop(s_input)
        output = self.tanh_layer1.backward_prop([np.arange(5).reshape(5, 1)], 1, 1, 1)[0]
        self.assertEqual(np.shape(output), (5, 1))

class TestConvolution(unittest.TestCase):
    def assertNumpyArraysEqual(self, this, that):  # 比较np.array
        if this.shape != that.shape:
            raise AssertionError("Shapes don't match")
        if not np.allclose(this, that):
            raise AssertionError("Elements don't match!")

    def setUp(self):
        self.conv1 = layer.convolution(2, 3, 1, 2, 3, True, True, 2)  # l2_loss = True is_bias = True的情况
        #stride, in_channel, kernelsize_h, kernelsize_w, kernel_num, l2_loss, is_bias, padding
        self.conv2 = layer.convolution(1, 3, 2, 3, 2, False, True, 0)  # l2_loss = False is_bias = False的情况
        self.initor = {
            'conv': {'weight': self.gaussian_initializer(), 'bias': self.constant_initializer(0)}
        }

    def test_init(self):
        self.assertEqual(self.conv1.stride, 2)
        self.assertEqual(self.conv1.module_name, "conv")
        self.assertEqual(self.conv1.in_channel, 3)
        self.assertEqual(self.conv1.kernel_height, 1)
        self.assertEqual(self.conv1.kernel_width, 2)
        self.assertEqual(self.conv1.kernel_num, 3)
        self.assertEqual(self.conv1.l2_loss, True)  # l2_loss
        self.assertEqual(self.conv1.is_bias, True)  # is_bias
        self.assertEqual(self.conv1.pad, 2)

    def constant_initializer(self, constant):
        return lambda x: x + constant

    def gaussian_initializer(self):
        return lambda w: w + 0.001 * np.random.randn(np.shape(w)[0], np.shape(w)[1], np.shape(w)[2], np.shape(w)[3])

    def test_initial(self):
        self.conv1.initial(self.initor)
        self.conv2.initial(self.initor)
        #print(l2_loss_vars.l2_loss_vars[0])
        self.assertEqual(np.shape(self.conv1.weight), (3, 3, 1, 2))  # weight初始化
        self.assertNumpyArraysEqual(self.conv1.bias, np.zeros((3, )))    # bias初始化
        self.assertNumpyArraysEqual(self.conv1.ms, np.zeros((3, 3, 1, 2)))
        self.assertNumpyArraysEqual(self.conv1.vs, np.zeros((3, 3, 1, 2)))
        self.assertNumpyArraysEqual(self.conv1.mb, np.zeros((3,)))
        self.assertNumpyArraysEqual(self.conv1.vb, np.zeros((3,)))
        #self.assertEqual(np.shape(l2_loss_vars.l2_loss_vars[0]), (3, 3, 1, 2))  #正则化参数成功添加
        #self.assertEqual(len(l2_loss_vars.l2_loss_vars), 1)  # 正则化参数长度

    def test_forward_prop(self):
        #self.conv2.initial(self.initor)  # 若未初始化报错
        self.conv2.weight = np.ones((2, 3, 2, 3))
        self.conv2.bias = np.array([1,2])
        input = [np.ones((3, 4, 4)), np.ones((3, 4, 4))]
        out = self.conv2.forward_prop(input)
        #print(out)
        self.assertEqual(np.shape(out[0]), (2, 3, 2))
        self.assertEqual(len(out), 2)

    def test_backward_prop(self):     #sgd
        self.conv2.initial(self.initor)
        self.conv2.weight = np.ones((2, 3, 2, 3))
        self.conv2.bias = np.array([1, 2])
        input = [np.ones((3, 4, 4)), np.ones((3, 4, 4))]
        out = self.conv2.forward_prop(input)
        #self.assertEqual(id(l2_loss_vars.l2_loss_vars[0]), id(self.conv2.weight))
        Ho = (np.shape(input[0])[1] + 2 * 0 - np.shape(self.conv2.weight)[2]) // 1 + 1
        Wo = (np.shape(input[0])[2] + 2 * 0 - np.shape(self.conv2.weight)[3]) // 1 + 1
        dout = [np.ones((np.shape(self.conv2.weight)[0], Ho, Wo)), np.ones((np.shape(self.conv2.weight)[0], Ho, Wo))]
        old_id = id(self.conv2.weight)
        #old_l2 = copy.deepcopy(l2_loss_vars.l2_loss_vars[0])
        old_w = copy.deepcopy(self.conv2.weight)
        old_b = copy.deepcopy(self.conv2.bias)
        backout = self.conv2.backward_prop(dout, 2, 1, 4)
        #print("a", self.conv2.weight)
        #print("G", backout)
        self.assertEqual(np.shape(backout[0]), (3, 4, 4))
        self.assertEqual(len(backout), 2)
        #self.assertNumpyArraysEqual(self.conv2.weight, old_w)   #不相同说明w更新了
        #self.assertNumpyArraysEqual(self.conv2.bias, old_b)     #不相同说明b更新了
        #self.assertNumpyArraysEqual(l2_loss_vars.l2_loss_vars[0], old_l2)  #不相同说明l2_loss_vars更新了
        self.assertEqual(old_id, id(self.conv2.weight))
        #self.assertEqual(id(l2_loss_vars.l2_loss_vars[0]), id(self.conv2.weight))

    def test_backward_prop(self):       #adam
        self.conv2.initial(self.initor)
        self.conv2.weight = np.ones((2, 3, 2, 3))
        self.conv2.bias = np.array([1, 2])
        input = [np.ones((3, 4, 4)), np.ones((3, 4, 4))]
        out = self.conv2.forward_prop(input)
        #self.assertEqual(id(l2_loss_vars.l2_loss_vars[0]), id(self.conv2.weight))
        Ho = (np.shape(input[0])[1] + 2 * 0 - np.shape(self.conv2.weight)[2]) // 1 + 1
        Wo = (np.shape(input[0])[2] + 2 * 0 - np.shape(self.conv2.weight)[3]) // 1 + 1
        dout = [np.ones((np.shape(self.conv2.weight)[0], Ho, Wo)), np.ones((np.shape(self.conv2.weight)[0], Ho, Wo))]
        old_id = id(self.conv2.weight)
        #old_l2 = copy.deepcopy(l2_loss_vars.l2_loss_vars[0])
        old_w = copy.deepcopy(self.conv2.weight)
        old_b = copy.deepcopy(self.conv2.bias)
        old_ms = copy.deepcopy(self.conv2.ms)
        old_vs = copy.deepcopy(self.conv2.vs)
        old_mb = copy.deepcopy(self.conv2.mb)
        old_vb = copy.deepcopy(self.conv2.vb)
        backout = self.conv2.backward_prop(dout, 2, 1, 4, method = "adam")
        #print("a", self.conv2.weight)
        #print("G", backout)
        self.assertEqual(np.shape(backout[0]), (3, 4, 4))
        self.assertEqual(len(backout), 2)
        #self.assertNumpyArraysEqual(self.conv2.ms, old_ms)    # 不相同说明ms更新了
        #self.assertNumpyArraysEqual(self.conv2.vs, old_vs)  # 不相同说明vs更新了
        #self.assertNumpyArraysEqual(self.conv2.mb, old_mb)  # 不相同说明mb更新了
        #self.assertNumpyArraysEqual(self.conv2.vb, old_vb)  # 不相同说明vb更新了
        #self.assertNumpyArraysEqual(self.conv2.weight, old_w)   #不相同说明w更新了
        #self.assertNumpyArraysEqual(self.conv2.bias, old_b)     #不相同说明b更新了
        #self.assertNumpyArraysEqual(l2_loss_vars.l2_loss_vars[0], old_l2)  #不相同说明l2_loss_vars更新了
        self.assertEqual(old_id, id(self.conv2.weight))     #地址没变
        #self.assertEqual(id(l2_loss_vars.l2_loss_vars[0]), id(self.conv2.weight))

    def test_load_para(self):
        self.conv2.instance_name = "conv0"
        np.savez("saved_test.npz", conv0weight = np.zeros((2, 3, 2, 3)), conv0bias = np.arange(2).reshape(2, ), conv0ms = np.zeros((2, 3, 2, 3)),
                 conv0vs = np.zeros((2, 3, 2, 3)), conv0mb = np.zeros((2, )), conv0vb = np.zeros((2, )))
        d = np.load("saved_test.npz")
        self.conv2.load_para(d)
        self.assertNumpyArraysEqual(self.conv2.weight, np.zeros((2, 3, 2, 3)))
        self.assertNumpyArraysEqual(self.conv2.bias, np.arange(2).reshape(2, ))
        self.assertNumpyArraysEqual(self.conv2.ms, np.zeros((2, 3, 2, 3)))
        self.assertNumpyArraysEqual(self.conv2.vs, np.zeros((2, 3, 2, 3)))
        self.assertNumpyArraysEqual(self.conv2.mb, np.zeros((2, )))
        self.assertNumpyArraysEqual(self.conv2.vb, np.zeros((2, )))

    def test_save_para(self):
        self.conv2.instance_name = "conv0"
        self.conv2.weight = np.arange(36).reshape(2, 3, 2, 3)
        self.conv2.bias = np.arange(2).reshape(2, )
        self.conv2.ms = np.arange(36).reshape(2, 3, 2, 3)
        self.conv2.vs = np.arange(36).reshape(2, 3, 2, 3)
        self.conv2.mb = np.arange(2).reshape(2, )
        self.conv2.vb = np.arange(2).reshape(2, )
        self.conv2.save_para("st.npz")
        f = np.load("st.npz")
        self.assertNumpyArraysEqual(self.conv2.weight, f["conv0weight"])
        self.assertNumpyArraysEqual(self.conv2.bias, f["conv0bias"])
        self.assertNumpyArraysEqual(self.conv2.ms, f["conv0ms"])
        self.assertNumpyArraysEqual(self.conv2.vs, f["conv0vs"])
        self.assertNumpyArraysEqual(self.conv2.mb, f["conv0mb"])
        self.assertNumpyArraysEqual(self.conv2.vb, f["conv0vb"])

class TestMax_pool(unittest.TestCase):

    def setUp(self):
        self.mp = layer.max_pool(2, 1, 2)   #height, width, stride

    def test_init(self):
        self.assertEqual(self.mp.s, 2)
        self.assertEqual(self.mp.HH, 2)
        self.assertEqual(self.mp.WW, 1)
        self.assertEqual(self.mp.module_name, "max_pool")

    def test_forward_prop(self):
        input = [np.ones((3, 4, 4)), np.ones((3, 4, 4))]
        N = len(input)
        C, H, W = input[0].shape
        H_new = 1 + (H - 2) // 2
        W_new = 1 + (W - 1) // 2
        out = self.mp.forward_prop(input)
        self.assertEqual(len(out), 2)
        self.assertEqual(np.shape(out[0]), (C, H_new, W_new))

    def test_backward_compute(self):
        input = [np.ones((3, 4, 4)), np.ones((3, 4, 4))]
        N = len(input)
        C, H, W = input[0].shape
        H_new = 1 + (H - 2) // 2
        W_new = 1 + (W - 1) // 2
        out = self.mp.forward_prop(input)
        dout = [np.ones((C, H_new, W_new)), np.ones((C, H_new, W_new))]
        backout = self.mp.backward_prop(dout, 1, 1, 1)
        self.assertEqual(len(backout), 2)
        self.assertEqual(np.shape(backout[0]), (C, H, W))

class TestAvg_pool(unittest.TestCase):

    def setUp(self):
        self.ap = layer.avg_pool(2, 1, 2)   #height, width, stride

    def test_init(self):
        self.assertEqual(self.ap.s, 2)
        self.assertEqual(self.ap.HH, 2)
        self.assertEqual(self.ap.WW, 1)
        self.assertEqual(self.ap.module_name, "avg_pool")

    def test_forward_prop(self):
        input = [np.ones((3, 4, 4)), np.ones((3, 4, 4))]
        N = len(input)
        C, H, W = input[0].shape
        H_new = 1 + (H - 2) // 2
        W_new = 1 + (W - 1) // 2
        out = self.ap.forward_prop(input)
        self.assertEqual(len(out), 2)
        self.assertEqual(np.shape(out[0]), (C, H_new, W_new))

    def test_backward_compute(self):
        input = [np.ones((3, 4, 4)), np.ones((3, 4, 4))]
        N = len(input)
        C, H, W = input[0].shape
        H_new = 1 + (H - 2) // 2
        W_new = 1 + (W - 1) // 2
        out = self.ap.forward_prop(input)
        dout = [np.ones((C, H_new, W_new)), np.ones((C, H_new, W_new))]
        backout = self.ap.backward_prop(dout, 1, 1, 1)
        self.assertEqual(len(backout), 2)
        self.assertEqual(np.shape(backout[0]), (C, H, W))

class TestDropout(unittest.TestCase):

    def setUp(self):
        self.dp = layer.dropout(0.5)
        #self.dp2 = layer.dropout(1.1)   #报错 raise exception

    def assertNumpyArraysEqual(self, this, that):  # 比较np.array
        if this.shape != that.shape:
            raise AssertionError("Shapes don't match")
        if not np.allclose(this, that):
            raise AssertionError("Elements don't match!")

    def test_init(self):
        self.assertEqual(self.dp.level, 0.5)
        self.assertEqual(self.dp.module_name, "dropout")

    def test_forward_prop(self):
        input = [np.ones((3, 4, 4)), np.ones((3, 4, 4))]
        out1 = self.dp.forward_prop(input)
        self.assertEqual(len(out1), 2)
        out2 = self.dp.forward_prop(input, test = True)
        #self.assertEqual(np.sum(out1[0]), np.sum(out2[0]))    #随机

    def test_backward_compute(self):
        input = [np.ones((3, 4, 4)), np.ones((3, 4, 4))]
        out = self.dp.forward_prop(input)
        dout = [np.ones((3, 4, 4)), np.ones((3, 4, 4))]
        backout = self.dp.backward_prop(dout, 1, 1, 1)
        self.assertEqual(len(backout), 2)
        self.assertEqual(np.shape(backout[0]), (3, 4, 4))
        self.assertNumpyArraysEqual(out[0], backout[0])

if __name__ == '__main__':
   unittest.main()

