import numpy as np

import l2_loss_vars
import layer

class layers(object):

	def __init__(self, name):
		self.num_layers = 0
		self.stacked_layers = []  #实例属性
		self.instance_name = name
		l2_loss_vars.instance_name[self.instance_name] = {}

	def init_model(self, initor):
		for l in self.stacked_layers:
			l.initial(initor)

	def add(self,layer_s):   #layers or layer
		if isinstance(layer_s, layer.layer):  #判断是layer
			self.stacked_layers.append(layer_s)
			layer_s.set_name(self.instance_name)  #给单层layer赋值实体名称
			self.num_layers+=1
		else:
			self.num_layers = self.num_layers + len(layer_s.stacked_layers)
			self.stacked_layers.extend(layer_s.stacked_layers)
			for l in layer_s.stacked_layers:
				l.set_name(self.instance_name)

	'''
	def pass_name(self, name_list):  #如果从npz文件读参数需要调用此函数
		for i in range(0, self.num_layers):
			self.stacked_layers[i].set_name(name_list[i])
	'''

	def infer(self, input, test = False):  #模型参数已加载或初始化的情况下,接收numpy.array,输出 logit 结果
		#input是batchsize长度的列表
		activation = input
		for layer in self.stacked_layers:
			activation = layer.forward_prop(activation, test = test)
		return activation

	def train(self, data, target, lr, wd, n, beta_1 = 0.9, beta_2 = 0.999, method = "sgd"):  #mini_batch的data和target
		#接收data和target的numpy.array,进行一次参数更新 n代表整个训练集的size
		self.lmbda = wd  #供loss使用
		net_list = []
		net = data
		for layer in self.stacked_layers:
			net_list.append(net)
			net = layer.forward_prop(net)
			#print(layer.instance_name)
		loss, grads = self.compute_gradient(net, target)
		for l in range(2, self.num_layers):
			grads = self.stacked_layers[-l].backward_prop(grads, lr, wd, n, beta_1 = beta_1, beta_2 = beta_2, method = method)
		return (loss, net)

	def load(self, filepath):
		f = np.load(filepath)
		for l in self.stacked_layers:
			l.load_para(f)

	def save(self, file_path):
		for l in self.stacked_layers:
			l.save_para(file_path)

	def compute_gradient(self, net, target):  # net就是logit
		# loss 部分仅考虑 sigmoid_cross_entropy 与 weight_decay(L2 范数)的和值形式
		self.l2_loss_vars = l2_loss_vars.l2_loss_vars
		cost = 0.0
		gradient = []
		for a, y in zip(net, target):
			gradient.append(a - y)  # 交叉熵 C关于a的偏导
			cost += self.fn(a, y) / len(target)
		cost += 0.5 * (self.lmbda / len(target)) * sum(
			np.linalg.norm(w) ** 2 for w in self.l2_loss_vars)
		return (cost, gradient)

	def fn(self, a, y):  #CrossEntropyCost
		#print("aaa", np.nan_to_num(np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))))
		return np.nan_to_num(np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))))


