import numpy as np

import l2_loss_vars

class layer(object):  # fc, conv, relu, reshape 等层的基类
	def __init__(self): 
		pass

	def initial(self, initor):
		pass

	def set_name(self, layers_name):
		if self.module_name not in l2_loss_vars.instance_name[layers_name]:
			l2_loss_vars.instance_name[layers_name][self.module_name] = 1
			self.instance_name = self.module_name + str(0)
		else:
			i = l2_loss_vars.instance_name[layers_name][self.module_name]
			self.instance_name = self.module_name + str(i)
			l2_loss_vars.instance_name[layers_name][self.module_name]+=1

	def forward_prop(self, layer_input, test = False):
		return self.forward_compute(layer_input, test = test)

	def forward_compute(self, layer_input): #前向
		try:
			pass
		except AttributeError as msg:
			print("raise error:" + str(msg))

	def backward_prop(self, gradient, lr, wd, n, beta_1 = 0.9, beta_2 = 0.999, method = "sgd"): #传递gradient
		if method == "sgd":
			return self.backward_compute(gradient, lr, wd, n)    #sgd
		elif method == "adam":
			#print("进method", np.shape(gradient))
			return self.backward_compute2(gradient, lr, wd, n, beta_1, beta_2)    #adam
		else:
			pass

	def backward_compute(self,gradient, lr, wd, n):
		pass

	def backward_compute2(self, gradient, lr, wd, n, beta_1 = 0.9, beta_2 = 0.999):
		pass

	def load_para(self, file_path, ):
		pass

	def save_para(self, file_path):
		pass

class fc(layer):

	def __init__(self, input_size, size, l2_loss, is_bias): 
		self.input_size = input_size
		self.size = size
		self.l2_loss = l2_loss
		self.is_bias = is_bias
		self.module_name =  "fc"

	def initial(self, initor):
		w = np.zeros((self.size, self.input_size))
		self.weight = initor[self.module_name]["weight"](w)
		self.ms = np.zeros_like(w)  # 一阶矩估计初始值
		self.vs = np.zeros_like(w)  # 二阶矩估计初始值
		if self.is_bias:
			b = np.zeros((self.size, 1))
			self.bias = initor[self.module_name]["bias"](b)
			self.mb = np.zeros_like(b)
			self.vb = np.zeros_like(b)
		if self.l2_loss:
			l2_loss_vars.l2_loss_vars.append(self.weight)

	def forward_compute(self, layer_input, test = False): #前向 layer_input是batchsize大小的列表
		self.input = layer_input
		self.z = []
		#print(self.instance_name, self.input)
		try:
			if self.is_bias:
				for x in layer_input:
					self.z.append(np.dot(self.weight, x)+self.bias)
				return self.z
			else:
				for x in layer_input:
					self.z.append(np.dot(self.weight, x))
				return self.z
		except AttributeError as msg:
			print("raise error:" + str(msg))

	def backward_compute(self, gradient, lr, wd, n): #传递gradient n是整个训练集的大小
		#传进gradient=delta=w(l+1)*pd_az*delta(l+1)
		#gradient (self.size,1)
		self.lmbda = wd
		nabla_w = np.zeros(self.weight.shape)
		for x, g in zip(self.input, gradient):
			nabla_w = nabla_w+np.dot(g, x.transpose())
		new_g = []
		for g in gradient:  #计算传回的梯度
			new_g.append(np.dot(self.weight.transpose(), g))
		if self.l2_loss:  #是否需要正则化
			for i in range(0, len(self.weight)):
				self.weight[i] = (1-lr*(self.lmbda/n))*self.weight[i]-(lr/len(self.input))*nabla_w[i]
		else:
			for i in range(0, len(self.weight)):
				self.weight[i] = self.weight[i]-(lr/len(self.input))*nabla_w[i]
		if self.is_bias:
			for g in gradient:
				self.bias = self.bias-(lr/len(self.input))*g
		return new_g

	def backward_compute2(self, gradient, lr, wd, n, beta_1, beta_2):  #传递gradient n是整个训练集的大小
		#传进gradient=delta=w(l+1)*pd_az*delta(l+1)
		#gradient (self.size,1)
		self.lmbda = wd
		nabla_w = np.zeros(self.weight.shape)
		epsilon = 0.00000001
		for x, g in zip(self.input, gradient):
			nabla_w = nabla_w+(1/len(self.input))*np.dot(g, x.transpose())    #adam中这里就乘上1/m
		new_g = []
		for g in gradient:  #计算传回的梯度
			new_g.append(np.dot(self.weight.transpose(), g))
		self.ms = beta_1*self.ms + (1-beta_1)*nabla_w
		self.vs = beta_2*self.vs + (1-beta_2)*(nabla_w*nabla_w)
		m_t = (1/(1-beta_1*beta_1))*self.ms
		v_t = (1/(1-beta_2*beta_2))*self.vs
		theta = m_t/(np.sqrt(v_t)+epsilon)
		if self.l2_loss:  #是否需要正则化
			for i in range(0, len(self.weight)):
				self.weight[i] = (1-lr*(self.lmbda/n))*self.weight[i]-lr*theta[i]
		else:
			for i in range(0, len(self.weight)):
				self.weight[i] = self.weight[i]-(lr)*theta[i]
		if self.is_bias:
			gradient_b = np.zeros_like(gradient[0], dtype=np.float64)
			for g in gradient:
				gradient_b += (1/len(self.input))*g
			self.mb = beta_1*self.mb + (1-beta_1)*gradient_b
			self.vb = beta_2*self.vb + (1-beta_2)*(gradient_b*gradient_b)
			m_tb = (1 / (1 - beta_1 * beta_1)) * self.mb
			v_tb = (1 / (1 - beta_2 * beta_2)) * self.vb
			theta_b = m_tb / (np.sqrt(v_tb) + epsilon)
			self.bias = self.bias-lr*theta_b
		return new_g

	def load_para(self, f):
		self.weight = f[self.instance_name+"weight"]
		self.ms = f[self.instance_name+"ms"]
		self.vs = f[self.instance_name+"vs"]
		if self.l2_loss:
			l2_loss_vars.l2_loss_vars.append(self.weight)
		if self.is_bias:
			self.bias = f[self.instance_name+"bias"]
			self.mb = f[self.instance_name+"mb"]
			self.vb = f[self.instance_name+"vb"]

	def save_para(self, file_path):
		name_w = self.instance_name + "weight"
		l2_loss_vars.dd[name_w] = self.weight
		name_ms = self.instance_name + "ms"
		l2_loss_vars.dd[name_ms] = self.ms
		name_vs = self.instance_name + "vs"
		l2_loss_vars.dd[name_vs] = self.vs
		if self.is_bias:
			name_b = self.instance_name + "bias"
			l2_loss_vars.dd[name_b] = self.bias
			name_mb = self.instance_name + "mb"
			l2_loss_vars.dd[name_mb] = self.mb
			name_vb = self.instance_name + "vb"
			l2_loss_vars.dd[name_vb] = self.vb
		np.savez(file_path, **(l2_loss_vars.dd))

class reshape(layer): 
	def __init__(self, shape):   #shape = (x,y)
		self.shape = shape
		self.module_name = "reshape"

	def forward_compute(self, layer_input, test = False): #前向
		self.input_shape = np.shape(layer_input[0])
		#print("奇怪", np.shape(layer_input[0]))
		self.z = []
		try:
			for x in layer_input:
				self.z.append(np.reshape(x, self.shape))
			return self.z
		except AttributeError as msg:
			print('raise error:' + str(msg))

	def backward_compute(self, gradient, lr, wd, n):
		new_g = []
		for g in gradient:
			new_g.append(np.reshape(g, self.input_shape))
		return new_g

	def backward_compute2(self, gradient, lr, wd, n, beta_1, beta_2):
		new_g = []
		for g in gradient:
			new_g.append(np.reshape(g, self.input_shape))
		return new_g

class relu(layer):  #激活函数relu
	def __init__(self):
		self.module_name = "relu"

	def forward_compute(self, layer_input, test = False): #前向
		self.input = layer_input
		self.activation = []
		try:
			for x in self.input:
				self.activation.append(np.maximum(x, 0))  #relu
				#if self.instance_name == "relu0":
					#print("relu0", self.activation)
			return self.activation
		except AttributeError as msg:
			print("raise error:" + str(msg))

	def backward_compute(self, gradient, lr, wd, n): #传递gradient
		new_g = []
		for p, g in zip(self.activation, gradient):
			p[p>0] = 1     #activation关于z的偏导 a>0偏导为1否则为0
			new_g.append(g*p)
		#print("back relu",self.instance_name, new_g)
		return new_g

	def backward_compute2(self, gradient, lr, wd, n, beta_1, beta_2):
		new_g = []
		for p, g in zip(self.activation, gradient):
			p[p > 0] = 1  # activation关于z的偏导 a>0偏导为1否则为0
			new_g.append(g * p)
		# print("back relu",self.instance_name, new_g)
		return new_g

class sigmoid(layer): #激活函数sigmoid
	def __init__(self):
		self.module_name = "sigmoid"
		
	def forward_compute(self, layer_input, test = False): #前向
		self.input = layer_input
		self.activation = []
		try:
			for x in self.input:
				self.activation.append(1.0/(1.0+np.exp(-x)))
			#print('sigmoid', self.activation)
			return self.activation
		except AttributeError as msg:
			print("raise error:" + str(msg))

	def backward_compute(self, gradient, lr, wd, n): #传递gradient
		new_a = []
		for a in self.activation:
			new_a.append(a*(1-a))
		new_g = []
		for p, g in zip(new_a, gradient):
			new_g.append(g*p)
		return new_g

	def backward_compute2(self, gradient, lr, wd, n, beta_1, beta_2):
		new_a = []
		for a in self.activation:
			new_a.append(a * (1 - a))
		new_g = []
		for p, g in zip(new_a, gradient):
			new_g.append(g * p)
		return new_g

class tanh(layer):  # 激活函数sigmoid
	def __init__(self):
		self.module_name = "tanh"

	def forward_compute(self, layer_input, test = False):  # 前向
		self.input = layer_input
		self.activation = []
		try:
			for x in self.input:
				self.activation.append(np.tanh(x))
			# print('sigmoid', self.activation)
			return self.activation
		except AttributeError as msg:
			print("raise error:" + str(msg))

	def backward_compute(self, gradient, lr, wd, n):  # 传递gradient
		new_a = []
		for a in self.activation:
			new_a.append(1-a*a)
		new_g = []
		for p, g in zip(new_a, gradient):
			new_g.append(g * p)
		return new_g

	def backward_compute2(self, gradient, lr, wd, n, beta_1, beta_2):  # 传递gradient
		new_a = []
		for a in self.activation:
			new_a.append(1-a*a)
		new_g = []
		for p, g in zip(new_a, gradient):
			new_g.append(g * p)
		return new_g

class convolution(layer):

	def __init__(self, stride, in_channel, kernelsize_h, kernelsize_w, kernel_num, l2_loss, is_bias, padding):
		self.stride = stride
		self.module_name = "conv"
		self.kernel_height = kernelsize_h    #卷积核 size
		self.kernel_width = kernelsize_w
		self.kernel_num = kernel_num    #卷积核个数
		self.is_bias = is_bias
		self.l2_loss = l2_loss
		self.pad = padding
		self.in_channel = in_channel   #输入数据的通道数 灰度图-1 RGB-3

	def initial(self, initor):
		w = np.zeros((self.kernel_num, self.in_channel,        #（F,C,H,W）
					  self.kernel_height, self.kernel_width))   #卷积核权重
		self.weight = initor[self.module_name]["weight"](w)
		self.ms = np.zeros_like(w)  # 一阶矩估计初始值
		self.vs = np.zeros_like(w)  # 二阶矩估计初始值
		if self.is_bias:
			b = np.zeros((self.kernel_num,))
			self.bias = initor[self.module_name]["bias"](b)
			self.mb = np.zeros_like(b)
			self.vb = np.zeros_like(b)
		if self.l2_loss:
			l2_loss_vars.l2_loss_vars.append(self.weight)

	def forward_compute(self, layer_input, test = False):  #layer_input:N个(C, H, W)的list
		self.input = layer_input  #N (C, H, W)
		#print("conv", np.shape(layer_input[0]))
		try:
			#w_flip = self.rotate180(self.weight)
			self.H = self.input[0].shape[1]    #输入的高
			self.W = self.input[0].shape[2]    #输入的宽
			self.Ho = 1 + (self.H + 2 * self.pad - self.kernel_height) // self.stride   #边缘不要了
			self.Wo = 1 + (self.W + 2 * self.pad - self.kernel_width) // self.stride
			x_pad = np.zeros((len(self.input), self.in_channel, self.H + 2 * self.pad, self.W + 2 * self.pad)) #padding之后的input
			x = np.array(self.input)
			x_pad[:, :, self.pad:self.pad + self.H, self.pad:self.pad + self.W] = x
			out = np.zeros((len(self.input), self.kernel_num, self.Ho, self.Wo))
			for f in range(self.kernel_num):
				for i in range(self.Ho):
					for j in range(self.Wo):
						# 输出是卷积核在多通道上（i, j）处的卷积结果相加
						out[:,f, i, j] = np.sum(x_pad[:,:, i*self.stride : i*self.stride+self.kernel_height,
							j*self.stride : j*self.stride+self.kernel_width] * self.weight[f, :, :, :],axis=(1, 2, 3))
				if self.is_bias:
					out[:, f, :, :] += self.bias[f]
			#outs = out.tolist()
			#print("conv weight forward", self.weight)
			return convert(out)
		except AttributeError as msg:
			print("raise error:" + str(msg))

	def backward_compute(self, gradient, lr, wd, n): #gradient：N(F,H1,W1)
		x = np.array(self.input)
		# 翻转卷积核  （F,C,H,W）
		w_flip = self.rotate180(self.weight)
		# 翻转输入  x：(N,C,H,W)
		x_flip = self.rotate180(x)
		x_pad = np.pad(x_flip, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], 'constant')
		dx, dw = np.zeros_like(x), np.zeros_like(self.weight)
		dx_pad = np.pad(dx, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], 'constant')
		# x_pad_array = np.array(x_pad)
		dout = np.array(gradient)
		for i in range(len(self.input)):  # ith image
			for f in range(self.kernel_num):  # fth filter
				for j in range(self.Ho):
					for k in range(self.Wo):
						window = x_pad[i, :, j * self.stride:self.kernel_height + j * self.stride,
								 k * self.stride:self.kernel_width + k * self.stride]
						#db[f] += dout[i, f, j, k]
						dw[f] += window * dout[i, f, j, k]
						dx_pad[i, :, j * self.stride:self.kernel_height + j * self.stride,
						k * self.stride:self.kernel_width + k * self.stride] += w_flip[f] * dout[i, f, j, k]  # 上面的式子，关键就在于+号
		new_g = convert(dx_pad[:, :, self.pad:self.pad + self.H, self.pad:self.pad + self.W])
		if self.l2_loss:  # 是否需要正则化
			self.weight[:, :, :, :] = (1 - lr * (wd / n)) * self.weight[:, :, :, :] - (lr / len(self.input)) * dw[:, :, :, :]
		else:
			self.weight[:, :, :, :] = self.weight[:, :, :, :] - (lr / len(self.input)) * dw[:, :, :, :]
		if self.is_bias:
			for g in gradient:
				self.bias = self.bias - (lr / len(self.input)) * np.sum(g, axis=(1, 2))
		# print("db dim1", np.sum(g, axis=(1,2)))
		return new_g  # N（C, H，W）

	def backward_compute2(self, gradient, lr, wd, n, beta_1, beta_2):
		epsilon = 0.00000001
		x = np.array(self.input)
		# 翻转卷积核  （F,C,H,W）
		w_flip = self.rotate180(self.weight)
		# 翻转输入  x：(N,C,H,W)
		x_flip = self.rotate180(x)
		x_pad = np.pad(x_flip, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], 'constant')
		dx, dw = np.zeros_like(x), np.zeros_like(self.weight)
		dx_pad = np.pad(dx, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], 'constant')
		# x_pad_array = np.array(x_pad)
		dout = np.array(gradient)
		for i in range(len(self.input)):  # ith image
			for f in range(self.kernel_num):  # fth filter
				for j in range(self.Ho):
					for k in range(self.Wo):
						window = x_pad[i, :, j * self.stride:self.kernel_height + j * self.stride,
								 k * self.stride:self.kernel_width + k * self.stride]
						# db[f] += dout[i, f, j, k]
						#print(np.shape(dout))
						dw[f] += (1 / len(self.input))*(window * dout[i, f, j, k])
						dx_pad[i, :, j * self.stride:self.kernel_height + j * self.stride,
						k * self.stride:self.kernel_width + k * self.stride] += w_flip[f] * dout[
							i, f, j, k]  # 上面的式子，关键就在于+号
		new_g = convert(dx_pad[:, :, self.pad:self.pad + self.H, self.pad:self.pad + self.W])
		self.ms = beta_1 * self.ms + (1 - beta_1) * dw
		self.vs = beta_2 * self.vs + (1 - beta_2) * (dw * dw)
		m_t = (1 / (1 - beta_1 * beta_1)) * self.ms
		v_t = (1 / (1 - beta_2 * beta_2)) * self.vs
		theta = m_t / (np.sqrt(v_t) + epsilon)
		if self.l2_loss:  # 是否需要正则化
			self.weight[:, :, :, :] = (1 - lr * (wd / n)) * self.weight[:, :, :, :] - (lr) * theta[:, :, :, :]
		else:
			self.weight[:, :, :, :] = self.weight[:, :, :, :] - (lr) * theta[:, :, :, :]
		if self.is_bias:
			gradient_b = np.zeros_like(self.bias, dtype=np.float64)
			for g in gradient:
				gradient_b += (1 / len(self.input)) * np.sum(g, axis=(1, 2))
			self.mb = beta_1 * self.mb + (1 - beta_1) * gradient_b
			self.vb = beta_2 * self.vb + (1 - beta_2) * (gradient_b * gradient_b)
			m_tb = (1 / (1 - beta_1 * beta_1)) * self.mb
			v_tb = (1 / (1 - beta_2 * beta_2)) * self.vb
			theta_b = m_tb / (np.sqrt(v_tb) + epsilon)
			self.bias = self.bias - lr * theta_b
		# print("db dim1", np.sum(g, axis=(1,2)))
		return new_g  # N（C, H，W）

	def load_para(self, f):
		self.weight = f[self.instance_name+"weight"]
		self.ms = f[self.instance_name + "ms"]
		self.vs = f[self.instance_name + "vs"]
		if self.l2_loss:
			l2_loss_vars.l2_loss_vars.append(self.weight)
		if self.is_bias:
			self.bias = f[self.instance_name+"bias"]
			self.mb = f[self.instance_name + "mb"]
			self.vb = f[self.instance_name + "vb"]

	def save_para(self, file_path):
		name_w = self.instance_name + "weight"
		l2_loss_vars.dd[name_w] = self.weight
		name_ms = self.instance_name + "ms"
		l2_loss_vars.dd[name_ms] = self.ms
		name_vs = self.instance_name + "vs"
		l2_loss_vars.dd[name_vs] = self.vs
		if self.is_bias:
			name_b = self.instance_name + "bias"
			l2_loss_vars.dd[name_b] = self.bias
			name_mb = self.instance_name + "mb"
			l2_loss_vars.dd[name_mb] = self.mb
			name_vb = self.instance_name + "vb"
			l2_loss_vars.dd[name_vb] = self.vb
		np.savez(file_path, **(l2_loss_vars.dd))

	def rotate180(self, a):   #将一个四维array最里面的两维进行上下左右翻转
		for i in range(np.shape(a)[0]):
			for j in range(np.shape(a)[1]):
				a[i][j] = np.fliplr(a[i][j])
				a[i][j] = np.flipud(a[i][j])
		return a

class max_pool(layer):
	def __init__(self, height, width, stride):
		self.HH = height
		self.WW = width
		self.s = stride
		self.module_name = "max_pool"

	def forward_compute(self, layer_input, test = False):    #N(C,H,W)
		self.input = layer_input
		#print("max_pool", np.shape(layer_input[0]))
		N = len(self.input)
		C, H, W = self.input[0].shape
		H_new = 1 + (H - self.HH) // self.s
		W_new = 1 + (W - self.WW) // self.s
		out = np.zeros((N, C, H_new, W_new))
		x = np.array(layer_input)
		for i in range(N):
			for j in range(C):
				for k in range(H_new):
					for l in range(W_new):
						window = x[i, j, k * self.s:self.HH + k * self.s, l * self.s:self.WW + l * self.s]
						out[i, j, k, l] = np.max(window)
		return convert(out)

	def backward_compute(self, gradient, lr, wd, n):
		N = len(self.input)
		C, H, W = self.input[0].shape
		H_new = 1 + (H - self.HH) // self.s
		W_new = 1 + (W - self.WW) // self.s
		x = np.array(self.input)
		dx = np.zeros_like(x)
		dout = np.array(gradient)
		for i in range(N):
			for j in range(C):
				for k in range(H_new):
					for l in range(W_new):
						window = x[i, j, k * self.s:self.HH + k * self.s, l * self.s:self.WW + l * self.s]
						m = np.max(window)  # 获得之前的那个值，这样下面只要windows==m就能得到相应的位置
						dx[i, j, k * self.s:self.HH + k * self.s, l * self.s:self.WW + l * self.s] = (window == m) * dout[i, j, k, l]

		return convert(dx)

	def backward_compute2(self, gradient, lr, wd, n, beta_1, beta_2):
		N = len(self.input)
		C, H, W = self.input[0].shape
		H_new = 1 + (H - self.HH) // self.s
		W_new = 1 + (W - self.WW) // self.s
		x = np.array(self.input)
		dx = np.zeros_like(x)
		dout = np.array(gradient)
		for i in range(N):
			for j in range(C):
				for k in range(H_new):
					for l in range(W_new):
						window = x[i, j, k * self.s:self.HH + k * self.s, l * self.s:self.WW + l * self.s]
						m = np.max(window)  # 获得之前的那个值，这样下面只要windows==m就能得到相应的位置
						dx[i, j, k * self.s:self.HH + k * self.s, l * self.s:self.WW + l * self.s] = (window == m) * dout[i, j, k, l]

		return convert(dx)

class avg_pool(layer):
	def __init__(self, height, width, stride):
		self.HH = height
		self.WW = width
		self.s = stride
		self.module_name = "avg_pool"

	def forward_compute(self, layer_input, test = False):    #N(C,H,W)
		self.input = layer_input
		N = len(self.input)
		C, H, W = layer_input[0].shape
		H_new = 1 + (H - self.HH) // self.s
		W_new = 1 + (W - self.WW) // self.s
		out = np.zeros((N, C, H_new, W_new))
		x = np.array(layer_input)
		for i in range(N):
			for j in range(C):
				for k in range(H_new):
					for l in range(W_new):
						window = x[i, j, k * self.s:self.HH + k * self.s, l * self.s:self.WW + l * self.s]
						out[i, j, k, l] = np.average(window)
		return convert(out)

	def backward_compute(self, gradient, lr, wd, n):
		N = len(self.input)
		C, H, W = self.input[0].shape
		H_new = 1 + (H - self.HH) // self.s
		W_new = 1 + (W - self.WW) // self.s
		x = np.array(self.input)
		dx = np.zeros_like(x)
		dout = np.array(gradient)
		for i in range(N):
			for j in range(C):
				for k in range(H_new):
					for l in range(W_new):
						window = x[i, j, k * self.s:self.HH + k * self.s, l * self.s:self.WW + l * self.s]
						m = self.HH*self.WW    #窗口大小
						dx[i, j, k * self.s:self.HH + k * self.s, l * self.s:self.WW + l * self.s] = 1/m * dout[i, j, k, l]

		return convert(dx)

	def backward_compute2(self, gradient, lr, wd, n, beta_1, beta_2):
		N = len(self.input)
		C, H, W = self.input[0].shape
		H_new = 1 + (H - self.HH) // self.s
		W_new = 1 + (W - self.WW) // self.s
		x = np.array(self.input)
		dx = np.zeros_like(x)
		dout = np.array(gradient)
		for i in range(N):
			for j in range(C):
				for k in range(H_new):
					for l in range(W_new):
						window = x[i, j, k * self.s:self.HH + k * self.s, l * self.s:self.WW + l * self.s]
						m = self.HH * self.WW  # 窗口大小
						dx[i, j, k * self.s:self.HH + k * self.s, l * self.s:self.WW + l * self.s] = 1 / m * dout[
							i, j, k, l]

		return convert(dx)

class dropout(layer):
	def __init__(self, level):
		self.module_name = "dropout"
		if level < 0. or level >= 1:  # level是概率值，必须在0~1之间
			raise Exception('Dropout level must be in interval [0, 1]')
		self.level = level    #dropout的比例

	def forward_compute(self, layer_input, test = False):    #N(C,H,W)
		self.mask = np.random.binomial(n=1, p=1-self.level, size=layer_input[0].shape)
		#二项分布生成掩码矩阵
		out = []
		if test == False:
			for x in layer_input:
				out.append(x*self.mask)
		else:
			for x in layer_input:
				out.append(x * (1-self.level))
		return out

	def backward_compute(self, gradient, lr, wd, n):
		new_g = []
		for g in gradient:
			new_g.append(self.mask*g)
		return new_g

	def backward_compute2(self, gradient, lr, wd, n, beta_1, beta_2):
		new_g = []
		for g in gradient:
			new_g.append(self.mask * g)
		return new_g

def convert(arr):  # 多维numpy array最外层转为list
	l = []
	for i in arr:
		l.append(i)
	return l



