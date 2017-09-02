import numpy as np
import unittest

import layers
import layer
import l2_loss_vars

class TestLayers(unittest.TestCase):
    def setUp(self):
        self.layers = layers.layers("model1")
        self.ls = layers.layers("model2")
        self.initor = {
            'fc': {'weight': self.xavier_initializer(), 'bias': self.constant_initializer(0)},
            'bn': {'alpha': self.constant_initializer(1), 'beta': self.constant_initializer(0)},
            'conv': {'weight': self.gaussian_initializer(), 'bias': self.constant_initializer(0)}
        }

    def xavier_initializer(self):
        return lambda w: w + np.random.randn(np.shape(w)[0], np.shape(w)[1]) / np.sqrt(np.shape(w)[1])

    def constant_initializer(self, constant):
        return lambda x: x + constant

    def gaussian_initializer(self):
        return lambda w: w + 0.001 * np.random.randn(np.shape(w)[0], np.shape(w)[1], np.shape(w)[2], np.shape(w)[3])

    def assertNumpyArraysEqual(self, this, that):  # 比较np.array
         if this.shape != that.shape:
              raise AssertionError("Shapes don't match")
         if not np.allclose(this, that):
              raise AssertionError("Elements don't match!")

    def test_add_initmodel(self):
        ls = layers.layers("model0")
        ls.add(layer.fc(3, 4, True, True))  # add layer
        self.assertEqual(ls.num_layers, 1)
        ls.add(layer.relu())
        ls.add(layer.sigmoid())
        ls.add(layer.sigmoid())
        self.layers.add(layer.convolution(2, 3, 2, 2, 2, True, True, 1))
        self.layers.add(layer.dropout(0.5))
        self.layers.add(layer.max_pool(2, 2, 2))
        self.layers.add(layer.avg_pool(2, 2, 2))
        self.layers.add(ls)     # add layers
        self.assertEqual(self.layers.num_layers, 8)
        self.layers.init_model(self.initor)    #init_model
        self.assertEqual(np.shape(self.layers.stacked_layers[0].weight), (2, 3, 2, 2))
        self.assertNumpyArraysEqual(self.layers.stacked_layers[0].bias, np.zeros((2, )))
        self.assertEqual(np.shape(self.layers.stacked_layers[4].weight), (4, 3))
        self.assertNumpyArraysEqual(self.layers.stacked_layers[4].bias, np.zeros((4, 1)))
        self.assertEqual(np.shape(l2_loss_vars.l2_loss_vars[1]), (4, 3))
        self.assertEqual(self.layers.stacked_layers[0].instance_name, "conv0")
        self.assertEqual(self.layers.stacked_layers[1].instance_name, "dropout0")
        self.assertEqual(self.layers.stacked_layers[2].instance_name, "max_pool0")
        self.assertEqual(self.layers.stacked_layers[3].instance_name, "avg_pool0")
        self.assertEqual(self.layers.stacked_layers[4].instance_name, "fc0")
        self.assertEqual(self.layers.stacked_layers[5].instance_name, "relu0")
        self.assertEqual(self.layers.stacked_layers[6].instance_name, "sigmoid0")
        self.assertEqual(self.layers.stacked_layers[7].instance_name, "sigmoid1")

    def test_infer_save_load(self):
        t_shape = (18, 1)
        self.layers.add(layer.convolution(2, 3, 2, 2, 2, True, True, 1))
        self.layers.add(layer.max_pool(2, 2, 2))
        self.layers.add(layer.avg_pool(2, 2, 2))
        self.layers.add(layer.dropout(0.5))
        self.layers.add(layer.reshape(t_shape))
        self.layers.add(layer.fc(18, 4, True, True))
        self.layers.add(layer.relu())
        self.layers.add(layer.fc(4, 2, False, False))
        self.layers.add(layer.sigmoid())
        self.layers.init_model(self.initor)
        output = self.layers.infer([np.zeros((3, 28, 28)), np.zeros((3, 28, 28))])[0]
        #list1 = ["reshape", "fc0", "relu0", "fc1", "sigmoid0"]  # pass_name
        #self.layers.pass_name(list1)
        self.assertEqual(np.shape(output), (2, 1))
        self.layers.save("ss.npz")
        self.layers.load("ss.npz")  #文件后缀名是npz

    def test_compute_gradient(self):
        target = [np.arange(5).reshape(5, 1), np.ones((5, 1))]
        logit = [np.ones((5, 1)), np.zeros((5, 1))]
        self.layers.lmbda = 2.0
        loss, grad = self.layers.compute_gradient(logit, target)
        self.assertEqual(np.shape(grad[0]), (5, 1))
        self.assertEqual(len(grad), 2)

    def test_train(self):    #sgd
        t_shape = (18, 1)
        self.layers.add(layer.convolution(2, 3, 2, 2, 2, True, True, 1))
        self.layers.add(layer.max_pool(2, 2, 2))
        self.layers.add(layer.avg_pool(2, 2, 2))
        self.layers.add(layer.dropout(0.5))
        self.layers.add(layer.reshape(t_shape))
        self.layers.add(layer.fc(18, 4, True, True))
        self.layers.add(layer.relu())
        self.layers.add(layer.fc(4, 2, False, False))
        self.layers.add(layer.sigmoid())
        self.layers.init_model(self.initor)
        data = [np.zeros((3, 28, 28)), np.zeros((3, 28, 28))]
        target = [np.array([[0], [1]]), np.array([[1], [0]])]
        loss, logit = self.layers.train(data, target, 1, 1, 2)
        self.assertEqual(len(logit), 2)

    def test_train(self):    #adam
        t_shape = (18, 1)
        self.layers.add(layer.convolution(2, 3, 2, 2, 2, True, True, 1))
        self.layers.add(layer.max_pool(2, 2, 2))
        self.layers.add(layer.avg_pool(2, 2, 2))
        self.layers.add(layer.dropout(0.5))
        self.layers.add(layer.reshape(t_shape))
        self.layers.add(layer.fc(18, 4, True, True))
        self.layers.add(layer.relu())
        self.layers.add(layer.fc(4, 2, False, False))
        self.layers.add(layer.sigmoid())
        self.layers.init_model(self.initor)
        data = [np.zeros((3, 28, 28)), np.zeros((3, 28, 28))]
        target = [np.array([[0], [1]]), np.array([[1], [0]])]
        loss, logit = self.layers.train(data, target, 1, 1, 2)
        self.assertEqual(len(logit), 2)

if __name__ == '__main__':
   unittest.main()



