# NeuralNetwork-python
forward inference and backward propagation  
**Binary classification**  
Test:  
using 100 pictures of mnist dataset for training and 20 pictures for testing(only have 0&1).   
SGD: achieves 85% accuracy after 3 epochs(10 steps per epoch)  
adam: achieves 90% accuracy after 1 epoch(10 steps per epoch)

input: only python numpy array  
data load and save: npz file  
optimization：SGD and Adam  
loss function: the sum of sigmoid cross entropy and weight decay  
environment:python 3.5  

module:  
1.layer：Sigle layer，consists of base class layer,derived class fc(full connect)、reshape、relu、sigmoid、tanh、convolution、max_pool、  avg_pool、dropout,is resiponsible for forward reference、backward gradient calculating and parameters updating&initializing&saving&loading. 
2.layerTest: Test of layer module.  
3.layers: All layers of the network save here.Complete the forward reference and backward propagation of whole network by calling fuctions of sigle layer.  
4.layersTest: Test of layers module.  
5.L2_loss_vars: Common variable of module layer and layers.  
6.nn: Package of layer and layers.Users use the framework by importing this module.  
7.conv call nn: Test of module nn.Chose SGD optimization with input 0,adam optimization with 1  

user tips:  
1.preparation: download mnist dataset and transfer to numpy array(code in 'dispose Data')  
2.modify the size of data set and model to achieve higher accuracy  
3.can add softmax layer in module layer to dispose multiple classification problem  
4.more problem:woskii@126.com
  



