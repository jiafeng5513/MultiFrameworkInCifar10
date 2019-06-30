Multi framework test in Cifar-10
=====
使用Cifar-10数据测试多种深度学习框架

#### A. 环境和网络模型
1. Windows 10,Anaconda3,python3.6.1,Keras,PyTorch,Tensorflow,CNTK.<br>
2. 使用CNTK时如果提示找不到DLL,请尝试删除java环境变量.<br>
3. 硬件:一块GTX1080,8GB内存,i7-6700K.<br>
4. 使用CIFAR-10数据集,其中包含50000张训练图片和10000张测试图片,我们没有使用任何的数据扩增.<br>
5. 学习率初始化为0.03,每当epoch到达80,120,160时乘以0.1,最终停止在5e-5.<br>
6. epoches=200,batch=32,使用Adam方法.<br>
7. 网络结构如图.<br>
8. 图中的ResBlockV1和ResBlockV2如图所示.<br>

#### B. 实验说明
1. 在Tensorflow实验中,采用的是1.13版本,在这个版本中已经内置了Keras的API,但是实验并未混合使用tensorflow和Keras的函数,尽管其中的很多函数是兼容的.<br>
2. 在Keras的实验中使用的并不是tensorflow.keras中的函数,而是单独安装了Keras并使用了tensorflow作为后端.
3. 由于神经网络的初始化和训练过程存在随机性,无法保证每次训练都能得到相同的结果,但正如我们的五次实验表现的那样,这些随机因素并不会带来特别大的性能波动.<br>
4. 同一个框架上的五次实验是在一台机器上连续五次进行的,而不同的框架上的实验是不同时间进行的.这导致进行不同框架上的实验时,机器硬件状态(例如温度)和软件状态(例如系统更新等后台程序)不完全相同.这会对不同框架的训练时间造成影响,因此本实验提供的训练时间具有有限的参考价值.<br>


#### C.结果
##### 1.时间和准确率

##### 2.在Tensorflow上实现了去掉batch_normalization的版本,五次实验的结果如下
batch_normalization在各个框架中的实现不完全相同,为了证明batch_normalization对于该网络的性能影响,我们使用tensorflow实现了一个不带batch_normalization的实验.<br>


#### D.结果分析
1. 速度:
2. 准确率:
3. 编码量:
4. 相同的网络在不同的框架上具备不同的性能.我们认为除了随机因素的影响之外,不同框架对于相同操作的实现有所不同是主要原因.<br>
5. 我们认为测试中的框架的性能是相近的,CNTK的训练速度更快,但是考虑到文档完善度和社区活跃度,Keras,Tensorflow和PyTorch要更胜一筹.<br>
6. 这些框架的安装都非常方便,对Linux和Windows的支持都很好.<br>

#### E.参考和引用
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. arXiv:1512.03385<br>
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py<br>
[3] https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10<br>
[4] https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Classification/ResNet/Python<br>
[5] https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py<br>

#### F.说明
实验数据仅供参考,请勿在您的出版物中使用.<br>