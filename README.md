Multi framework test in Cifar-10
=====

1. 使用Cifar-10数据测试多种深度学习框架<br>
2. 额外包GraphViz pydot

#### 环境说明
1. Windows 10,Anaconda3,python3.6.1,Keras,PyTorch,Tensorflow,CNTK
2. 使用CNTK时如果提示找不到DLL,请尝试删除java环境变量.

#### 实验结果
| | | CNTK-python |CNTK-CSharp| Tensorflow |PyTorch |Keras(Tensorflow)|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|1|时间(s) | 2487.314524|2359.11673| |4236.16136 |6977.23662 |
| |准确率  | 0.8996|0.8305| |0.8214 |0.8395|
|2|时间(s)| 2435.918163|2818.40215| | |6815.64579|
| |准确率 | 0.8982|0.8181| | |0.8410 |
|3|时间(s)| 2432.328233|3586.0540| | | |
| |准确率 | 0.9006|0.8410| | | |
|4|时间(s)| 2444.089096|2784.99560| | | |
| |准确率 | 0.8973|0.8229| | | |
|5|时间(s)| 2390.495292|2531.13368| | | |
| |准确率 | 0.8979|0.8171| | | |
|平|时间(s)|2438.029062 |2815.94043| | | |
|均|准确率 | 0.8987|0.8259| | | |

