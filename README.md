# libfacedetection_learning

近段时间，为了将自己的模型适配到于老师的[libfacedetection](https://github.com/ShiqiYu/libfacedetection)框架上，就阅读了下老师的源码，受益良多。并且因为自己的模型有些特殊层，所以添加了些实现，比如depthwise conv, average pooling, prelu 等等，记录下来，一起学习。

## 1. Processing

#### raw_Input：

![image](https://github.com/Rick51/libfacedetection_learning/blob/master/images/forever2.jpg)

#### padding：

![image](https://github.com/Rick51/libfacedetection_learning/blob/master/images/padding.png)

#### scaleing:(padding黑边的基础上放缩到320X240):

![image](https://github.com/Rick51/libfacedetection_learning/blob/master/images/scaleing.png)

#### output:

face[0]: x1: 225, y1: 43, x2: 304, y2: 122

face[1]: x1: 268, y1: 387, x2: 332, y2: 451

face[2]: x1: 364, y1: 281, x2: 447, y2: 364

face[3]: x1: 218, y1: 293, x2: 292, y2: 368

face[4]: x1: 141, y1: 37, x2: 215, y2: 112



## 2. How to use:

cmake .

make

python jpg2bin.py  images/forever2.jpg （因为main.cpp里面的输入是bin文件而不是利用opencv库读取图片，所以先把jpg转bin）

./facedetect_demo 500 500 (按顺序输入图片的实际宽，高)



## 3. blog记录

[关于libfacedetection的一些思考](https://blog.csdn.net/weixin_42616332/article/details/103171233)
