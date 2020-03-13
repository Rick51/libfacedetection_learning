# libfacedetection_learning
近段时间，为了将自己的模型适配到于老师的[libfacedetection](https://github.com/ShiqiYu/libfacedetection)框架上，就阅读了下老师的源码，受益良多。并且因为自己的模型有些特殊层，所以添加了些实现，比如depthwise conv, average pooling, prelu 等等，记录下来，一起学习。

How to use:

cmake .

make

python jpg2bin.py face2.jpg （因为main.cpp里面的输入是bin文件而不是利用opencv库读取图片，所以先把jpg转bin）

./facedetect_demo 198 166 (按顺序输入图片的宽，高)



未完待续～

