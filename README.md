# TensorflowLite-flexdelegate
## 1. Environment
- x86_64 PC + Ubuntu 18.04 (glibc2.27)
- armv7l 
## 2. Models to be tested
|No.|Model name|Note|
|:--|:--|:--|
|1|**[multi_add_flex](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/testdata)**|Tensorflow official tutorial model.|
|2|**[ENet](https://github.com/PINTO0309/TensorFlow-ENet.git)**|Lightweight semantic segmentation model.|
|3|**[Learning-to-See-Moving-Objects-in-the-Dark](https://github.com/MichaelHYJiang/Learning-to-See-Moving-Objects-in-the-Dark)**|Learning to See Moving Objects in the Dark. ICCV 2019.|

## 3. How to build Tensorflow Lite shared library with Flex Delegate enabled
### 3-1. x86_64 machine
```bash
$ cd ~
$ git clone -b v2.0.0 https://github.com/tensorflow/tensorflow.git
$ cd ~/tensorflow
$ ./configure
```
### 3-2. armv7l machine


## 4. How to generate a Tensorflow Lite model file with Flex Delegate enabled
