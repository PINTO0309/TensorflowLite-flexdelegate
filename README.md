# TensorflowLite-flexdelegate

November 22, 2019, under construction.  

## 1. Environment
- Ubuntu 18.04 (glibc2.27) + x86_64 PC 
- Raspbian Buster (glibc2.28) + armv7l RaspberryPi3/4
- Tensorflow v2.0.0
- **[Bazel 0.26.1](https://github.com/PINTO0309/Bazel_bin.git)**
- **[Bazel-Remote](https://github.com/buchgr/bazel-remote.git)**
- opnejdk-8-jdk (**[for Raspbian Buster - openjdk-8-jdk](https://qiita.com/PINTO/items/612718c0ce4f1def6c6e)**)
- **[OpenCV 4.1.2-openvino](https://github.com/PINTO0309/OpenVINO-bin.git)**

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
$ cd ~
$ sudo nano /etc/dphys-swapfile
CONF_SWAPFILE=2048
CONF_MAXSWAP=2048

$ sudo systemctl stop dphys-swapfile
$ sudo systemctl start dphys-swapfile

$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/zram.sh
$ chmod 755 zram.sh
$ sudo mv zram.sh /etc/init.d/
$ sudo update-rc.d zram.sh defaults
$ sudo reboot

$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev libatlas-base-dev libopenblas-dev openjdk-8-jdk
$ sudo pip3 install keras_applications==1.0.8 --no-deps
$ sudo pip3 install keras_preprocessing==1.1.0 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo -H pip3 install -U --user six numpy wheel mock

$ git clone -b v2.0.0 https://github.com/tensorflow/tensorflow.git
$ cd ~/tensorflow
$ ./configure

WARNING: ignoring LD_PRELOAD in environment.
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.26.1- (@non-git) installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3

Found possible Python library paths:
  /usr/local/lib
  /usr/local/lib/python3.6/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python3.6/dist-packages]
/usr/local/lib/python3.6/dist-packages
Do you wish to build TensorFlow with XLA JIT support? [Y/n]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: n
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
    --config=mkl            # Build with MKL support.
    --config=monolithic     # Config for mostly static monolithic build.
    --config=gdr            # Build with GDR support.
    --config=verbs          # Build with libverbs support.
    --config=ngraph         # Build with Intel nGraph support.
    --config=numa           # Build with NUMA support.
    --config=dynamic_kernels    # (Experimental) Build kernels into separate shared objects.
    --config=v2             # Build TensorFlow 2.x instead of 1.x.
Preconfigured Bazel build configs to DISABLE default on features:
    --config=noaws          # Disable AWS S3 filesystem support.
    --config=nogcp          # Disable GCP support.
    --config=nohdfs         # Disable HDFS support.
    --config=noignite       # Disable Apache Ignite support.
    --config=nokafka        # Disable Apache Kafka support.
    --config=nonccl         # Disable NVIDIA NCCL support.
Configuration finished
```
```bash
$ cd tensorflow/lite
$ nano BUILD
```
```bzl
tflite_cc_shared_object(
    name = "libtensorflowlite.so",
    linkopts = select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite:tflite_exported_symbols.lds)",
            "-Wl,-install_name,@rpath/libtensorflowlite.so",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //tensorflow/lite:tflite_version_script.lds)",
        ],
    }),
    deps = [
        ":framework",
        ":tflite_exported_symbols.lds",
        ":tflite_version_script.lds",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/delegates/flex:delegate",
    ],
)
```
```bash
$ nano tools/make/Makefile
```
```python
BUILD_WITH_NNAPI=true
↓
BUILD_WITH_NNAPI=false
```
```bash
$ sudo bazel build \
--config=monolithic \
--config=noaws \
--config=nohdfs \
--config=noignite \
--config=nokafka \
--config=nonccl \
--config=v2 \
--define=tflite_convert_with_select_tf_ops=true \
--define=with_select_tf_ops=true \
//tensorflow/lite:libtensorflowlite.so
```
```bash
$ sudo chmod 777 libtensorflowlite.so
```
### 3-2. armv7l machine
```bash
$ cd ~
$ sudo nano /etc/dphys-swapfile
CONF_SWAPFILE=2048
CONF_MAXSWAP=2048

$ sudo systemctl stop dphys-swapfile
$ sudo systemctl start dphys-swapfile

$ wget https://github.com/PINTO0309/Tensorflow-bin/raw/master/zram.sh
$ chmod 755 zram.sh
$ sudo mv zram.sh /etc/init.d/
$ sudo update-rc.d zram.sh defaults
$ sudo reboot

$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev libatlas-base-dev libopenblas-dev
$ sudo pip3 install keras_applications==1.0.8 --no-deps
$ sudo pip3 install keras_preprocessing==1.1.0 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo -H pip3 install -U --user six numpy wheel mock

$ cd ~
$ git clone https://github.com/PINTO0309/Bazel_bin.git
$ cd Bazel_bin
$ ./0.26.1/Raspbian_Debian_Buster_armhf/openjdk-8-jdk/install.sh

$ git clone -b v2.0.0 https://github.com/tensorflow/tensorflow.git
$ cd ~/tensorflow
$ ./configure

WARNING: ignoring LD_PRELOAD in environment.
WARNING: --batch mode is deprecated. Please instead explicitly shut down your Bazel server using the command "bazel shutdown".
You have bazel 0.26.1- (@non-git) installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3

Found possible Python library paths:
  /usr/local/lib
  /usr/local/lib/python3.7/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python3.7/dist-packages]
/usr/local/lib/python3.7/dist-packages
Do you wish to build TensorFlow with XLA JIT support? [Y/n]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: n
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
    --config=mkl            # Build with MKL support.
    --config=monolithic     # Config for mostly static monolithic build.
    --config=gdr            # Build with GDR support.
    --config=verbs          # Build with libverbs support.
    --config=ngraph         # Build with Intel nGraph support.
    --config=numa           # Build with NUMA support.
    --config=dynamic_kernels    # (Experimental) Build kernels into separate shared objects.
    --config=v2             # Build TensorFlow 2.x instead of 1.x.
Preconfigured Bazel build configs to DISABLE default on features:
    --config=noaws          # Disable AWS S3 filesystem support.
    --config=nogcp          # Disable GCP support.
    --config=nohdfs         # Disable HDFS support.
    --config=noignite       # Disable Apache Ignite support.
    --config=nokafka        # Disable Apache Kafka support.
    --config=nonccl         # Disable NVIDIA NCCL support.
Configuration finished
```
```bash
$ cd tensorflow/lite
$ nano BUILD
```
```bzl
tflite_cc_shared_object(
    name = "libtensorflowlite.so",
    linkopts = select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location //tensorflow/lite:tflite_exported_symbols.lds)",
            "-Wl,-install_name,@rpath/libtensorflowlite.so",
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //tensorflow/lite:tflite_version_script.lds)",
        ],
    }),
    deps = [
        ":framework",
        ":tflite_exported_symbols.lds",
        ":tflite_version_script.lds",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/delegates/flex:delegate",
    ],
)
```
```bash
$ nano tools/make/Makefile
```
```python
BUILD_WITH_NNAPI=true
↓
BUILD_WITH_NNAPI=false
```
```bash
$ nano experimental/ruy/pack_arm.cc
```
```bash
### Line 1292
"mov r0, 0\n"
  ↓
"mov r0, #0\n"
```
```bash
$ sudo bazel --host_jvm_args=-Xmx512m build \
--config=monolithic \
--config=noaws \
--config=nohdfs \
--config=noignite \
--config=nokafka \
--config=nonccl \
--config=v2 \
--define=tflite_convert_with_select_tf_ops=true \
--define=with_select_tf_ops=true \
--local_resources=4096.0,3.0,1.0 \
--copt=-mfpu=neon-vfpv4 \
--copt=-ftree-vectorize \
--copt=-funsafe-math-optimizations \
--copt=-ftree-loop-vectorize \
--copt=-fomit-frame-pointer \
--copt=-DRASPBERRY_PI \
--host_copt=-DRASPBERRY_PI \
--remote_http_cache=http://192.168.0.xx:9080 \
//tensorflow/lite:libtensorflowlite.so
```
```bash
$ sudo chmod 777 libtensorflowlite.so
```

## 4. How to generate a Tensorflow Lite model file with Flex Delegate enabled
