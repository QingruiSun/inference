#!/bin/bash

## INSTALL cuDnn & TensorRT
cd /root/debs
echo 'Installing cuDnn...'
dpkg -i libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb
dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb
echo 'Installing TensorRT...'
dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb
apt update
apt install tensorrt libnvinfer7

## INSTALL CUSTOM TF
cd /root
pip install /root/tensorflow_pkg/tensorflow-1.15.5-cp37-cp37m-linux_x86_64.whl
