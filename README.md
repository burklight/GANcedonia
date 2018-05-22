# GANcedonia

In this repository you will find a cycle consistency GAN prepared to perform object transfiguration between different fruits. 
This PyTorch implementation is based on the [CycleGAN original paper](https://arxiv.org/pdf/1703.10593.pdf) from [Jun-Yan Zhu](https://github.com/junyanz), [Taesung Park](https://github.com/taesung89) et al. 
The main differences with the original paper are:
- the usage of a different generator network (we use a residual network based on the good results from [Leidig et al.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf)). The [latest implementation](https://github.com/junyanz/CycleGAN) of the original authors also considers residual networks and U-nets.
- the usage of odd weighted filters for the PatchGAN.

![alt text](https://github.com/burklight/GANcedonia/blob/master/images/example_gancedonia.png)

## Dataset

The dataset used for this repository is the [Kaggle Fruits 360](https://www.kaggle.com/moltean/fruits) presented by [Muresan and Oltean](https://arxiv.org/pdf/1712.00580.pdf).
The images from this dataset are placed randomly over the textures in the [textures folder](https://github.com/burklight/GANcedonia/tree/master/Dataset/textures) with random scaling.

## Prerequisites

- PyTorch 0.3.0 or superior.
- Python2 or Python3
- If using an NVIDIA GPU: CUDA 

### Installation (Ubuntu)

- Installation of CUDA

```
wget developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

- Installation of Pytorch 3.6 for Python 3.6

```
sudo -H pip3 install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl 
sudo -H pip3 install torchvision
```
- Installation of Pytorch 3.6 for Python 3.5

```
sudo -H pip3 install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp35m-linux_x86_64.whl 
sudo -H pip3 install torchvision
```
