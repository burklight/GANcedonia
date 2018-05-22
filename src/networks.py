import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import database as db
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import itertools
import random
import time
import yaml
import os
from time import gmtime, strftime
from torchvision.utils import save_image
import csv
import shutil

## DISCRIMINATOR

class GANdiscriminator(nn.Module):
	''' This class implements a PatchGAN discriminator for a 100x100 image.
		Small modification of the one used in:
			- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
			  Jun-Yan Zhu, 2017'''

	def __init__(self, n_image_channels = 3):
		super(GANdiscriminator, self).__init__()

		def createLayer(n_filt_in, n_filt_out, ker_size, stride, norm = True, last = False):
			''' This function creates the differnt convolutional layers, all with same structure'''
			layers = [nn.Conv2d(n_filt_in, n_filt_out, ker_size, stride=stride)]
			if (norm):
				layers.append(nn.InstanceNorm2d(n_filt_out)) # batch normalization
			if (last):
				layers.append(nn.Sigmoid()) # we output the probability
			else:
				layers.append(nn.LeakyReLU(negative_slope = 0.05, inplace=True)) # we use Leaky ReLU
			return layers


		''' Input number of filters: Image channels
			Intermediate number of filters: 64*h, with h being the depth of the layer
			Output number of filters: 1 -> Decision of true or false
			It takes patches of 61x61 pixels'''
		layers = []
		n_layers = 5
		ker_size = 5
		strides = [1, 1, 1, 2, 2]
		n_filters = [n_image_channels, 64, 128, 256, 512, 1]
		lasts = [False, False, False, False, True]
		for i in range(n_layers): # For each layer
			layers.extend(createLayer(n_filters[i], n_filters[i+1], ker_size, strides[i], last = lasts[i]))

		self.model = nn.Sequential(*layers)


	def forward(self, image):
		return self.model(image)



class residual_block(nn.Module):
	''' This class implements the residual block of the RES net we will implement as the generator'''
	def __init__(self, n_channels):
		super(residual_block, self).__init__()

		layers = [
				  nn.ReflectionPad2d(1), # mirroring of 1 for the 3 kernel size convolution
				  nn.Conv2d(n_channels, n_channels, 3), # the convolution :)
				  nn.InstanceNorm2d(n_channels), # batch normalization
				  nn.LeakyReLU(negative_slope=0.05, inplace=True),
				  # We repeat the process
				  nn.ReflectionPad2d(1),
				  nn.Conv2d(n_channels, n_channels, 3),
				  nn.InstanceNorm2d(n_channels)
				 ]

		self.conv_block = nn.Sequential(*layers)

	def forward(self, image):
		return image + self.conv_block(image)


## GENERATOR

class GANgenerator(nn.Module):
	''' This class implements a RES Net for generating the images
		Small modification of the one defined in:
			- Deep Residual Learning for Image Recognition
			  Kaiming He, 2015'''

	def __init__(self, n_image_channels = 3, n_res_blocks = 9):
		super(GANgenerator, self).__init__()

		''' High kernel convolution '''
		n_channels_high = 64
		layers = [ nn.ReflectionPad2d(3), # mirroring of 3 for the 7 kernel size convolution
					nn.Conv2d(n_image_channels, n_channels_high, 7), # 64 new channels of 7x7 convolution :)
					nn.InstanceNorm2d(n_channels_high),
					nn.LeakyReLU(negative_slope=0.05, inplace=True)
				  ]

		''' Variables for down and up sampling '''
		n_layers = 2
		ker_size = 3
		strides = 2
		paddings = 1
		n_filters = [n_channels_high, n_channels_high*2, n_channels_high*4]

		''' Downsampling steps '''
		for i in range(n_layers): # for each layer
			layers.extend([ nn.Conv2d(n_filters[i], n_filters[i+1], ker_size,
							strides, padding=paddings),
							nn.InstanceNorm2d(n_filters[i+1]),
							nn.LeakyReLU(negative_slope=0.05, inplace=True)])

		''' Residual blocks '''
		for i in range(n_res_blocks):
			layers.extend([residual_block(n_filters[-1])]) # the residual blocks are applied to the
														   # last number of channels in the down sampling

		''' Upsampling steps '''
		for i in range(n_layers): # for each layer
			layers.extend([ nn.ConvTranspose2d(n_filters[-(i+1)], n_filters[-(i+2)], ker_size,
							strides, padding=paddings, output_padding=1),
							nn.InstanceNorm2d(n_filters[-(i+2)]),
							nn.LeakyReLU(negative_slope=0.05, inplace=True)])
		''' Output '''
		layers.extend([ nn.ReflectionPad2d(3), # mirroring of 3 for the 7 kernel size convolution
						nn.Conv2d(n_channels_high, n_image_channels, 7), # 64 new channels of 7x7 convolution :)
						nn.Tanh() ])

		self.res_net = nn.Sequential(*layers)

	def forward(self, image):
		return self.res_net(image)
