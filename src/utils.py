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

# Initialize weights function
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.kaiming_normal(m.weight.data, a=0.05)
	elif classname.find('BatchNorm2d') != -1:
		torch.nn.init.kaiming_normal(m.weight.data, a=0.05)
		torch.nn.init.constant_(m.bias.data, 0.0)

# Learning rate update class
class lr_update():
	def __init__(self, n_epochs, epoch, start_decay_epoch):
		assert ((n_epochs - start_decay_epoch) > 0), "You can't decay after finishing"
		self.n_epochs = n_epochs
		self.epoch = epoch
		self.start_decay_epoch = start_decay_epoch

	def decay(self, epoch):
		return 1.0 - max(0, epoch + self.epoch - self.start_decay_epoch) \
			/ (self.n_epochs - self.start_decay_epoch)

# class to store the images
class imageBuffer():
	def __init__(self, max_size = 50):
		assert (max_size > 0), 'You need to be able to store something'
		self.max_size = max_size
		self.data = []

	def push_and_pop(self, data):
		result = []
		for element in data.data:
			element = torch.unsqueeze(element, 0)
			# If we can store data
			if len(self.data) < self.max_size:
				self.data.append(element)
				result.append(element)
			# Else change a random element of the data with probability 0.5
			else:
				if random.uniform(0,1) > 0.5:
					pos = random.randint(0, self.max_size-1)
					result.append(self.data[pos].clone())
					self.data[pos] = element
				else:
					result.append(element)
		# Return the result as a torch variable
		return Variable(torch.cat(result))
