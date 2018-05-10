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

# Import parameters
with open("parameters.yml", 'r') as ymlfile:
	param = yaml.load(ymlfile)


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
			layers.extend([ nn.Conv2d(n_filters[i], n_filters[i+1], ker_size,                                     strides, padding=paddings),
							nn.InstanceNorm2d(n_filters[i+1]),
							nn.LeakyReLU(negative_slope=0.05, inplace=True)])

		''' Residual blocks '''
		for i in range(n_res_blocks):
			layers.extend([residual_block(n_filters[-1])]) # the residual blocks are applied to the
														   # last number of channels in the down sampling

		''' Upsampling steps '''
		for i in range(n_layers): # for each layer
			layers.extend([ nn.ConvTranspose2d(n_filters[-(i+1)], n_filters[-(i+2)], ker_size,                                     strides, padding=paddings, output_padding=1),
							nn.InstanceNorm2d(n_filters[-(i+2)]),
							nn.LeakyReLU(negative_slope=0.05, inplace=True)])
		''' Output '''
		layers.extend([ nn.ReflectionPad2d(3), # mirroring of 3 for the 7 kernel size convolution
						nn.Conv2d(n_channels_high, n_image_channels, 7), # 64 new channels of 7x7 convolution :)
						nn.Tanh() ])

		self.res_net = nn.Sequential(*layers)

	def forward(self, image):
		return self.res_net(image)

#################
#	TRAINING	#
#################

# Losses
criterion_gan = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Prepare for using cuda or not
cuda = True if torch.cuda.is_available() else False # for using the GPU if possible

# Calculate the number of patches (61x61) separated 25
batch_size = param['train']['batch_size']
img_x = param['input']['img_x']
img_y = param['input']['img_y']
separation = param['train']['patch_sep']
patch_x, patch_y = param['train']['patch_x'], param['train']['patch_y']
patch = (batch_size, 1 , patch_x, patch_y)

# Generation of the discriminator and generator
D_A = GANdiscriminator()
D_B = GANdiscriminator()
G_AB = GANgenerator()
G_BA = GANgenerator()
if cuda:
	D_A = D_A.cuda()
	D_B = D_B.cuda()
	G_AB = G_AB.cuda()
	G_BA = G_BA.cuda()
	criterion_gan.cuda()
	criterion_cycle.cuda()
	criterion_identity.cuda()


# Initialize weights
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.kaiming_normal(m.weight.data, a=0.05)
	elif classname.find('BatchNorm2d') != -1:
		torch.nn.init.kaiming_normal(m.weight.data, a=0.05)
		torch.nn.init.constant_(m.bias.data, 0.0)

if param['log']['save_path']=='auto':
	filepath=os.path.join('Log', strftime("%Y%m%d_%H%M", gmtime()))
else:
	filepath=os.path.join('Log',(param['log']['save_path']+'_'+param['log']['save_path_folder_flag']))

# Create folders if they do not exist

if not os.path.exists(os.path.join(filepath,'G_AB')):
	os.makedirs(os.path.join(filepath,'G_AB'))

if not os.path.exists(os.path.join(filepath,'G_BA')):
	os.makedirs(os.path.join(filepath,'G_BA'))

if not os.path.exists(os.path.join(filepath,'D_A')):
	os.makedirs(os.path.join(filepath,'D_A'))

if not os.path.exists(os.path.join(filepath,'D_B')):
	os.makedirs(os.path.join(filepath,'D_B'))

if not os.path.exists(os.path.join(filepath,'images')):
	os.makedirs(os.path.join(filepath,'images'))


if param['load']['load_weights']:
	G_AB_state = torch.load(os.path.join(filepath,'G_AB','epoch_'+str(param['load']['load_epoch'])+'.pkl'))
	G_AB.load_state_dict(G_AB_state['state_dict'])
	G_BA_state = torch.load(os.path.join(filepath,'G_BA','epoch_'+str(param['load']['load_epoch'])+'.pkl'))
	G_BA.load_state_dict(G_BA_state['state_dict'])
	D_A_state = torch.load(os.path.join(filepath,'D_A','epoch_'+str(param['load']['load_epoch'])+'.pkl'))
	D_A.load_state_dict(D_Astate['state_dict'])
	D_B_state = torch.load(os.path.join(filepath,'D_B','epoch_'+str(param['load']['load_epoch'])+'.pkl'))
	D_B.load_state_dict(Dstate['state_dict'])
else:
	G_AB.apply(weights_init); # He initialization of the weights
	G_BA.apply(weights_init); # He initialization of the weights
	D_A.apply(weights_init); # He initialization of the weights
	D_B.apply(weights_init); # He initialization of the weights


# Define the optimizer for the generator
lr = float(param['train']['learning_rate'])
beta_1 = param['train']['beta_1']
beta_2 = param['train']['beta_2']
gan_params = list(G_AB.parameters()) + list(G_BA.parameters())
optimizer_G = torch.optim.Adam(gan_params, lr = lr, betas=(beta_1, beta_2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(beta_1, beta_2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(beta_1, beta_2))


class lr_update():
	def __init__(self, n_epochs, epoch, start_decay_epoch):
		assert ((n_epochs - start_decay_epoch) > 0), "You can't decay after finishing"
		self.n_epochs = n_epochs
		self.epoch = epoch
		self.start_decay_epoch = start_decay_epoch

	def decay(self, epoch):
		return 1.0 - max(0, epoch + self.epoch - self.start_decay_epoch) \
			/ (self.n_epochs - self.start_decay_epoch)

# Scheduler for optimization
n_epochs = param['train']['n_epochs']
epoch = 0
start_decay_epoch = param['decay']['epoch_start']

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, \
	lr_lambda=lr_update(n_epochs, epoch, start_decay_epoch).decay)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, \
	lr_lambda=lr_update(n_epochs, epoch, start_decay_epoch).decay)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, \
	lr_lambda=lr_update(n_epochs, epoch, start_decay_epoch).decay)

# Structure for using cuda tensors or just cpu tensors
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


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

# Buffers of previously generated samples
fake_A_buffer = imageBuffer()
fake_B_buffer = imageBuffer()


# Prepare the data
fruits_file = 'Dataset/dataset_index.csv'
textures_file = 'Dataset/textures_index.csv'
textures = db.TexturesDataset(csv_file=textures_file)
imgs_db = db.FruitsDataset(csv_file=fruits_file, cl_A=param['input']['fruit_1'], cl_B=param['input']['fruit_2'],
						transform = transforms.Compose([db.ChangeBackground(textures),db.myReshape()]))

dataloader = DataLoader(imgs_db, batch_size=batch_size,
						shuffle=True, num_workers=4, drop_last=True)

val_dataloader = DataLoader(imgs_db, batch_size=5,
						shuffle=True, num_workers=1, drop_last=True)

def sample_images(folder, epoch):
	''' Saves a generated sample from the validation set'''
	imgs = next(iter(val_dataloader))
	#sample_batched[0][j]
	img_A = imgs[0]
	img_B = imgs[1]
	real_A = Variable(img_A.type(Tensor))
	fake_B = G_AB(real_A)
	real_B = Variable(img_B.type(Tensor))
	fake_A = G_BA(real_B)
	recov_A = G_BA(fake_B)
	recov_B = G_AB(fake_A)
	## Gudardar-les després és cosa d'en Marcel
	img_sample = torch.cat((real_A.data+0.5, fake_B.data+0.5, recov_A.data+0.5, \
		real_B.data+0.5, fake_A.data+0.5, recov_B.data+0.5), 0)
	save_image(img_sample, folder+'/images/'+str(epoch)+'.png', nrow=5, normalize=True)


# Initialize losses csv
losses_log='losses.csv'
losses_filepath=os.path.join(filepath, losses_log)

with open(losses_filepath, "a") as file:
	csv_header=['Epoch', 'Timestamp', 'loss_identity_A', 'loss_identity_B', 'loss_identity','loss_gan_AB', 'loss_gan_BA', 'loss_gan', 'loss_cycle_A','loss_cycle_B','loss_cycle','loss_G', 'loss_real_A', 'loss_real_B', 'loss_fake_A','loss_fake_B','loss_D_A','loss_D_B','loss_D']
	writer = csv.writer(losses_filepath, delimiter=',')
	writer.writerow(csv_header)

def log_losses(losses_list, losses_filepath):
	for i in range(len(losses_list)):
		if (isinstance(lossed_list[i], float)):
			losses_list[i] = ("%.4f" % i)
	with open(losses_filepath, "a") as file:
	    writer = csv.writer(file, delimiter=',')
	    writer.writerow(losses_list)


valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)


# Training phase

alpha_cycle = param['train']['alpha_cycle']
alpha_identy = param['train']['alpha_identity']

if param['load']['load_weights']:
	previous_epochs = G_ABstate['epoch']
else:
	previous_epochs = 0


for epoch in range(n_epochs):
	start_time = time.time()
	print("##########################################################")
	print("- EPOCH: " + str(epoch)) + "  -  " + time.strftime("%Y%m%d-%H:%M", start_time)
	print("##########################################################")
	for i, batch in enumerate(dataloader):

		# Get model input
		real_A = Variable(batch[0].type(Tensor))
		real_B = Variable(batch[1].type(Tensor))

		# ----------- #
		#  Generator  #
		# ----------- #

		# Set the optimizer
		optimizer_G.zero_grad()

		# Set the identity loss
		loss_identity_A = criterion_identity(G_BA(real_A), real_A)
		loss_identity_B = criterion_identity(G_AB(real_B), real_B)
		loss_identity = 0.5*(loss_identity_A + loss_identity_B)

		# Generate two images
		fake_B = G_AB(real_A)
		fake_A = G_BA(real_B)

		# Set GAN loss
		loss_gan_AB = criterion_gan(D_B(fake_B), valid) # the discriminator finds B real enough
		loss_gan_BA = criterion_gan(D_A(fake_A), valid) # the discriminarot finds A real enough
		loss_gan = 0.5*(loss_gan_AB + loss_gan_BA)

		# "recover" the two images
		recovered_A = G_BA(fake_B)
		recovered_B = G_AB(fake_A)

		# Set cycle loss
		loss_cycle_A = criterion_cycle(recovered_A, real_A)
		loss_cycle_B = criterion_cycle(recovered_B, real_B)
		loss_cycle = 0.5*(loss_cycle_A + loss_cycle_B)

		# Total loss
		loss_G = loss_gan + alpha_cycle * loss_cycle_A + alpha_identy * loss_identity

		# Backpropagate the gradient of the loss
		loss_G.backward()
		optimizer_G.step()

		# -------------- #
		#  Discriminator #
		# -------------- #

		# Set the optimizer
		optimizer_D_A.zero_grad()
		optimizer_D_B.zero_grad()

		# Get the real loss
		loss_real_A = criterion_gan(D_A(real_A), valid)
		loss_real_B = criterion_gan(D_B(real_B), valid)

		# Fake loss (on previously generated samples)
		fake_A_prev = fake_A_buffer.push_and_pop(fake_A)
		fake_B_prev = fake_B_buffer.push_and_pop(fake_B)
		loss_fake_A = criterion_gan(D_A(fake_A_prev.detach()), fake)
		loss_fake_B = criterion_gan(D_B(fake_B_prev.detach()), fake)

		# Total losses
		loss_D_A = 0.5*(loss_real_A + loss_fake_A)
		loss_D_B = 0.5*(loss_real_B + loss_fake_B)

		# Backpropagate the gradient of the losses
		loss_D_A.backward()
		optimizer_D_A.step()
		loss_D_B.backward()
		optimizer_D_B.step()

		# Final total loss for discriminator (only for plotting purposes)
		loss_D = 0.5*(loss_D_A + loss_D_B)

		# Saving stuff and shit... (marcel ^.^)

		if i % param['log']['print_batch_interval'] == 0:
			print("Image #: " + str(i*batch_size))

	# Update learning rates
	lr_scheduler_G.step()
	lr_scheduler_D_A.step()
	lr_scheduler_D_B.step()

	elapsed_time_epoch = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

	if param['log']['save_losses']:
		log_losses([epoch, elapsed_time_epoch, loss_identity_A, loss_identity_B, loss_identity, loss_gan_AB, loss_gan_BA, loss_gan, loss_cycle_A, loss_cycle_B, loss_cycle, loss_G, loss_real_A, loss_real_B, loss_fake_A, loss_fake_B, loss_D_A, loss_D_B, loss_D], losses_filepath)

	# Saving models... (agin marcel ^.^)
	if param['log']['save_weights']:
		if epoch % param['log']['save_weight_interval'] == 0:
		# Saving Generator
			state_G_AB = {
				'epoch': epoch+previous_epochs,
				'state_dict': G_AB.state_dict(),
				'optimizer': optimizer_G.state_dict()

			}

			torch.save(state_G_AB, os.path.join(filepath,'G_AB','epoch_' + str(epoch+1) + '.pkl'))

			state_G_BA = {
				'epoch': epoch+previous_epochs,
				'state_dict': G_BA.state_dict(),
				'optimizer': optimizer_G.state_dict()

			}

			torch.save(state_G_AB, os.path.join(filepath,'G_BA','epoch_' + str(epoch+1) + '.pkl'))

			# Saving Discriminator
			state_D_A = {
				'epoch': epoch+previous_epochs+1,
				'state_dict': D_A.state_dict(),
				'optimizer': optimizer_D_A.state_dict()
			}

			torch.save(state_D_A, os.path.join(filepath,'D_A','epoch_' + str(epoch+1) + '.pkl'))

			state_D_B = {
				'epoch': epoch+previous_epochs+1,
				'state_dict': D_B.state_dict(),
				'optimizer': optimizer_D_B.state_dict()
			}

			torch.save(state_D_B, os.path.join(filepath,'D_B','epoch_' + str(epoch+1) + '.pkl'))

	if param['log']['save_imgs']:
		if epoch % param['log']['save_image_interval'] == 0:
			sample_images(filepath, epoch)
