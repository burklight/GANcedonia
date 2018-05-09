
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import database as db
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# # Discriminator

# In[2]:


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


# # Generator

# In[3]:


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


# In[4]:


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
                        nn.Sigmoid() ])
                
        self.res_net = nn.Sequential(*layers)
        
    def forward(self, image):
        return self.res_net(image)


# # Training

# In[5]:


criterion = nn.BCELoss()
cuda = True if torch.cuda.is_available() else False # for using the GPU if possible


# In[6]:


# Calculate the number of patches (61x61) separated 25
batch_size = 4
img_x = 128
img_y = 128
separation = 25
patch_x, patch_y = 26, 26
patch = (batch_size, 1 , patch_x, patch_y)


# In[7]:


# Generation of the discriminator and generator
D = GANdiscriminator()
G = GANgenerator()

if cuda:
    D = D.cuda()
    G = G.cuda()
    criterion = criterion.cuda()


# In[8]:


# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal(m.weight.data, a=0.05)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.kaiming_normal(m.weight.data, a=0.05)
        torch.nn.init.constant_(m.bias.data, 0.0)

G.apply(weights_init); # He initialization of the weights
D.apply(weights_init); # He initialization of the weights


# In[9]:


# Define the optimizer for the generator
lr = 1e-4
beta_1 = 0.9
beta_2 = 0.99
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta_1, beta_2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta_1, beta_2))


# In[10]:


# Scheduler (TO DO)
patch


# In[11]:


# Inputs and targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_img = Tensor(batch_size,3,img_x,img_y)
valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)


# In[12]:


# Prepare the data
fruits_file = 'Dataset/dataset_index.csv'
textures_file = 'Dataset/textures_index.csv'
textures = db.TexturesDataset(csv_file=textures_file)
train_imgs = db.FruitsDataset(csv_file=fruits_file, cl='Strawberry', 
                        transform = transforms.Compose(
                            [db.ChangeBackground(textures),
                             db.myReshape()]))
dataloader = DataLoader(train_imgs, batch_size=batch_size,
                        shuffle=True, num_workers=4)


# In[ ]:


# Training phase
n_epochs = 5
for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):
        
        # Get model input
        input_img = Variable(batch.type(Tensor))
        
        ''' Generator '''
        
        # Set the optimizer
        optimizer_G.zero_grad()
        
        # Create some noise
        noise = Variable(Tensor(np.random.normal(loc = 0.0,             scale= 0.5, size=batch.shape)))
    
        # Generate some images from the noise
        generated = G(noise)
        
        # See how well they do
        loss_g = criterion(D(generated),valid)
        
        # We use backpropagation
        loss_g.backward(retain_graph=True)
        
        # And make ADAM do its shit
        optimizer_G.step()
        
        ''' Discriminator '''
        
        # Set the optimizer
        optimizer_D.zero_grad()
        
        loss_d_real = criterion(D(input_img),valid)
        loss_d_fake = criterion(D(generated),fake)
        loss_d = 0.5 * (loss_d_real + loss_d_fake)
        
        # We use backpropagation
        loss_d.backward(retain_graph=True)
        
        # We let ADAM do its shit
        optimizer_D.step()
        
        print("Image #: " + str(i*batch_size))
        


# In[ ]:


# Create some noise
noise = Tensor(np.random.normal(loc = 0.0,    scale= 0.5, size=batch.shape))

# Generate some images from the noise
generated = G(noise)


# In[ ]:


v = generated[0].data.cpu().numpy()
v = np.transpose(v, (2,1,0))
v.shape


# In[ ]:


plt.imshow(v)
plt.show()


# In[ ]:


a = np.random.normal()

