import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from random import randint
from os import listdir
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def createDataCSV(fruits_file, textures_file, fruits_path, textures_path):
    # If the fruits csv does not exist, create it
    if (not os.path.isfile(dataset_file)):
        # Create the fruits name list
        names = []
        for fruit_class in os.listdir(os.path.join(fruits_path, 'Training')):
            names.append(str(fruit_class))

        # Create a data frame and later a csv with the file names
        folders_to_add = ['Training', 'Validation']
        df = pd.DataFrame(columns=['Class', 'Path'])
        for folder in folders_to_add:
            if 'DS' not in folder:
                for fruit_class in os.listdir(os.path.join(fruits_path, folder)):
                    if 'DS' not in fruit_class:
                        for name in os.listdir(os.path.join(fruits_path, folder, fruit_class)):
                            df = df.append({'Class': str(fruit_class), 'Path': str(os.path.join(fruits_path,folder, fruit_class, name))}, ignore_index=True )
        df.to_csv(fruits_file, index=False)
        del(df)

    # If the textures csv does not exist, create it
    if (not os.path.isfile(textures_file)):
        df = pd.DataFrame(columns=['Path'])

        for t in os.listdir(textures_path):
            df = df.append({'Path': str(os.path.join(textures_path,t))}, ignore_index=True )
        df.to_csv(textures_file, index=False)

        del(df)

class FruitsDataset(Dataset):
    """Fruits dataset."""

    def __init__(self, csv_file, cl_A, cl_B, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        aux = pd.read_csv(csv_file)
        self.fruits_A_idx = aux[aux['Class'] == cl_A]
        self.fruits_B_idx = aux[aux['Class'] == cl_B]
        self.transform = transform

    def __len__(self):
        return min(len(self.fruits_A_idx), len(self.fruits_B_idx))

    def __getitem__(self, idx):
        img_name_A = self.fruits_A_idx.iloc[idx, 1]
        image_A = io.imread(img_name_A)
        img_name_B = self.fruits_B_idx.iloc[idx, 1]
        image_B = io.imread(img_name_B)
        result = [image_A,image_B]

        if self.transform:
            result = self.transform(result)

        return result

class TexturesDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.textures = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.textures)

    def __getitem__(self, idx):
        img_name = self.textures.iloc[idx, 0]
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image

def insert_image(background, image, nsize, start_img_x, start_img_y, end_img_x, end_img_y, start_res_x, start_res_y, end_res_x, end_res_y):
    img = Image.fromarray(image)
    img = img.resize((nsize,nsize), Image.ANTIALIAS)
    image = np.array(img)
    # Write our image in top of the background
    result = background.copy()
    result[start_res_x:end_res_x,start_res_y:end_res_y] = \
        np.where(image[start_img_x:end_img_x,start_img_y:end_img_y] >= [240,240,240], \
        result[start_res_x:end_res_x,start_res_y:end_res_y], image[start_img_x:end_img_x,start_img_y:end_img_y])
    # Resize the image to a lower size
    low_size = 128
    res = Image.fromarray(np.array(result))
    res = res.resize((low_size,low_size), Image.ANTIALIAS)
    result = np.array(res)
    # Divide by 255 and substract 0.5
    result = (result / 255.0) - 0.5
    return result

class ChangeBackground(object):


    def __init__(self, textures):
        self.textures = textures

    def __call__(self, images):
        maxvals = len(self.textures)
        text_idx = randint(0,maxvals-1) # random selection of the texture background
        background = self.textures[text_idx] # adquisition of the background
        h_res, w_res = background.shape[:2]
        nsize = randint(int(h_res / 4), int(h_res)) # random scaling of the image
        h_img, w_img = nsize, nsize
        # Generate random position (1/4 outside include)
        start_res_x = randint(int(-h_img/2), int(h_res-h_img/2))
        start_res_y = randint(int(-w_img/2), int(w_res-w_img/2))
        end_res_x = start_res_x + h_img
        end_res_y = start_res_y + w_img
        # Fix too far left
        if start_res_x < 0:
            start_img_x = -start_res_x
            start_res_x = 0
        else:
            start_img_x = 0
        # Fix too far top
        if start_res_y < 0:
            start_img_y = -start_res_y
            start_res_y = 0
        else:
            start_img_y = 0
        # Fix too far right
        if end_res_x > h_res:
            end_res_x = h_res
            end_img_x = (end_res_x-start_res_x)
        else:
            end_img_x = h_img
        # Fix too far bottom
        if end_res_y > w_res:
            end_res_y = w_res
            end_img_y = (end_res_y-start_res_y)
        else:
            end_img_y = w_img
        # We genereate both images
        result_A = insert_image(background, images[0], nsize, start_img_x,
            start_img_y, end_img_x, end_img_y, start_res_x, start_res_y, end_res_x, end_res_y)
        result_B = insert_image(background, images[1], nsize, start_img_x,
            start_img_y, end_img_x, end_img_y, start_res_x, start_res_y, end_res_x, end_res_y)
        return [result_A, result_B]


class myReshape(object):

   # def __init__(self):

    def __call__(self,images):
        result_A = np.transpose(np.transpose(images[0], (0,2,1)),(1,0,2))
        result_B = np.transpose(np.transpose(images[1], (0,2,1)),(1,0,2))
        return [result_A, result_B]
