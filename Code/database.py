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

    def __init__(self, csv_file, cl, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        aux = pd.read_csv(csv_file)
        self.fruits_idx = aux[aux['Class'] == cl]
        self.transform = transform

    def __len__(self):
        return len(self.fruits_idx)

    def __getitem__(self, idx):
        img_name = self.fruits_idx.iloc[idx, 1]
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image

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

class ChangeBackground(object):


    def __init__(self, textures):
        self.textures = textures


    def __call__(self, image):
        maxvals = len(self.textures)
        text_idx = randint(0,maxvals-1) # random selection of the texture background
        result = self.textures[text_idx] # adquisition of the background
        h_res, w_res = result.shape[:2]
        # We resize our image to be 1/4
        nsize = randint(int(h_res / 4), int(h_res))
        #img = Image(image)
        img = Image.fromarray(image)
        img = img.resize((nsize,nsize), Image.ANTIALIAS)
        image = np.array(img)
        h_img, w_img = image.shape[:2]
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
        # Write our image in the positions selected
        result[start_res_x:end_res_x,start_res_y:end_res_y] = \
            np.where(image[start_img_x:end_img_x,start_img_y:end_img_y] >= [240,240,240], \
            result[start_res_x:end_res_x,start_res_y:end_res_y], image[start_img_x:end_img_x,start_img_y:end_img_y])
        # Divide by 255 and substract 0.5
        result = (result / 255.0) - 0.5
        return result
