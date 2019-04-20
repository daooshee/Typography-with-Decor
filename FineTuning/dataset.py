from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable, grad
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
import math
import glob
import cv2


# Important paths
Decoration_Data = [img for img in glob.glob('Data/Decoration/*.png')]
training_set_path = "Data/TextEffects"


# Functions and Classes
def default_loader(path):
    return cv2.imread(path)

def add_decoration(Character, Styled):
    possi_pose = np.where(Character[:,:,2]==255)

    Num = random.randint(1, 5)
    Decoration_Mask = np.zeros([320,320,3])
    Final_Styled = Styled.copy()

    if len(possi_pose[0]) > 0:
        for i in range(Num):
            FilePath = Decoration_Data[random.randint(0, len(Decoration_Data)-1)]
            TempD = cv2.imread(FilePath, cv2.IMREAD_UNCHANGED)
            img_w,img_h = TempD.shape[0],TempD.shape[1]
            img_w_new = random.randint(30, 60)
            img_h_new = int(float(img_h) / img_w * img_w_new)

            TempD = cv2.resize(TempD,(img_w_new,img_h_new))

            while_count = 0

            while(1):
                pos = random.randint(0, len(possi_pose[0])-1)
                pos_x = possi_pose[0][pos]
                pos_y = possi_pose[1][pos]

                pos_x -= int(img_w_new/2)
                pos_y -= int(img_h_new/2)

                if pos_x < 0 or pos_x+img_h_new > 320 or pos_y < 0 or pos_y+img_w_new > 320:
                    continue

                Max = Decoration_Mask[pos_x:pos_x+img_h_new,pos_y:pos_y+img_w_new, 0].max()

                if Max == 0:
                    break

                while_count += 1

                if while_count > 30:
                    break

            if while_count > 30:
                    break

            if TempD.shape[2] == 3:
                TempD_Mask = np.ones([img_h_new,img_w_new]) * 255
            else:
                TempD_Mask = TempD[:,:,3]

            for j in range(3):
                Decoration_Mask[pos_x:pos_x+img_h_new,pos_y:pos_y+img_w_new, j] = TempD_Mask

            TempD_Mask = TempD_Mask/255.

            for j in range(3):
                Final_Styled[pos_x:pos_x+img_h_new,pos_y:pos_y+img_w_new,j] = Final_Styled[pos_x:pos_x+img_h_new,pos_y:pos_y+img_w_new,j] * (1-TempD_Mask) + TempD[:,:,j] * TempD_Mask

    return Decoration_Mask, Final_Styled

def ColorChange(Img, randint):
    Img = cv2.cvtColor(Img,cv2.COLOR_BGR2HSV)
    Img[:,:,0] = (Img[:,:,0] + randint) % 181
    return cv2.cvtColor(Img,cv2.COLOR_HSV2BGR)

def RandomColorType1(Character, random1, random2, random3, r1, r2, r3, r4, r5, r6):
    FG = Character[:,:,0] / 255.
    FG_ = FG ** random1
    Result = Character.copy()
    Result[:,:,0] = r1 * FG_ + (1-FG_) * r2
    FG_ = FG ** random2
    Result[:,:,1] = r3 * FG_ + (1-FG_) * r4
    FG_ = FG ** random3
    Result[:,:,2] = r5 * FG_ + (1-FG_) * r6
    return Result

def ToTensor(pic):
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    return img.float().div(255)

class NewDataset(data.Dataset):
    def __init__(self, style_path, glyph_path):
        super(NewDataset, self).__init__()
        self.Blank_1 = default_loader(glyph_path)
        self.Blank_1 = cv2.resize(self.Blank_1,(320,320))
        self.Blank_2 = self.Blank_1.copy()

        self.Stylied_1 = default_loader(style_path)
        self.Stylied_1 = cv2.resize(self.Stylied_1,(320,320))
        self.Stylied_2 = self.Stylied_1.copy()

        self.Mask = 255-default_loader("temp/mask_final.jpg")
        self.Mask = cv2.resize(self.Mask,(320,320))

        self.loadSize = 288
        self.fineSize = 256
        self.flip = False

        self.training_set = glob.glob(training_set_path+"/*/train")

    def Process(self, img2, size, x1, y1, flip_rand):
        h,w,c = img2.shape

        if(h != size):
            img2 = cv2.resize(img2,(size, size))
            img2 = img2[x1:x1+self.fineSize, y1:y1+self.fineSize, :]


        if(self.flip == 1):
            if flip_rand <= 0.25:
                img2 = cv2.flip(img2, 1)
            elif flip_rand <= 0.5:
                img2 = cv2.flip(img2, 0)
            elif flip_rand <= 0.75:
                img2 = cv2.flip(img2, -1)

        img2 = ToTensor(img2) 

        return img2.mul_(2).add_(-1)

    def Process_mask(self, img2, size, x1, y1, flip_rand):
        h,w,c = img2.shape

        if(h != size):
            img2 = cv2.resize(img2,(size, size))
            img2 = img2[x1:x1+self.fineSize, y1:y1+self.fineSize, :]


        img2 = ToTensor(img2)
        
        return img2

    def __getitem__(self, index):
        flip_rand = random.random()
        random_style = random.random()

        # Original training data
        if random_style < 0.6:

            # Random Type 1
            if random_style < 0.3:
                random1 = random.random()
                random2 = random.random()
                random3 = random.random()
                r1 = random.randint(0, 255)
                r2 = random.randint(0, 255)
                r3 = random.randint(0, 255)
                r4 = random.randint(0, 255)
                r5 = random.randint(0, 255)
                r6 = random.randint(0, 255)

                Blank_1 = self.Blank_1.copy()
                Blank_2 = self.Blank_2.copy()

                Stylied_1 = RandomColorType1(Blank_1, random1, random2, random3, r1, r2, r3, r4, r5, r6)
                Stylied_2 = RandomColorType1(Blank_2, random1, random2, random3, r1, r2, r3, r4, r5, r6)
                _ , Stylied_1 = add_decoration(Blank_1, Stylied_1)
                Mask = np.ones(self.Mask.shape) * 255

            # Training Set
            else:
                style = random.randint(0, len(self.training_set)-1)
                content1 = random.randint(0, 854)
                content2 = random.randint(0, 854)
                img1 = default_loader((self.training_set[style]+"/%d.png")%(content1))
                img2 = default_loader((self.training_set[style]+"/%d.png")%(content2))
                h,w,c = img1.shape
                Blank_1 = img1[:,:h,:]
                Blank_2 = img2[:,:h,:]
                Stylied_1 = img1[:,h:,:]
                Stylied_2 = img2[:,h:,:]
                randint = random.randint(-180, 180)
                Stylied_1 = ColorChange(Stylied_1,randint)
                Stylied_2 = ColorChange(Stylied_2,randint)
                _ , Stylied_1 = add_decoration(Blank_1, Stylied_1)
                Mask = np.ones(self.Mask.shape) * 255

        # Patches randomly cropped from the given image for one-shot fine-tuning
        else:
            Blank_1 = self.Blank_1.copy()
            Blank_2 = self.Blank_2.copy()
            Stylied_1 = self.Stylied_1.copy()
            Stylied_2 = self.Stylied_2.copy()
            Mask = self.Mask.copy()

        # Processing
        size = self.loadSize

        x1 = random.randint(0, size - self.fineSize)
        y1 = random.randint(0, size - self.fineSize)
        x2 = random.randint(0, size - self.fineSize)
        y2 = random.randint(0, size - self.fineSize)

        Data = {}

        Data['Blank_1'] = self.Process(Blank_1,size,x1,y1,flip_rand)
        Data['Blank_2'] = self.Process(Blank_2,size,x2,y2,flip_rand)

        Stylied_1 = cv2.cvtColor(Stylied_1,cv2.COLOR_BGR2RGB)
        Data['Stylied_1'] = self.Process(Stylied_1,size,x1,y1,flip_rand)

        Stylied_2 = cv2.cvtColor(Stylied_2,cv2.COLOR_BGR2RGB)
        Data['Stylied_2'] = self.Process(Stylied_2,size,x2,y2,flip_rand)

        Data['Mask'] = self.Process_mask(Mask,size,x2,y2,flip_rand)

        return Data

    def __len__(self):
        return 1000