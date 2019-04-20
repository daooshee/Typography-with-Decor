import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import glob
import math
import cv2

# Important paths
Decoration_Data = glob.glob('Data/Decoration/*.png')
training_set_path = "Data/TextEffects"
Texture_Data = glob.glob('Data/Texture/texture*.*')
WildData = glob.glob('Data/WildDataMask/*.jpeg')

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

def ColorChange(Img):
    Img = cv2.cvtColor(Img,cv2.COLOR_BGR2HSV)
    Img[:,:,0] = (Img[:,:,0] + random.randint(-180, 180)) % 181
    return cv2.cvtColor(Img,cv2.COLOR_HSV2BGR)

def RandomColorType1(Character):
    FG = Character[:,:,0] / 255.
    FG_ = FG ** random.random()
    Result = Character.copy()
    Result[:,:,0] = random.randint(0,255) * FG_ + (1-FG_) * random.randint(0,255)
    FG_ = FG ** random.random()
    Result[:,:,1] = random.randint(0,255) * FG_ + (1-FG_) * random.randint(0,255)
    FG_ = FG ** random.random()
    Result[:,:,2] = random.randint(0,255) * FG_ + (1-FG_) * random.randint(0,255)
    return Result

def RandomColorType2(Character):
    FG = Character[:,:,2] / 255.
    TX_BG = cv2.imread(Texture_Data[random.randint(0, len(Texture_Data)-1)])
    TX_FG = cv2.imread(Texture_Data[random.randint(0, len(Texture_Data)-1)])
    Result = TX_BG.copy()
    for i in range(3):
        Result[:,:,i] = Result[:,:,i] * FG + TX_FG[:,:,i] * (1-FG)
    return Result

def ToTensor(pic):
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    return img.float().div(255)


class NewDataset(data.Dataset):
    def __init__(self, loadSize=64, fineSize=64, flip=1):
        super(NewDataset, self).__init__()
        self.img_path = glob.glob(training_set_path+"/*/train/*.png")
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip
        self.unsupervied = False

    def Process(self, img2, size, x1, y1, flip_rand):
        h,w,c = img2.shape
        if(h != size):
            img2 = cv2.resize(img2,(size, size))

        if(size != self.fineSize):
            img2 = img2[x1:x1+self.fineSize, y1:y1+self.fineSize, :]

        if(self.flip == 1):
            if flip_rand <= 0.25:
                img2 = cv2.flip(img2, 1)
            elif flip_rand <= 0.5:
                img2 = cv2.flip(img2, 0)
            elif flip_rand <= 0.75:
                img2 = cv2.flip(img2, -1)

        img2 = ToTensor(img2) # 3 x 256 x 256

        img2 = img2.mul_(2).add_(-1)
        # 3. Return a data pair (e.g. image and label).
        return img2

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).

        size = random.randint(self.fineSize, self.loadSize)
        x1 = random.randint(0, size - self.fineSize)
        y1 = random.randint(0, size - self.fineSize)
        flip_rand = random.random()
        random_style = random.random()
        random_color_change = random.randint(0, 1)

        Data = {}

        if (self.unsupervied):
            img = default_loader(WildData[random.randint(0,len(WildData)-1)])
            h,w,c = img.shape
            Character = img[:,:h,:]
            Final_Styled = img[:,h:,:]
            if random_color_change == 1:
                Final_Styled = ColorChange(Final_Styled)
            Decoration_Mask = np.zeros([size,size,3])
        else:
            img = default_loader(self.img_path[index])
            h,w,c = img.shape

            Character = img[:,:h,:]

            if random_style < 0.5:
                Style = img[:,h:,:]
                if random_color_change == 1:
                    Style = ColorChange(Style)
            elif random_style < 0.75:
                Style = RandomColorType2(Character)
                if random_color_change == 1:
                    Style = ColorChange(Style)
            else:
                Style = RandomColorType1(Character)
            Decoration_Mask, Final_Styled = add_decoration(Character, Style)
            Character = cv2.cvtColor(Character,cv2.COLOR_BGR2RGB)

        Data['Character'] = self.Process(Character,size,x1,y1,flip_rand)
        Data['Mask'] = self.Process(Decoration_Mask,size,x1,y1,flip_rand)
        Final_Styled = cv2.cvtColor(Final_Styled,cv2.COLOR_BGR2RGB)
        Data['FullStyled'] = self.Process(Final_Styled,size,x1,y1,flip_rand)
        return Data

    def __len__(self):
        return len(self.img_path)

