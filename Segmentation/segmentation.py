# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np
import glob
import copy
import warnings
import cv2

import networks

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--img', help='The artistic word')
parser.add_argument('--img_content', help='The word with distance information')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = parser.parse_args()

###### Parameters ######
ngf = 64
###### Parameters ######

def image_loader(image_name):
    image = cv2.imread(image_name)
    image = cv2.copyMakeBorder(image,16,16,16,16,cv2.BORDER_REPLICATE)
    image = cv2.resize(image,(256, 256))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = image.float().div(255)
    image = image.mul_(2).add_(-1)
    return image.unsqueeze(0)

def imsave(tensor, title):
    tensor = tensor[0,:,:,:].data.cpu().numpy()
    tensor = tensor.transpose((1, 2, 0))
    tensor = cv2.resize(tensor,(288, 288))
    image = tensor[16:-16,16:-16,:]/2+0.5
    cv2.imwrite('%s.jpg' % (title), image*255)

netSeg = networks.define_G(6, 3, ngf,'unet_256').to(device)
netSeg.load_state_dict(torch.load('Segmentation/netSeg.pth', map_location=lambda storage, loc: storage))

input_content = image_loader(opt.img_content).to(device)
input_style = image_loader(opt.img).to(device)

result = netSeg(torch.cat([input_content, input_style], 1))
imsave(result, title='temp/mask_ori')
