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

import networks

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='gpu device')
parser.add_argument('--outf', default='result/', help='folder to save output images')
parser.add_argument('--style_name', help='name of the style image')
parser.add_argument('--style_path', help='path to the input style image')
parser.add_argument('--glyph_path', help='path to the corresponding glyph of the input style image')
parser.add_argument('--content_path', help='path to the target content image')
parser.add_argument('--save_name', default='name to save')

plt.switch_backend('agg')

opt = parser.parse_args()

opt.cuda = (opt.gpu != -1)
cudnn.benchmark = True
device = torch.device("cuda:%d"%(opt.gpu) if opt.cuda else "cpu")


###############   Model  ####################
netG = networks.define_G(9, 3).to(device)
netG.load_state_dict(torch.load('cache/%s_netG.pth'%(opt.style_name), map_location=lambda storage, loc: storage))
netG.eval()
for p in netG.parameters():
    p.requires_grad = False


###############   Processing   ####################
def image_loader(image_name):
    image = cv2.imread(image_name)
    image = cv2.resize(image,(256, 256))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = image.float().div(255)
    image = image.mul_(2).add_(-1)
    return image.unsqueeze(0)

def image_loader_v2(image_name):
    image = cv2.imread(image_name)
    image = cv2.resize(image,(256, 256))
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = image.float().div(255)
    image = image.mul_(2).add_(-1)
    return image.unsqueeze(0)

def imsave(tensor, title="Output"):
    vutils.save_image((tensor.data+1)*0.5,'%s.png' % (title))

Blank_1 = image_loader_v2(opt.glyph_path).to(device)
Blank_2 = image_loader_v2(opt.content_path).to(device)
Stylied_1 = image_loader(opt.style_path).to(device)

Stylied_2_recon = netG(torch.cat([Blank_2, Blank_1, Stylied_1], 1), 256)

vutils.save_image(Stylied_2_recon.data[0:1,:,:,:]/2+0.5, '%s/%s.png' % (opt.outf, opt.save_name))
