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
from dataset import NewDataset


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=2, help='batch size')
parser.add_argument('--niter', type=int, default=300, help='number of iterations for fine-tuning')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--gpu', type=int, default=-1, help='gpu device, -1 for cpu')
parser.add_argument('--netf', help='where are netG.pth and netD.pth')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--style_name', help='name of the style image')
parser.add_argument('--style_path', help='path to the style image')
parser.add_argument('--glyph_path', help='path to the corresponding glyph of the style image')


plt.switch_backend('agg')

opt = parser.parse_args()
print(opt)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.cuda = (opt.gpu != -1)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
device = torch.device("cuda:%d"%(opt.gpu) if opt.cuda else "cpu")


###############   Model   ####################
netG = networks.define_G(9, 3).to(device)
netD = networks.define_D(12).to(device)

netG.load_state_dict(torch.load('%s/netG.pth'%(opt.netf), map_location=lambda storage, loc: storage))
netD.load_state_dict(torch.load('%s/netD.pth'%(opt.netf), map_location=lambda storage, loc: storage))

optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=opt.lr, betas=(0.5, 0.9))
optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=opt.lr, betas=(0.5, 0.9))

criterion = nn.L1Loss()


###########    Loss   ###########
def calc_gradient_penalty(netD, real_data, fake_data):

    alpha = torch.rand(real_data.shape[0], 1, 1, 1).to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


###############   Dataset   ##################
dataset = NewDataset(opt.style_path, opt.glyph_path)
loader_ = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=8)
loader = iter(loader_)

###########   Training   ###########
CRITIC_ITERS = 2
lambda_gp = 10
current_size = 256
Min_loss = 100000

for iteration in range(1,opt.niter+1):

    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():
        p.requires_grad = True
    for p in netG.parameters():
        p.requires_grad = False

    for i in range(CRITIC_ITERS):
        # 1. generate results of netG
        try:
            data= loader.next()
        except StopIteration:
            loader = iter(loader_)
            data = loader.next()

        Blank_1 = data['Blank_1'].to(device)
        Blank_2 = data['Blank_2'].to(device)
        Stylied_1 = data['Stylied_1'].to(device)
        Stylied_2 = data['Stylied_2'].to(device)
        Mask = data['Mask'].to(device)

        Stylied_2_recon = netG(torch.cat([Blank_2, Blank_1, Stylied_1], 1), current_size)

        # 2. train netD
        input_real = torch.cat([Blank_2, Blank_1, Stylied_1, Stylied_2 * Mask], 1)
        input_fake = torch.cat([Blank_2, Blank_1, Stylied_1, Stylied_2_recon * Mask], 1)

        netD.zero_grad()
        D_real = netD(input_real).mean()
        D_fake = netD(input_fake).mean()
        gradient_penalty = calc_gradient_penalty(netD, input_real.data, input_fake.data)
        errD = D_fake.mean() - D_real.mean() + lambda_gp * gradient_penalty
        errD.backward()
        Wasserstein_D = (D_real.mean() - D_fake.mean()).data.mean()
        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False
    for p in netG.parameters(): 
        p.requires_grad = True

    netG.zero_grad()

    # 1. load data
    try:
        data= loader.next()
    except StopIteration:
        loader = iter(loader_)
        data = loader.next()

    Blank_1 = data['Blank_1'].to(device)
    Blank_2 = data['Blank_2'].to(device)
    Stylied_1 = data['Stylied_1'].to(device)
    Stylied_2 = data['Stylied_2'].to(device)
    Mask = data['Mask'].to(device)

    # 2. netG process
    Stylied_2_recon = netG(torch.cat([Blank_2, Blank_1, Stylied_1], 1), 256)

    # 3. loss
    errS2 = torch.mean(torch.abs(Stylied_2_recon-Stylied_2) * Mask)

    input_fake = torch.cat([Blank_2, Blank_1, Stylied_1, Stylied_2_recon * Mask], 1)
    errD = netD(input_fake).mean()

    G_cost = errS2 * 100 - errD

    # 4. back propogation and update
    G_cost.backward()
    optimizerG.step()

    print('[%d/%d] Loss_L1: %.4f Loss_adv: %.4f Wasserstein_D: %.4f' % (iteration, opt.niter, errS2.item(), errD.item(), Wasserstein_D.item()))

    if errS2.item() < Min_loss and iteration > 100:
        Min_loss = errS2.item()
        
        vutils.save_image(Stylied_1.data,
                    'checkpoints/input_style.png',
                    normalize=True)
        vutils.save_image(Stylied_2_recon.data,
                    'checkpoints/output.png',
                    normalize=True)
        vutils.save_image(Stylied_2.data,
                    'checkpoints/ground_truth.png',
                    normalize=True)
        vutils.save_image(Stylied_2 * Mask.data,
                    'checkpoints/ground_truth_masked.png',
                    normalize=True)

        torch.save(netG.state_dict(), 'cache/%s_netG.pth'%(opt.style_name))

