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
from dataset import NewDataset
import numpy as np
import torch.nn.functional as F
import networks
from torchvision import models

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--loadSize', type=int, default=288, help='the height / width of the input image to network')
parser.add_argument('--fineSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--nz', type=int, default=64, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=300000, help='number of training iterations')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--gpu', type=int, default=-1, help='gpu device, -1 for cpu')
parser.add_argument('--outf', default='result/', help='folder to save checkpoint images and models')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--save_step', type=int, default=2000, help='save weights every how many iterations')
parser.add_argument('--use_decay', action='store_true', help='use learning rate decay')
parser.add_argument('--lr_decay_every', type=int, default=1000, help='decay lr this many iterations')
parser.add_argument('--use_perceptual', action='store_true', help='use perceptual loss')
parser.add_argument('--domain_adaptation', action='store_true', help='use wild data for domain adaptation')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

opt.cuda = (opt.gpu != -1)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
device = torch.device("cuda:%d"%(opt.gpu) if opt.cuda else "cpu")

###############   Dataset   ##################
dataset = NewDataset(opt.loadSize,opt.fineSize,opt.flip)
loader_ = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=8)
loader = iter(loader_)

if opt.domain_adaptation:
    wild_dataset = NewDataset(opt.loadSize,opt.fineSize,opt.flip)
    wild_dataset.unsupervied = True
    wild_loader_ = torch.utils.data.DataLoader(dataset=wild_dataset,
                                               batch_size=opt.batchSize,
                                               shuffle=True,
                                               num_workers=8)
    wild_loader = iter(wild_loader_)

###############   Model   ####################

netG = networks.define_G(6, 3, opt.ngf,'unet_256')
if opt.use_perceptual or opt.domain_adaptation:
    netG.load_state_dict(torch.load('%s/netG.pth' % (opt.outf), map_location=lambda storage, loc: storage))
netG = netG.to(device)

optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=opt.lr, betas=(0.5, 0.9))

if opt.use_perceptual:
    loss_network = networks.LossNetwork().to(device).eval()
    for p in loss_network.parameters(): 
        p.requires_grad = True

if opt.domain_adaptation:
    netD = networks.FCDiscriminator(64).to(device)
    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=opt.lr, betas=(0.5, 0.9))

    bce_loss = torch.nn.BCEWithLogitsLoss()


###########   Training   ###########
CRITIC_ITERS = 5

def adjust_learning_rate(optimizer, niter):
    lr = opt.lr * (0.95 ** (niter // opt.lr_decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

Loss_G_ = []
Loss_Per_ = []
Loss_D_ = []
Loss_Dis_ = []
lambda_gp = 10

for iteration in range(1,opt.niter+1):

    if opt.domain_adaptation:
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for i in range(CRITIC_ITERS):
            # 1. generate results of netG
            try:
                data= loader.next()
            except StopIteration:
                loader = iter(loader_)
                data = loader.next()

            try:
                wild_data= wild_loader.next()
            except StopIteration:
                wild_loader = iter(wild_loader_)
                wild_data = wild_loader.next()

            Mask = data['Mask'].to(device)
            FullStyled = data['FullStyled'].to(device)
            Character = data['Character'].to(device)
            WildFullStyled = wild_data['FullStyled'].to(device)
            WildCharacter = wild_data['Character'].to(device)

            # Adversarial loss
            Mask_predic, Features = netG(torch.cat([Character, FullStyled], 1))
            WildMask_predic, WildFeatures = netG(torch.cat([WildCharacter, WildFullStyled], 1))

            netD.zero_grad()

            D_real = netD(Features)
            D_fake = netD(WildFeatures)
            errD_D = bce_loss(D_real, Variable(torch.FloatTensor(D_real.data.size()).fill_(1)).to(device)) + \
            bce_loss(D_fake, Variable(torch.FloatTensor(D_fake.data.size()).fill_(0)).to(device))
            errD_D.backward()

            optimizerD.step()
            Loss_D_.append(errD_D.item())
    else:
        errD_D = 0

    ############################
    # (2) Update G network
    ###########################
    netG.zero_grad()

    # 1. load data
    try:
        data= loader.next()
    except StopIteration:
        loader = iter(loader_)
        data = loader.next()

    Mask = data['Mask'].to(device)
    FullStyled = data['FullStyled'].to(device)
    Character = data['Character'].to(device)

    if opt.domain_adaptation:
        try:
            wild_data= wild_loader.next()
        except StopIteration:
            wild_loader = iter(wild_loader_)
            wild_data = wild_loader.next()
        WildFullStyled = wild_data['FullStyled'].to(device)
        WildCharacter = wild_data['Character'].to(device)

    # 2. segmentation
    Mask_predic, Features = netG(torch.cat([Character, FullStyled], 1))
    if opt.domain_adaptation:
        WildMask_predic, WildFeatures = netG(torch.cat([WildCharacter, WildFullStyled], 1))

    # 3. loss
    err1 = torch.mean(torch.abs(Mask_predic-Mask))
    Loss_G_.append(err1.item())

    errD = 0
    if opt.domain_adaptation:
        D_out1 = netD(WildFeatures)
        errD = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(1)).to(device))
        Loss_Dis_.append(errD.item())
    else:
        Loss_Dis_.append(0)

    err2 = 0
    if opt.use_perceptual:
        Result_predic = loss_network(Mask_predic)
        Result_ori = loss_network(Mask)
        for p in range(5):
            err2 += torch.mean(torch.abs(Result_predic[p]-Result_ori[p]))
        Loss_Per_.append(err2.item())
    else:
        Loss_Per_.append(0)

    G_cost = err1 + err2 + errD * 0.01

    # 4. bp and update
    G_cost.backward()
    optimizerG.step()

    print('[%d/%d] Loss_L1: %.4f Loss_per %.4f Loss_dis: %.4f Loss_D: %.4f' % (iteration, opt.niter, err1, err2, errD, errD_D))

    ########### Drawing Fig #########
    if(iteration % 100 == 0):
        Fig = plt.figure(0)
        Axes = plt.subplot(111)
        plt.ylabel('Loss L1')
        plt.xlabel('Iteration')
        plt.plot(range(iteration), Loss_G_, 'r')
        plt.plot(range(iteration), Loss_Per_, 'b')
        plt.plot(range(iteration), Loss_Dis_, 'g')
        plt.savefig('%s/Fig_netG.eps' % (opt.outf), dpi = 1000, bbox_inches='tight')
        plt.close()

        if opt.domain_adaptation:
            Fig = plt.figure(0)
            Axes = plt.subplot(111)
            plt.ylabel('Loss L1')
            plt.xlabel('Iteration')
            plt.plot(range(iteration*CRITIC_ITERS), Loss_D_, 'r')
            plt.savefig('%s/Fig_netD.eps' % (opt.outf), dpi = 1000, bbox_inches='tight')
            plt.close()

        if opt.use_decay:
            optimizerG = adjust_learning_rate(optimizerG, iteration)
            optimizerD = adjust_learning_rate(optimizerD, iteration)

        vutils.save_image(Character.data,
                    '%s/Character.png' % (opt.outf),
                    normalize=True)
        vutils.save_image(Mask.data,
                    '%s/Mask.png' % (opt.outf),
                    normalize=True)
        vutils.save_image(FullStyled.data,
                    '%s/FullStyled.png' % (opt.outf),
                    normalize=True)
        vutils.save_image(Mask_predic.data,
                    '%s/Mask_predic.png' % (opt.outf),
                    normalize=True)
        if opt.domain_adaptation:
            vutils.save_image(WildFullStyled.data,
                        '%s/WildFullStyled.png' % (opt.outf),
                        normalize=True)
            vutils.save_image(WildMask_predic.data,
                        '%s/WildMask_predic.png' % (opt.outf),
                        normalize=True)

    ########## Visualize #########
    if(iteration % opt.save_step == 0):
        torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
        torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
