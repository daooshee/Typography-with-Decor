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
import networks

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--loadSize', type=int, default=288, help='the height / width of the input image to network')
parser.add_argument('--fineSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
parser.add_argument('--epoches', type=int, default=50, help='number of training epoches')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--gpu', type=int, default=-1, help='gpu device, -1 for cpu')
parser.add_argument('--outf', default='result/', help='folder to save checkpoint images and models')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

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

print(netG)
print(netD)

optimizerG = optim.Adam(filter(lambda p: p.requires_grad, netG.parameters()), lr=opt.lr, betas=(0.5, 0.9))
optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=opt.lr, betas=(0.5, 0.9))


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


###########   Training   ###########
CRITIC_ITERS = 1 

lambda_gp = 10
finesize = 64

history = 0.0
new = 1.0

BS = {'64':200, '128': 90, '256':40}

for epoch in range(1,opt.epoches+1):
    batch_size = BS['%d'%finesize]
    print('Batch Size: %d'%(batch_size))

    Loss_Dis_ = []
    Loss_Stylied_2_ = []
    Loss_D_ = []

    ###############   DATASET   ##################
    dataset = NewDataset(opt.loadSize, opt.fineSize, opt.flip, finesize)
    loader_ = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)
    loader = iter(loader_)

    iter_per_epoch = int(len(dataset) / batch_size)


    for iteration in range(1,iter_per_epoch+1):
        netG.zero_grad()

        if(history>0 and new<1):
            history -= 0.001
            new += 0.001


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

            Blank_1 = data['Blank_1'].to(device)
            Blank_2 = data['Blank_2'].to(device)
            Stylied_1 = data['Stylied_1'].to(device)
            Stylied_2 = data['Stylied_2'].to(device)
        
            # Adversarial loss
            Stylied_2_recon = netG(torch.cat([Blank_2, Blank_1, Stylied_1], 1), finesize, history, new)

            # 2. train netD
            input_real = torch.cat([Blank_2, Blank_1, Stylied_1, Stylied_2], 1)
            input_fake = torch.cat([Blank_2, Blank_1, Stylied_1, Stylied_2_recon], 1)

            netD.zero_grad()
            D_real = netD(input_real).mean()
            D_fake = netD(input_fake).mean()
            gradient_penalty = calc_gradient_penalty(netD, input_real.data, input_fake.data)
            errD = D_fake.mean() - D_real.mean() + lambda_gp * gradient_penalty
            errD.backward()
            Wasserstein_D = (D_real.mean() - D_fake.mean()).data.mean()
            optimizerD.step()
            Loss_D_.append(Wasserstein_D.item())


        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
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
    

        # 2. net process
        Stylied_2_recon = netG(torch.cat([Blank_2, Blank_1, Stylied_1], 1), finesize, history, new)

        # 3. loss
        errS2 = torch.mean(torch.abs(Stylied_2_recon-Stylied_2))
        input_fake = torch.cat([Blank_2, Blank_1, Stylied_1, Stylied_2_recon], 1)
        errD2 = netD(input_fake).mean()

        G_cost = errS2 * 100 - errD2

        # 4. bp and update
        G_cost.backward()

        optimizerG.step()

        Loss_Stylied_2_.append(errS2.item()*100)
        Loss_Dis_.append(-errD2.item())

        print('Size:%d Epoch:%d [%d/%d] Loss_S2: %.4f Loss_Dis2: %.4f Wasserstein_D: %.4f' \
            % (finesize, epoch, iteration, iter_per_epoch, errS2.item(), errD2.item(), Wasserstein_D.item()))

        ########### Drawing Fig #########
        if(iteration % 100 == 0):
            Fig = plt.figure(0)
            Axes = plt.subplot(111)
            plt.ylabel('Loss')
            plt.xlabel('Iteration')
            plt.plot(range(0,iteration), Loss_Dis_, 'r')
            plt.savefig('%s/Fig_wgan.eps' % (opt.outf), dpi = 1000, bbox_inches='tight')
            plt.close()

            Fig = plt.figure(0)
            Axes = plt.subplot(111)
            plt.ylabel('Loss')
            plt.xlabel('Iteration')
            plt.plot(range(0,iteration), Loss_Stylied_2_, 'g')
            plt.savefig('%s/Fig_L1.eps' % (opt.outf), dpi = 1000, bbox_inches='tight')
            plt.close()

            Fig = plt.figure(1)
            Axes = plt.subplot(111)
            plt.ylabel('Loss')
            plt.xlabel('Iteration')
            plt.plot(range(0,iteration*CRITIC_ITERS), Loss_D_, 'r')
            plt.savefig('%s/Fig_adv.eps' % (opt.outf), dpi = 1000, bbox_inches='tight')
            plt.close()

            vutils.save_image(Blank_1.data,
                        '%s/Glyph_1.png' % (opt.outf),
                        normalize=True)

            vutils.save_image(Blank_2.data,
                        '%s/Glyph_2.png' % (opt.outf),
                        normalize=True)

            vutils.save_image(Stylied_1.data,
                        '%s/Stylied_1.png' % (opt.outf),
                        normalize=True)

            vutils.save_image(Stylied_2_recon.data,
                        '%s/Stylied_2_recon.png' % (opt.outf),
                        normalize=True)

            vutils.save_image(Stylied_2.data,
                        '%s/Stylied_2.png' % (opt.outf),
                        normalize=True)

    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))

    if epoch%5==0 and finesize < 256:
        finesize *= 2
        history = 1.0
        new = 0.0
