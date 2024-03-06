import math
import cv2
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common
from dct2d import Dct2d
from Quantization import Quantization
from Subsample import chroma_subsampling, chroma_upsampling, ycbcr_to_rgb_jpeg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

load(c.MODEL_PATH + c.suffix)

net.eval()

dct = Dct2d()
jpeg = Quantization()
jpeg.set_quality(90)

up = chroma_upsampling()

with torch.no_grad():
    for i, data in enumerate(datasets.backwardloader):
        data = data.to(device)          #first channel(batch size) = 2
        # cover = data[data.shape[0] // 2:, :, :, :]
        # secret = data[:data.shape[0] // 2, :, :, :]
        # cover_input = dct(cover)
        # secret_input = dct(secret)
        # input_img = torch.cat((cover_input, secret_input), 1)
        
        #data = up(data)
        steg_input = dct(data)

        #################
        #    forward:   #
        #################
        '''output = net(input_img)
        output_steg = output.narrow(1, 0, c.channel_dct * c.channels_in)
        output_z = output.narrow(1, c.channel_dct * c.channels_in, output.shape[1] - c.channel_dct * c.channels_in)
        steg_img = dct.inverse(output_steg)'''
        backward_z = gauss_noise(steg_input.shape)
        
        #####################
        #   quantization:   #
        #####################
        #output_steg = jpeg(output_steg)
        #output_steg = jpeg.inverse(output_steg)

        #################
        #   backward:   #
        #################
        output_rev = torch.cat((steg_input, backward_z), 1)
        bacward_img = net(output_rev, rev=True)
        secret_rev = bacward_img.narrow(1, c.channel_dct * c.channels_in, bacward_img.shape[1] - c.channel_dct * c.channels_in)
        secret_rev = dct.inverse(secret_rev)
        cover_rev = bacward_img.narrow(1, 0, c.channel_dct * c.channels_in)
        cover_rev = dct.inverse(cover_rev)
        # resi_cover = (steg_img - cover) * 20
        # resi_secret = (secret_rev - secret) * 20

        # torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i)
        # torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
        # torchvision.utils.save_image(steg_img, c.IMAGE_PATH_steg + '%.5d.png' % i)
        torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_backward + '%.5d.png' % i)
        
        




