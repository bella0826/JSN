import math
import cv2
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common
from dct2d import Dct2d
from Quantization import Quantization
from DiffJPEG import DiffJPEG
from Subsample import chroma_subsampling, chroma_upsampling, ycbcr_to_rgb_jpeg, rgb_to_ycbcr_jpeg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

def loadCb(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    netCb.load_state_dict(network_state_dict)
    try:
        optimCb.load_state_dict(state_dicts['opt'])
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


###################
#    net for Y:   #
###################
net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
# weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

load(c.MODEL_PATH + c.suffix)

net.eval()

#######################
#    net for Cb Cr:   #
#######################
netCb = Model()
netCb.cuda()
init_model(netCb)
netCb = torch.nn.DataParallel(netCb, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, netCb.parameters())))
optimCb = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

loadCb(c.MODEL_PATH + c.suffix_cb)

netCb.eval()

dct = Dct2d()
jpeg = Quantization()
jpeg.set_quality(80)

jpg = DiffJPEG(512, 512, differentiable=True)
jpg.set_quality(50)
subsampling = chroma_subsampling()
upsampling = chroma_upsampling()
rgb = ycbcr_to_rgb_jpeg()
ycbcr = rgb_to_ycbcr_jpeg()

with torch.no_grad():
    psnr_c = []
    psnr_s = []
    psnr_cc = []
    for i, data in enumerate(datasets.testloader):
        data = data.to(device)          #first channel(batch size) = 2

        data = ycbcr(data)
        y, cb, cr = subsampling(data)

        cover = data[data.shape[0] // 2:, :, :, :]
        secret = data[:data.shape[0] // 2, :, :, :]
        cover_y = y[data.shape[0] // 2:, :, :, :]
        secret_y = y[:data.shape[0] // 2, :, :, :]
        cover_cb = cb[data.shape[0] // 2:, :, :, :]
        secret_cb = cb[:data.shape[0] // 2, :, :, :]
        cover_cr = cr[data.shape[0] // 2:, :, :, :]
        secret_cr = cr[:data.shape[0] // 2, :, :, :]

        cover_input_y = dct(cover_y)
        secret_input_y = dct(secret_y)
        input_img_y = torch.cat((cover_input_y, secret_input_y), 1)

        cover_input_cb = dct(cover_cb)
        secret_input_cb = dct(secret_cb)
        input_img_cb = torch.cat((cover_input_cb, secret_input_cb), 1)

        cover_input_cr = dct(cover_cr)
        secret_input_cr = dct(secret_cr)
        input_img_cr = torch.cat((cover_input_cr, secret_input_cr), 1)

        #################
        #    forward:   #
        #################
        output_y = net(input_img_y)
        output_steg_y = output_y.narrow(1, 0, c.channel_dct * c.channels_in)
        output_z_y = output_y.narrow(1, c.channel_dct * c.channels_in, output_y.shape[1] - c.channel_dct * c.channels_in)
        steg_img_y = dct.inverse(output_steg_y)
        backward_z_y = gauss_noise(output_z_y.shape)

        output_cb = netCb(input_img_cb)
        output_steg_cb = output_cb.narrow(1, 0, c.channel_dct * c.channels_in)
        output_z_cb = output_cb.narrow(1, c.channel_dct * c.channels_in, output_cb.shape[1] - c.channel_dct * c.channels_in)
        steg_img_cb = dct.inverse(output_steg_cb)
        backward_z_cb = gauss_noise(output_z_cb.shape)

        output_cr = netCb(input_img_cr)
        output_steg_cr = output_cr.narrow(1, 0, c.channel_dct * c.channels_in)
        output_z_cr = output_cr.narrow(1, c.channel_dct * c.channels_in, output_cr.shape[1] - c.channel_dct * c.channels_in)
        steg_img_cr = dct.inverse(output_steg_cr)
        backward_z_cr = gauss_noise(output_z_cr.shape)

        ##############
        #    JPEG:   #
        ##############
        steg_img = upsampling(steg_img_y, steg_img_cb, steg_img_cr)
        # steg_img = rgb(steg_img)
        steg_img1 = rgb(steg_img)

        steg_img1 = steg_img1 * 255.0
        # steg_img = steg_img.expand(-1, 3, -1, -1)
        steg_img1 = jpg(steg_img1)
        # jpg.set_quality(50)
        # steg_img1 = jpg(steg_img1)
        # steg_img = torch.mean(steg_img, dim=1, keepdim=True)
        steg_img1 = steg_img1 / 255.0

        steg_img1 = ycbcr(steg_img1)
        steg_img_y, steg_img_cb, steg_img_cr = subsampling(steg_img1)
        steg_img = rgb(steg_img1)    # for saving as rgb image

        ##################
        #   filtering:   #
        ##################

        '''steg_img = upsampling(steg_img_y, steg_img_cb, steg_img_cr)
        # steg_img = rgb(steg_img)

        # print(steg_img.shape)
        steg_img_output = F.interpolate(steg_img, size=(256, 256), mode='bilinear', align_corners=False)
        # print(steg_img_output.shape)
        steg_img1 = F.interpolate(steg_img_output, size=(512, 512), mode='bilinear', align_corners=False)
        
        # steg_img1 = ycbcr(steg_img1)
        steg_img_y, steg_img_cb, steg_img_cr = subsampling(steg_img1)
        steg_img = rgb(steg_img1)'''

        ####################
        #   centor crop:   #
        ####################
        '''steg_img = upsampling(steg_img_y, steg_img_cb, steg_img_cr)
        steg_img1 = rgb(steg_img)

        crop_height = 200
        crop_width = 200

        start_height = (steg_img.size(2) - crop_height) // 2
        start_width = (steg_img.size(3) - crop_width) // 2

        # if i == 0:
        crop_tensor = steg_img1[:, :, start_height:start_height + crop_height, start_width:start_width + crop_width]
        cover_y = cover_y[:, :, start_height:start_height + crop_height, start_width:start_width + crop_width]
        secret_y = secret_y[:, :, start_height:start_height + crop_height, start_width:start_width + crop_width]
        padding_top = start_height
        padding_bottom = steg_img.size(2) - (start_height + crop_height)
        padding_left = start_width
        padding_right = steg_img.size(3) - (start_width + crop_width)
        steg_img1 = F.pad(crop_tensor, (padding_left, padding_right, padding_top, padding_bottom))
        cover_y = F.pad(cover_y, (padding_left, padding_right, padding_top, padding_bottom))
        secret_y = F.pad(secret_y, (padding_left, padding_right, padding_top, padding_bottom))
        # torchvision.utils.save_image(padding_tensor, './output.png')
        steg_img1[:, :, start_height:start_height + crop_height, start_width:start_width + crop_width] = 0
        cover_y[:, :, start_height:start_height + crop_height, start_width:start_width + crop_width] = 0
        secret_y[:, :, start_height:start_height + crop_height, start_width:start_width + crop_width] = 0
        steg_img1 = ycbcr(steg_img1)
        steg_img_y, steg_img_cb, steg_img_cr = subsampling(steg_img1)'''

        #####################
        #   quantization:   #
        #####################
        '''output_steg = jpeg(output_steg)
        output_steg = jpeg.inverse(output_steg)'''


        #################
        #   backward:   #
        #################
        output_steg_y = dct(steg_img_y)
        output_rev_y = torch.cat((output_steg_y, backward_z_y), 1)
        bacward_img_y = net(output_rev_y, rev=True)
        secret_rev_y = bacward_img_y.narrow(1, c.channel_dct * c.channels_in, bacward_img_y.shape[1] - c.channel_dct * c.channels_in)
        secret_rev_y = dct.inverse(secret_rev_y)
        cover_rev_y = bacward_img_y.narrow(1, 0, c.channel_dct * c.channels_in)
        cover_rev_y = dct.inverse(cover_rev_y)

        output_steg_cb = dct(steg_img_cb)
        output_rev_cb = torch.cat((output_steg_cb, backward_z_cb), 1)
        bacward_img_cb = netCb(output_rev_cb, rev=True)
        secret_rev_cb = bacward_img_cb.narrow(1, c.channel_dct * c.channels_in, bacward_img_cb.shape[1] - c.channel_dct * c.channels_in)
        secret_rev_cb = dct.inverse(secret_rev_cb)
        cover_rev_cb = bacward_img_cb.narrow(1, 0, c.channel_dct * c.channels_in)
        cover_rev_cb = dct.inverse(cover_rev_cb)

        output_steg_cr = dct(steg_img_cr)
        output_rev_cr = torch.cat((output_steg_cr, backward_z_cr), 1)
        bacward_img_cr = netCb(output_rev_cr, rev=True)
        secret_rev_cr = bacward_img_cr.narrow(1, c.channel_dct * c.channels_in, bacward_img_cr.shape[1] - c.channel_dct * c.channels_in)
        secret_rev_cr = dct.inverse(secret_rev_cr)
        cover_rev_cr = bacward_img_cr.narrow(1, 0, c.channel_dct * c.channels_in)
        cover_rev_cr = dct.inverse(cover_rev_cr)
        #resi_cover = (steg_img - cover_y) * 20
        #resi_secret = (secret_rev - secret_y) * 20

        #steg_img = torch.cat((steg_img, cover[:, 1:, :, :]), dim=1)
        secret_rev = upsampling(secret_rev_y, secret_rev_cb, secret_rev_cr)
        cover = rgb(cover)
        secret = rgb(secret)
        secret_rev = rgb(secret_rev)
        steg_img1 = rgb(steg_img1)

        cover_rev = upsampling(cover_rev_y, cover_rev_cb, cover_rev_cr)
        cover_rev = rgb(cover_rev)

        torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i)
        torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
        torchvision.utils.save_image(steg_img1, c.IMAGE_PATH_steg + '%.5d.png' % i)
        torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i)
        torchvision.utils.save_image(cover_rev, c.IMAGE_PATH_backward + '%.5d.png' % i)

        cover_y = cover_y.cpu().numpy().squeeze() * 255.0
        np.clip(cover_y, 0, 255)
        steg_img_y = steg_img_y.cpu().numpy().squeeze() * 255.0
        np.clip(steg_img_y, 0, 255)
        psnr_tmp = computePSNR(cover_y, steg_img_y)
        psnr_c.append(psnr_tmp)

        cover_rev_y = cover_rev_y.cpu().numpy().squeeze() * 255.0
        np.clip(cover_rev_y, 0, 255)
        psnr_tmp_1 = computePSNR(cover_y, cover_rev_y)
        psnr_cc.append(psnr_tmp_1)

        secret_y = secret_y.cpu().numpy().squeeze() * 255.0
        np.clip(secret_y, 0, 255)
        secret_rev_y = secret_rev_y.cpu().numpy().squeeze() * 255.0
        np.clip(secret_rev_y, 0, 255)
        psnr_tmp_s = computePSNR(secret_y, secret_rev_y)
        psnr_s.append(psnr_tmp_s)
        print(psnr_tmp, psnr_tmp_s, psnr_tmp_1)
        
        
    print(np.mean(psnr_c), np.mean(psnr_s), np.mean(psnr_cc))


