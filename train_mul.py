#!/usr/bin/env python
import torch
import torch.nn
import torch.optim
import math
import numpy as np
from model import *
import config as c
from tensorboardX import SummaryWriter
import datasets
import viz
import modules.Unet_common as common
import warnings
from dct2d import Dct2d
import torchvision
from Quantization import Quantization
from Subsample import chroma_subsampling, rgb_to_ycbcr_jpeg
import time

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)

'''def DC_coefficient_loss(steg_DC, cover_DC):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(steg_DC, cover_DC)
    return loss.to(device)'''

# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


#####################
# Model initialize: #
#####################
net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
para = get_parameter_number(net)
print(para)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

optim = torch.optim.AdamW(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

# DCT function
dct = Dct2d()

# Quantization
jpeg = Quantization()
jpeg.set_quality(90)

# Cb or Cr subsampling
subsampling = chroma_subsampling()
ycbcr = rgb_to_ycbcr_jpeg()

if c.tain_next:
    load(c.MODEL_PATH + c.suffix)
optim = torch.optim.AdamW(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

try:
    writer = SummaryWriter(comment='hinet', filename_suffix="steg")

    for i_epoch in range(c.epochs):
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []
        
        #################
        #     train:    #
        #################

        for i_batch, data in enumerate(datasets.trainloader):
            # start = time.time()
            data = data.to(device)
            
            # using the coloring change of DiffJPEG, and take the Y channel of the image
            data = ycbcr(data)          
            data, cb, cr = subsampling(data)

            cover = data[data.shape[0] // 3 * c.num_hiding_images:]
            secret = data[:data.shape[0] // 3 * c.num_hiding_images]
            
            cover_input = dct(cover)
            secret_input = dct(secret)

            input_secret = torch.cat((secret_input[0:secret_input.shape[0] // 2], secret_input[secret_input.shape[0] // 2:]), 1)

            input_img = torch.cat((cover_input, input_secret), 1)

            #################
            #    forward:   #
            #################
            output = net(input_img)
            output_steg = output.narrow(1, 0, c.channel_dct * c.channels_in)
            output_z = output.narrow(1, c.channel_dct * c.channels_in, output.shape[1] - c.channel_dct * c.channels_in)
            # steg_img = dct.inverse(output_steg)
            
            ######################
            #    quantization:   #
            ######################
            coeff = jpeg(output_steg)
            output_steg_q = jpeg.inverse(coeff)
            '''if i_epoch > 500:
                output_steg_q = jpeg(output_steg)
                output_steg_q = jpeg.inverse(output_steg_q)
            else:
                output_steg_q = output_steg'''

            #################
            #   backward:   #
            #################
            # what if I am not using random guassian noise of output_z instead of directly using output_z
            output_z_guass = gauss_noise(output_z.shape)

            output_rev = torch.cat((output_steg_q, output_z_guass), 1)
            output_image = net(output_rev, rev=True)

            secret_rev = output_image.narrow(1, c.channel_dct * c.channels_in, output_image.shape[1] - c.channel_dct * c.channels_in)
            # secret_rev = dct.inverse(secret_rev)
            #################
            #     loss:     #
            #################
            g_loss = guide_loss(output_steg.cuda(), cover_input.cuda())
            r_loss = reconstruction_loss(secret_rev.cuda(), input_secret.cuda())
            steg_low = output_steg.narrow(1, 0, c.channels_in)
            cover_low = cover_input.narrow(1, 0, c.channels_in)
            l_loss = low_frequency_loss(steg_low, cover_low)
            '''N, k, blocksize, blocksize = output_steg.shape
            steg_DC = output_steg[:, :, blocksize // 2, blocksize // 2]
            N, k, blocksize, blocksize = cover_input.shape
            cover_DC = cover_input[:, :, blocksize // 2, blocksize // 2]
            l_loss = DC_coefficient_loss(steg_DC, cover_DC)'''

            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])
            # end = time.time()
            # print(end - start)
        
        epoch_losses = np.mean(np.array(loss_history), axis=0)
        epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])

        #################
        #     val:      #
        #################
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                psnr_s = []
                psnr_c = []
                net.eval()
                # start = time.time()
                for i, x in enumerate(datasets.testloader):
                    x = x.to(device)

                    x = ycbcr(x)
                    x, cb, cr = subsampling(x)

                    cover = x[x.shape[0] // 3 * c.num_hiding_images:, :, :, :]
                    secret = x[:x.shape[0] // 3 * c.num_hiding_images, :, :, :]

                    cover_input = dct(cover)
                    secret_input = dct(secret)

                    input_secret = torch.cat((secret_input[0:secret_input.shape[0] // 2], secret_input[secret_input.shape[0] // 2:]), 1)

                    input_img = torch.cat((cover_input, input_secret), 1)

                    #################
                    #    forward:   #
                    #################

                    # 16 is a magic number, it means numbers of channel after dct
                    # so cover_input and secret_input has 16 channels, and the size of them is 1 x 1 (gray scale image) 
                    output = net(input_img)
                    output_steg = output.narrow(1, 0, c.channel_dct * c.channels_in)
                    steg = dct.inverse(output_steg)
                    output_z = output.narrow(1, c.channel_dct * c.channels_in, output.shape[1] - c.channel_dct * c.channels_in)
                    output_z = gauss_noise(output_z.shape)

                    #################
                    #   backward:   #
                    #################
                    output_steg = output_steg.cuda()
                    output_rev = torch.cat((output_steg, output_z), 1)
                    output_image = net(output_rev, rev=True)
                    secret_rev = output_image.narrow(1, c.channel_dct * c.channels_in, output_image.shape[1] - c.channel_dct * c.channels_in)
                    secret_rev = torch.cat((secret_rev[:, 0:secret_rev.shape[1]//2, :, :], secret_rev[:, secret_rev.shape[1]//2:, :, :]), 0)
                    secret_rev = dct.inverse(secret_rev)

                    torchvision.utils.save_image(cover, c.IMAGE_PATH_cover  + '%.5d.png' % i)
                    torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i)
                    torchvision.utils.save_image(steg, c.IMAGE_PATH_steg + '%.5d.png' % i)
                    torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i)

                    secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret = secret.cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    steg = steg.cpu().numpy().squeeze() * 255
                    np.clip(steg, 0, 255)
                    psnr_temp = computePSNR(secret_rev, secret)
                    psnr_s.append(psnr_temp)
                    psnr_temp_c = computePSNR(cover, steg)
                    psnr_c.append(psnr_temp_c)
                # end = time.time()
                # print(end - start)
                writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, i_epoch)
                writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)

        viz.show_loss(epoch_losses)
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)

        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i' % i_epoch + '.pt')
        weight_scheduler.step()

    torch.save({'opt': optim.state_dict(),
                'net': net.state_dict()}, c.MODEL_PATH + 'model' + '.pt')
    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, c.MODEL_PATH + 'model_ABORT' + '.pt')
    raise

finally:
    viz.signal_stop()