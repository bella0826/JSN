import numpy as np
import torch
import torch_dct as dct
import torch.nn as nn
import torch.nn.functional as F
import config as c
import datasets
import torchvision
import cv2
import modules.Unet_common as common
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Dct2d(nn.Module):
    """
    Blockwise 2D Haar-like DCT
    """
    def __init__(self, blocksize=c.blocksize_dct, interleaving=False):
        """
        Parameters:
        blocksize: int, size of the blocks for discrete cosine transform 
        interleaving: bool, should the blocks interleave?
        """
        super().__init__()  # Call the super constructor
        
        self.blocksize = blocksize
        self.interleaving = interleaving
        
        if interleaving:
            self.stride = self.blocksize // 2
        else:
            self.stride = self.blocksize
        
        '''# Precompute Haar-like DCT weight matrix
        A = np.zeros((blocksize,blocksize))
        for i in range(blocksize):
            c_i = 1/np.sqrt(2) if i == 0 else 1.
            for n in range(blocksize):
                A[i,n] = np.sqrt(2/blocksize) * c_i * np.cos((2*n+ 1)/(blocksize*2) * i * np.pi)
        # set up conv layer
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32, device=device), requires_grad=False)
        self.unfold = torch.nn.Unfold(kernel_size=blocksize, padding=0, stride=self.stride)
        #self.A = self.A.to(device)'''

        return
        
    def forward(self, x):
        """
        Performs 2D blockwise Haar-like DCT
        
        Parameters:
        x: tensor of dimension (N, 1, h, w)
        
        Return:
        tensor of dimension (N, k, blocksize, blocksize)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block DCT coefficients
        """
        '''(N, C, H, W) = x.shape
        print(x.shape)
        assert (C == 1), "DCT is only implemented for a single channel"
        assert (H >= self.blocksize), "Input too small for blocksize"
        assert (W >= self.blocksize), "Input too small for blocksize"
        assert (H % self.stride == 0) and (W % self.stride == 0), "FFT is only for dimensions divisible by the blocksize"
        # unfold to blocks
        x = self.unfold(x)

        # Now shape (N, blocksize**2, k)
        (N, _, k) = x.shape
        x = x.view(-1, self.blocksize, self.blocksize, k).permute(0, 3, 1, 2)
        
        # Perform Haar-like DCT
        coeff = self.A.matmul(x).matmul(self.A.transpose(0, 1))
        print(coeff.shape)
        return coeff'''
        # Initialize an array to store the DCT coefficients
        total = torch.zeros((x.shape[0], c.channel_dct, x.shape[2]//self.blocksize, x.shape[3]//self.blocksize), device=device)
        # Iterate through the batch of images
        for batch_idx in range(x.shape[0]):
            image = x[batch_idx, 0, :, :]  # Extract a single image from the batch
            dct_blocks = []
        # Iterate through the image in 128x128 blocks and apply DCT
            for i in range(0, image.shape[0], self.blocksize):
                for j in range(0, image.shape[1], self.blocksize):
            # Extract a 128x128 block
                    block = image[i:i+self.blocksize, j:j+self.blocksize]
            # Apply 2D DCT to the block
                    dct_block = dct.dct_2d(block, 'ortho')
                    dct_blocks.append(dct_block.contiguous().view(1, c.channel_dct, 1, 1))

            total[batch_idx] = torch.cat(dct_blocks, dim=2).view(1, -1, x.shape[2]//self.blocksize, x.shape[3]//self.blocksize)
        return total
    
    def inverse(self, coeff, output_shape=(c.cropsize_val, c.cropsize_val)):
        """
        Performs 2D blockwise inverse Haar-like DCT
        
        Parameters:
        coeff: tensor of dimension (N, k, blocksize, blocksize)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block DCT coefficients
        output_shape: (h, w) dimensions of the reconstructed image
        
        Return:
        tensor of dimension (N, 1, h, w)
        """
        '''if self.interleaving:
            raise Exception('Inverse block DCT is not implemented for interleaving blocks!')

        # Perform inverse Haar-like DCT
        x = self.A.transpose(0, 1).matmul(coeff).matmul(self.A)
        (N, k, _, _) = x.shape
        
        x = x.permute(0, 2, 3, 1).view(-1, self.blocksize**2, k)
        x = F.fold(x, output_size=(output_shape[-2], output_shape[-1]), kernel_size=self.blocksize, padding=0, stride=self.blocksize)
        return x'''
        reconstruct_shape = (coeff.shape[2]*self.blocksize, coeff.shape[3]*self.blocksize)
        reconstructed_images = torch.zeros((coeff.shape[0], c.channels_in, reconstruct_shape[0], reconstruct_shape[1]), device=device)

        # Iterate through the batch of DCT coefficients
        for batch_idx in range(coeff.shape[0]):

            # Iterate through the batch of DCT coefficients in 128x128 blocks and apply IDCT
            for i in range(0, reconstruct_shape[0], self.blocksize):
                for j in range(0, reconstruct_shape[1], self.blocksize):
                    block = torch.zeros((self.blocksize, self.blocksize), device=device)
                    # Apply 2D IDCT to the DCT coefficient block
                    
                    index = torch.arange(c.channel_dct, device=device)
                    block.view(-1).index_add_(0, index, coeff[batch_idx, :, i//self.blocksize, j//self.blocksize])
                    '''block[0, 0] = coeff[batch_idx, 0, i//self.blocksize, j//self.blocksize]
                    block[0, 1] = coeff[batch_idx, 1, i//self.blocksize, j//self.blocksize]
                    block[1, 0] = coeff[batch_idx, 2, i//self.blocksize, j//self.blocksize]
                    block[1, 1] = coeff[batch_idx, 3, i//self.blocksize, j//self.blocksize]'''

                    dct_block = dct.idct_2d(block, 'ortho')
                    
                    reconstructed_images[batch_idx, 0, i:i+self.blocksize, j:j+self.blocksize] = dct_block
            
        return reconstructed_images
    

y_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32).T
y_table = nn.Parameter(torch.from_numpy(y_table))

def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.

class Quantization(nn.Module):
    def __init__(self, quality=90):
        super(Quantization, self).__init__()
        self.y_table = y_table 
    def forward(self, x):
        ans = x.float() / (self.y_table )
        ans = torch.round(ans)
        return ans
    def inverse(self, x):
        ans = x.float() * (self.y_table )
        return ans
    def set_quality(self, quality):
        self.factor = quality_to_factor(quality)


if __name__ == "__main__":
    DCT = Dct2d()

    for i, data in enumerate(datasets.testloader):
        data = data.to(device)
        #print(data.device)
        coeff = DCT(data)
        print(coeff.shape)
        img = DCT.inverse(coeff)
        # torchvision.utils.save_image(coeff, 'hii.png')
        torchvision.utils.save_image(img, 'hi.png')
        if i == 0:
            break
    
