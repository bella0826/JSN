import numpy as np
import torch
import torch_dct as dct
import torch.nn as nn
import torch.nn.functional as F
import config as c
import datasets
import torchvision
import cv2

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
        
        # Precompute Haar-like DCT weight matrix
        A = np.zeros((blocksize,blocksize))
        for i in range(blocksize):
            c_i = 1/np.sqrt(2) if i == 0 else 1.
            for n in range(blocksize):
                A[i,n] = np.sqrt(2/blocksize) * c_i * np.cos((2*n+ 1)/(blocksize*2) * i * np.pi)
        # set up conv layer
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32, device=device), requires_grad=False)
        self.unfold = torch.nn.Unfold(kernel_size=blocksize, padding=0, stride=self.stride)
        #self.A = self.A.to(device)

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
        dct_coefficients = torch.zeros_like(x)

        # Iterate through the batch of images
        for batch_idx in range(x.shape[0]):
            image = x[batch_idx, 0, :, :]  # Extract a single image from the batch
            
        # Iterate through the image in 128x128 blocks and apply DCT
            for y in range(0, image.shape[0], self.blocksize):
                for i in range(0, image.shape[1], self.blocksize):
            # Extract a 128x128 block
                    block = image[y:y+self.blocksize, i:i+self.blocksize]

            # Apply 2D DCT to the block
                    dct_block = dct.dct_2d(block, norm='ortho')

            # Store the DCT coefficients in the corresponding region of the result
                    dct_coefficients[batch_idx, 0, y:y+self.blocksize, i:i+self.blocksize] = dct_block
        return dct_coefficients
    
    def inverse(self, x, output_shape=(c.cropsize, c.cropsize)):
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
        reconstructed_images = torch.zeros_like(x)

        # Iterate through the batch of DCT coefficients
        for batch_idx in range(x.shape[0]):
            dct_batch = x[batch_idx, 0, :, :]  # Extract a single DCT coefficients batch

            # Initialize an array to store the reconstructed image
            reconstructed_image = torch.zeros((256, 256))

            # Iterate through the batch of DCT coefficients in 128x128 blocks and apply IDCT
            for y in range(0, dct_batch.shape[0], self.blocksize):
                for i in range(0, dct_batch.shape[1], self.blocksize):
                    # Extract a 128x128 DCT coefficient block
                    dct_block = dct_batch[y:y+self.blocksize, i:i+self.blocksize]

                    # Apply 2D IDCT to the DCT coefficient block
                    block = dct.idct_2d(dct_block, norm='ortho')

                    # Store the reconstructed block in the corresponding region of the result
                    reconstructed_image[y:y+self.blocksize, i:i+self.blocksize] = block

    # Store the reconstructed image in the batch
            reconstructed_images[batch_idx, 0, :, :] = reconstructed_image

        return reconstructed_images
    
if __name__ == "__main__":
    DCT = Dct2d()
    for i, data in enumerate(datasets.testloader):
        data = data.to(device)
        #print(data.device)
        coeff = DCT(data)
        print(coeff.shape)
        img = DCT.inverse(coeff)
        torchvision.utils.save_image(coeff, 'hii.png')
        torchvision.utils.save_image(img, 'hi.png')
        if i == 0:
            break

