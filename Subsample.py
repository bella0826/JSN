import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """

    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        #print(image.shape)
        #image_2 = image.permute(0, 3, 1, 2).clone()
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False)
        cb = avg_pool(image[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image[:, 2, :, :].unsqueeze(1))
        #print(cb.shape)
        #cb = cb.permute(0, 2, 3, 1)
        #cr = cr.permute(0, 2, 3, 1)
        
        return image[:, :1, :, :], cb, cr
    
class chroma_upsampling(nn.Module):
    """ Upsample chroma layers
    Input:
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Output:
        image(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            batch = x.shape[0]
            height, width = x.shape[2:4]
            # x = x.unsqueeze(-1)
            x = x.permute(0, 2, 3, 1)
            x = x.repeat(1, 1, k, k)
            x = x.view(batch, 1, height * k, width * k)
            return x
        cb = repeat(cb)
        cr = repeat(cr)
        return torch.cat([y, cb, cr], dim=1)
    
class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Output:
        result(tensor): batch x 3 x height x width
    """

    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()
        matrix = np.array([
            [1., 0., 1.402],
            [1, -0.344136, -0.714136],
            [1, 1.772, 0]
        ], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.])).to(device)
        self.matrix = nn.Parameter(torch.from_numpy(matrix)).to(device)

    def forward(self, image):
        image = image * 255.0
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        result.view(image.shape)
        result = result / 255.0
        return result.permute(0, 3, 1, 2)
    
class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Output:
        result(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]
        ], dtype=np.float32).T

        self.shift = nn.Parameter(torch.tensor([0., 128., 128.])).to(device)
        self.matrix = nn.Parameter(torch.from_numpy(matrix)).to(device)

    def forward(self, image):
        image = image * 255.0
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        result.view(image.shape)
        result = result / 255.0
        return result.permute(0, 3, 1, 2)