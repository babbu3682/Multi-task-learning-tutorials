import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

import random
import numpy as np
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


# Reference: https://github.com/MrGiovanni/ModelsGenesis/blob/dce8b488dd44863fad0e4f2188efc5b9329fcaf7/pytorch/Genesis_Chest_CT.py
# ex) training_generator = generate_pair(x_train,conf.batch_size, conf)

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x

    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)

    _, _, img_rows, img_cols = x.shape  # x.shape = (256, 256)
    num_block = 10000

    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        
        window = orig_image[:, :, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y]  # 0은 channel
        
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((x.shape[0], x.shape[1], block_noise_size_x, block_noise_size_y))
        image_temp[:, :, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y] = window
        
        # window     = window.flatten(start_dim=2)
        # random_idx = torch.randperm(window.shape[-1])
        # window     = window[..., random_idx].view((x.shape[0], x.shape[1], block_noise_size_x, block_noise_size_y))         
        # image_temp[:, :, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y] = window        

    return image_temp

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def image_in_painting(x):
    _, _, img_rows, img_cols = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        
        # x[:, :, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y] = torch.rand(x.shape[0], x.shape[1], block_noise_size_x, block_noise_size_y) * 1.0
        x[:, :, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y] = np.random.rand(x.shape[0], x.shape[1], block_noise_size_x, block_noise_size_y) * 1.0
        cnt -= 1
    return x

def image_out_painting(x):
    _, _, img_rows, img_cols = x.shape
    
    # image_temp = x.clone()    
    # x[...]     = torch.rand(x.shape)

    image_temp = copy.deepcopy(x)
    x          = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0

    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    
    x[:, :, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y] = image_temp[:, :, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y]
    cnt = 4

    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        x[:, :, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y] = image_temp[:, :, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y]
        cnt -= 1
    
    del image_temp

    return x




def distort_image(image, mask):
    image = image.cpu().numpy()
    mask  = mask.cpu().numpy().round()
    
    # Local Shuffle Pixel
    distort = local_pixel_shuffling(image, prob=0.7)  # input shape = (H, W)

    # Apply non-Linear transformation with an assigned probability
    distort = nonlinear_transformation(distort, prob=0.7)

    # Inpainting & Outpainting
    if random.random() < 0.9:  # paint_rate
        if random.random() < 0.7:  # inpaint_rate and outpaint_rate
            # Inpainting
            distort = image_in_painting(distort)
        else:
            # Outpainting
            distort = image_out_painting(distort)

    # distort
    image[np.where(mask)] = distort[np.where(mask)] 

    return torch.tensor(image).float().cuda()

def distort_image_v2(image, mask):
    image[torch.where(mask.round())] = 0.0
    return image.float()


def activation_map(last_feat):
    mean = torch.mean(last_feat, dim=1, keepdim=True)
    mean = torch.sigmoid(mean)                      # (B, C, H, W)
    mean = F.upsample(mean, size=(256, 256), mode='bicubic')
    # Min-Max Normalize
    mean -= mean.min()
    mean /= mean.max()
    return mean

class Noiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_map    = activation_map
        self.distortion = distort_image

    def forward(self, x, last_feat):
        noise_mask = self.act_map(last_feat)
        image      = self.distortion(x, noise_mask) # https://github.com/MrGiovanni/ModelsGenesis/tree/master/pytorch

        return image, noise_mask

