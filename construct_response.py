import skimage.io as io
import numpy as np
import skimage
from skimage import filters
import torch 
import torch.nn.functional as F
import scipy.signal as sig
from torch.autograd import Variable as V
import matplotlib.pyplot as plt
import conv
from numpy.lib.stride_tricks import as_strided

def construct_response(img1, img2):
    img1 = smooth(img1)
    img2 = smooth(img2)

    img1 = skimage.img_as_ubyte(img1)
    img2 = skimage.img_as_ubyte(img2)

    img1 = gradient(img1)
    img2 = gradient(img2)

    response1 = hog(img1)
    response2 = hog(img2)

    response1 = smooth(response1)
    response2 = smooth(response2)

    response1 = nonlinear(response1)
    response2 = nonlinear(response2)

    response1 = smooth(response1)
    response2 = smooth(response2)
    response1 = add_ninth(response1)
    response2 = add_ninth(response2)

    response = np_get_response(response1, response2)

    return response

def gradient(img):
    kernel = np.array([[0, 0, 0], 
                       [1, 0, -1],
                       [0, 0, 0]])
    return_gradient = np.zeros((2, img.shape[0], img.shape[1]))
    return_gradient[0] = sig.convolve2d(img, kernel, mode='same', boundary='symm')
    return_gradient[1] = sig.convolve2d(img, kernel.transpose(), mode='same', boundary='symm')
    return return_gradient

def hog(img):
    hog = np.zeros((8, img.shape[1], img.shape[2]))
    num_ori = 8
    sin = np.array([np.sin(((-2)*(i-2)*np.pi)/num_ori) for i in range(num_ori)])
    cos = np.array([np.cos(((-2)*(i-2)*np.pi)/num_ori) for i in range(num_ori)])

    for i in range(num_ori):
        hog[i] = img[0]*sin[i] + img[1]*cos[i]
        hog[i][hog[i] < 0] = 0
    
    return hog

def nonlinear(hog, hog_sigmoid = 0.2):
        
    sigmoid = lambda x: (2/(1+np.exp((-1)*hog_sigmoid*x)))-1
        
    return sigmoid(hog)

def smooth(img, sigma=1):
    if len(img.shape) > 2:
        for i in range(img.shape[0]):
            img[i] = filters.gaussian(img[i], sigma=sigma)
    else:
        img = filters.gaussian(img, sigma=sigma)
    return img

def get_response(res1, res2, slice_shape = 8):
    img_h, img_w = res1.shape[1], res1.shape[2]
    high, width = int(img_h/slice_shape), int(img_w/slice_shape)
    filters = np.zeros((width*high, 9, slice_shape, slice_shape))
    norm_weight = np.ones((1, 9, int(slice_shape/2), int(slice_shape/2)))

    for h in range(high):
        for w in range(width):
            idx = h*width + w
            filters[idx] = res1[:, h*slice_shape:(h+1)*slice_shape, w*slice_shape:(w+1)*slice_shape]#.transpose((0, 2, 1))

    filter_tensor = V(torch.from_numpy(filters))
    img_tensor = V(torch.from_numpy(np.expand_dims(res2, axis=0)))
    norm_tensor = V(torch.from_numpy(norm_weight))

    filter_tensor = F.avg_pool2d(filter_tensor, 2, stride=2, padding=0)
    
    n = torch.pow(filter_tensor, 2)
    n = torch.sqrt(torch.sum(torch.sum(torch.sum(n, -1), -1), -1))
    
    for i in range(n.shape[0]):
        filter_tensor[i] /= n[i]
    
    img_tensor = F.avg_pool2d(img_tensor, 2, stride=2, padding=0)

    norm = F.conv2d(torch.pow(img_tensor, 2), norm_tensor, padding=int(slice_shape/4))
    norm = torch.sqrt(norm)

    res = F.conv2d(img_tensor, filter_tensor, padding=int(slice_shape/4))
    res = res.data.numpy()[0]
    norm = norm.data.numpy()[0, 0]
    for i in range(res.shape[0]):
        res[i] /= norm
    return (high, width), res

def np_get_response(res1, res2, slice_shape = 8):
    img_h, img_w = res1.shape[1], res1.shape[2]
    high, width = int(img_h/slice_shape), int(img_w/slice_shape)
    real_slice_shape = (int(slice_shape/2), int(slice_shape/2))
    
    ds_res1 = conv.average_pool(res1, (2, 2), stride=2)
    ds_res2 = conv.average_pool(res2, (2, 2), stride=2)
    filters = conv.slice_img(ds_res1, real_slice_shape)
    
    n = np.square(filters)
    n = np.sqrt(np.sum(np.sum(n, 0), -1))
    n = as_strided(n, filters.shape[1:], n.strides+(0, ))
    filters = filters/n

    #padding img
    padded_shape = (ds_res2.shape[0], ds_res2.shape[1]+real_slice_shape[0], ds_res2.shape[2]+real_slice_shape[1])
    padding = int(real_slice_shape[0]/2), int(real_slice_shape[1]/2), 
    norm = np.zeros(padded_shape)
    norm[:, padding[0]:-padding[0], padding[1]:-padding[1]] = ds_res2
    
    norm = np.square(norm)
    norm = conv.slide_window(norm, real_slice_shape)
    norm = np.sqrt(np.sum(np.sum(norm, 1), 0))

    res = conv.batch_conv(ds_res2, filters, real_slice_shape, padding)
    print(res.shape)
    res /= norm
    print(res.shape)
    res_shape = (res.shape[0], padded_shape[1]-real_slice_shape[0]+1, padded_shape[2]-real_slice_shape[1]+1)
    return (high, width), res.reshape(res_shape)


def add_ninth(hog, value = 0.3):
    return np.append(hog, np.full((1, hog.shape[1], hog.shape[2]), value), axis=0)



    
    
