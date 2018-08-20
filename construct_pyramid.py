from sparse_conv2 import sparse_conv2
import numpy as np 
import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
from collections import OrderedDict

def construct_pyramid(response_maps, upper_bound, kernel_size = 4):
    h, w = response_maps[0]
    maps = response_maps[1]
    maph, mapw = maps[0].shape

    pyramid_layer = {'size': (h, w), 'map_size': (maph, mapw) , 'response_maps':maps, 'weights': np.ones(h*w),
                     'kernel_size': kernel_size, 'step': 1, 'pool_idx': None, 'reverse_idx':None}
    pyramid = [pyramid_layer]

    while (pyramid[-1]['kernel_size']*2) <= upper_bound:
        next_layer = get_next_layer(pyramid[-1])
        pyramid.append(next_layer)
        print(pyramid[-1]['kernel_size']*2)

    return pyramid

def get_next_layer(layer):
    step = layer['step']*2 - 1 if layer['step'] > 1 else 2
    kernel_size = layer['kernel_size']*2
    
    maps = layer['response_maps']
    z, y, x = maps.shape
    t = np.zeros((1, z, y, x))
    t[0] = maps
    tensor_map = torch.from_numpy(t)
    pooled_map, pool_idx = F.max_pool2d(V(tensor_map), 3, stride=2, return_indices=True, ceil_mode=True, padding=1)
    maps = pooled_map.data.numpy()[0]
    layer['response_maps'] = maps # so that we could only store pooled map to save memory usage
    layer['pool_idx'] = pool_idx

    h, w, maps, weights, reverse_idx = sparse_conv2((layer['size'][0], layer['size'][1], maps), aggregation, layer['weights'], step)
    maph, mapw = maps[0].shape
    return {'size': (h, w), 'map_size': (maph, mapw), 'response_maps':maps, 'weights': weights, 'kernel_size': kernel_size, 
            'step': step, 'pool_idx': None, 'reverse_idx': reverse_idx}

def aggregation(imgs, weights):
    
    edge_correction_factor = get_weights_mat(weights, 0.9) #conv.cpp:_sparse_conv
    #next_layer = _aggregation_mat(imgs, weights)
    next_layer = _aggregation_slice(imgs, weights)

    return edge_correction(next_layer, edge_correction_factor)**1.4

def get_weights_mat(weights, inv):
    weights_mat = np.zeros((3, 3))
    #print(weights)
    weights_mat[1:, 1:] = weights[0] #top left
    weights_mat[1:, :-1] += weights[1] #top right
    weights_mat[:-1, 1:] += weights[2] #bottom left
    weights_mat[:-1, :-1] += weights[3] #bottom right
    for i in range(3):
        for j in range(3):
            x = weights_mat[i, j] 
            if x != 0:
                weights_mat[i, j] = (1/x)**(0.9)
    return weights_mat
    
def edge_correction(img, f):
    img[0,0] *= f[0,0]
    img[0,-1] *= f[0,-1]
    img[-1,0] *= f[-1,0]
    img[-1,-1] *= f[-1,-1]
    img[0,1:-1] *= f[0, 1]
    img[-1,1:-1] *= f[-1, 1]
    img[1:-1,0] *= f[1, 0]
    img[1:-1,-1] *= f[1, -1]
    img[2:-1,2:-1] *= f[1, 1]
    return img

def _aggregation_mat(imgs, weights):
    #print(imgs[0].shape)
    h, w = imgs[0].shape
    mask = np.zeros((4, h*w))
    mask[0, 0:(h-1)*w] = np.array([[weights[0] for n in range(w-1)]+[0] for m in range(h-1)]).flatten()
    mask[1, 0:(h-1)*w] = np.roll(np.array([[weights[1] for n in range(w-1)]+[0] for m in range(h-1)]).flatten(), 1)
    mask[2, w:h*w] = np.array([[weights[2] for n in range(w-1)]+[0] for m in range(h-1)]).flatten()
    mask[3, w:h*w] = np.roll(np.array([[weights[3] for n in range(w-1)]+[0] for m in range(h-1)]).flatten(), 1)
    im1 = np.roll(imgs[0].flatten()*mask[0], w+1)
    im2 = np.roll(imgs[1].flatten()*mask[1], w-1)
    im3 = np.roll(imgs[2].flatten()*mask[2], -(w-1))
    im4 = np.roll(imgs[3].flatten()*mask[3], -(w+1))
    
    return np.reshape((im1+im2+im3+im4), (h, w))

def _aggregation_slice(imgs, wights):
    res = np.zeros(imgs[0].shape)
    
    res[1:, 1:] += imgs[0][:-1, :-1]*wights[0]
    res[1:, :-1] += imgs[1][:-1, 1:]*wights[1]
    res[:-1, 1:] += imgs[2][1:, :-1]*wights[2]
    res[:-1, :-1] += imgs[3][1:, 1:]*wights[3]
    
    return res

