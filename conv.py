import numpy as np
from numpy.lib.stride_tricks import as_strided

def slice_img(img, slice_shape):
    img_h, img_w = img.shape[1:]
    n_filters_h, n_filters_w = int(img_h/slice_shape[0]), int(img_w/slice_shape[1])
    filters_shape = (img.shape[0], n_filters_h, n_filters_w) + slice_shape
    isize = img.itemsize
    strides = (img_h*img_w*isize, slice_shape[0]*img_w*isize, slice_shape[1]*isize, img_w*isize, isize)
    return as_strided(img, filters_shape, strides).reshape((filters_shape[0], filters_shape[1]*filters_shape[2], filters_shape[3]*filters_shape[4]))


def slice_img_loop(img, slice_shape):
    img_h, img_w = img.shape[1:]
    n_filters_h, n_filters_w = int(img_h/slice_shape[0]), int(img_w/slice_shape[1])
    filters_shape = (n_filters_h*n_filters_w, img.shape[0]) + slice_shape
    filters = np.zeros(filters_shape)
    i = 0
    for h in range(n_filters_h):
        h0 = h*slice_shape[0]
        for w in range(n_filters_w):
            w0 = w*slice_shape[1]
            #i = h0*n_filters_w + w0
            filters[i, :, :, :] = img[:, h0:h0+slice_shape[0], w0:w0+slice_shape[1]]
            i+=1
    return filters    


def batch_conv(img, filters, filter_size, padding):
    img_h, img_w = img.shape[1], img.shape[2]
    padded_shape = (img.shape[0], img_h+2*padding[0], img_w+2*padding[1])
    padded_img = np.zeros(padded_shape)
    padded_img[:, padding[0]:-padding[0], padding[1]:-padding[1]] = img
    im = slide_window(padded_img, filter_size)
    #conv_img = np.dot(filters, im)
    conv_img = np.einsum('ijk, ikl->jl', filters, im)
    return conv_img

def slide_window(img, window_size):
    img_h, img_w = img.shape[1:]
    n_windows_h, n_windows_w = img_h-window_size[0]+1, img_w-window_size[1]+1
    out_shape = (img.shape[0], n_windows_h, n_windows_w) + window_size
    isize = img.itemsize
    strides = (img_h*img_w*isize, img_w*isize, isize, img_w*isize, isize)
    res = as_strided(img, out_shape, strides).reshape((out_shape[0], out_shape[1]*out_shape[2], out_shape[3]*out_shape[4]))
    return res.swapaxes(1, 2)

