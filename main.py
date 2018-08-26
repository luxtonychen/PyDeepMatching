from sys import argv
import skimage
from skimage import io
from skimage import transform
import numpy as np 
import argparse
from construct_response import construct_response
from construct_pyramid import construct_pyramid
from matching import matching

def load_img(path, scale=None):
    img = io.imread(path)
    if img.ndim > 2:
        if img.shape[-1] == 3:
            img = skimage.color.rgb2gray(img)
        elif img.shape[-1] == 4:
            img = skimage.color.rgba2rgb(img)
            img = skimage.color.rgb2gray(img)
        else:
            print('can not handle color space of input img')
    if scale:
        if type(scale) == float:
            img = transform.rescale(img, (scale, scale))
        elif len(scale) == 2:
            img = transform.rescale(img, scale)
        else:
            print('Rescale: Too many paraments')
            raise ValueError
    print('Load img:', path, ', shape:', img.shape)
    return img

def viz_form(p, scale):
    return str(p[0][0]*scale) + ' ' + str(p[0][1]*scale) + ' ' + str(p[1][0]*scale) + ' ' + str(p[1][1]*scale) + ' ' + str(p[2]) + ' 1' 

def format_output(match_points, scale_factor, file_path = None):
    if file_path:
        with open(file_path, 'w') as f:
            for p in match_points:
                f.write(viz_form(p, scale_factor)+'\n')
    else:
        for p in match_points:
            print(viz_form(p, scale_factor))

def argument(argv):
    parser = argparse.ArgumentParser(prog='pyDeepMatching', usage='%(prog)s [options]')
    parser.add_argument('img', nargs=2)
    parser.add_argument('--rescale', '-r', nargs=1, help='Rescale image')
    parser.add_argument('-f', nargs=1, help='Output final corresponds to file')
    return parser.parse_args(argv[1:])

def main():
    arg = argument(argv)
    if arg.rescale:
        scale = float(arg.rescale[0])
        scale_factor = int(1/scale)
    else:
        scale = 1
        scale_factor = 1

    if arg.f:
        file_path = arg.f[0]
    else:
        file_path = None

    img1 = load_img(arg.img[0], scale)
    img2 = load_img(arg.img[1], scale)
    response = construct_response(img1, img2)
    pyramid = construct_pyramid(response, max(img2.shape), kernel_size=8)
    del(response)
    match_points = matching(pyramid, top_n=4000)
    format_output(match_points, scale_factor, file_path)

if __name__ == '__main__':
    main()

