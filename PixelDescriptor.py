import skimage
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

class PixelDescriptor(object):

    def __init__(self, presmooth_sigma=1, mid_smoothing=1,
                 post_smoothing=1, hog_sigmoid=0.2, ninth_dim=0,
                 norm_pixels = False):
            
            self.presmooth_sigma = presmooth_sigma
            self.mid_smoothing = mid_smoothing
            self.post_smoothing = post_smoothing
            self.hog_sigmoid = hog_sigmoid
            self.ninth_dim = ninth_dim
            self.norm_pixels = norm_pixels

    def GetDescriptor(self, img):

        if self.presmooth_sigma != 0:
            image = self._smooth(img, self.presmooth_sigma)
        else:
            image = img

        grad = self._get_gradient(image)

        if self.mid_smoothing != 0:
            grad = self._smooth(grad, self.mid_smoothing)

        hog = self._get_hog(grad)

        if self.post_smoothing != 0:
            hog = self._smooth(hog, self.post_smoothing)
#
        hog = self._non_linear(hog, self.hog_sigmoid)
        
        hog = self._add_ninth_dim(hog, self.ninth_dim)
#
        if self.norm_pixels:
            hog = self._norm(hog)
#
        return hog


    def _smooth(self, img, sigma):
        return skimage.filters.gaussian(img, sigma=sigma)


    def _get_gradient(self, img):
        kernel = np.array([[0, 0, 0], 
                           [-1, 0, 1], 
                           [0, 0, 0]])
        out = np.zeros((img.shape[0], img.shape[1], 2))
        out[:,:,0] = ndimage.convolve(img, kernel)
        out[:,:,1] = ndimage.convolve(img, kernel.T)

        return out

    def _get_hog(self, grad):
        hog = np.zeros((grad.shape[0], grad.shape[1], 9))
        num_ori = 8
        list_sin = [np.sin(((-2)*(i-2)*np.pi)/num_ori) for i in range(num_ori)]
        list_cos = [np.cos(((-2)*(i-2)*np.pi)/num_ori) for i in range(num_ori)]

        for i in range(num_ori):
            hog[:,:,i] = grad[:,:,0]*list_cos[i] +  grad[:,:,1]*list_sin[i]
        
        hog[hog<0] = 0

        return hog

    def _non_linear(self, hog, hog_sigmoid):
        
        sigmoid = lambda x: (2/(1+np.exp((-1)*hog_sigmoid*x)))-1
        
        return sigmoid(hog)


    def _add_ninth_dim(self, hog, constant):
        
        hog[:,:,8] = np.zeros((hog.shape[0], hog.shape[1])) + constant

        return hog

    def _norm(self, hog):
        pass
    

