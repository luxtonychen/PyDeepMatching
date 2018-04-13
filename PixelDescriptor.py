import skimage
import numpy as np
import matplotlib.pyplot as plt

class PixelDescriptor(object):

    def __init__(self, presmooth_sigma=1, mid_smoothing=1,
                 post_smoothing=1, hog_sigmoid=0.2, ninth_dim=0.3,
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

        hog = self._non_linear(hog, self.hog_sigmoid)
        
        hog = self._add_ninth_dim(hog, self.ninth_dim)

        if self.norm_pixels:
            hog = self._norm(hog)

        return hog


    def _smooth(self, img, sigma):
        return skimage.filters.gaussian(img, sigma=sigma)


    def _get_gradient(self, img):
        pass

    def _get_hog(self, grad):
        pass

    def _non_linear(self, img, hog_sigmoid):
        pass

    def _add_ninth_dim(self, hog, constant):
        pass

    def _norm(self, hog):
        pass
    

