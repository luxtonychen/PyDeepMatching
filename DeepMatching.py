import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class DeepMatching(object):

    #nt : input numpy array like output pytorch tensor type
    #tn : input pytorch tensor type output numpy array like

    def __init__(self, descriptor, atomic_filter_size=4):
        self.descriptor = descriptor
        self.atomic_filter_size = atomic_filter_size

    def DeepMatching(self, img_base, img_search):
        des_base = self.descriptor.GetDescriptor(img_base)
        des_search = self.descriptor.GetDescriptor(img_search)

        filters, _ = self._nt_atomic_filter(des_base, self.atomic_filter_size)
        activation_map = self._t_grt_correlation_map(des_search, filters)

        return des_base, filters, activation_map


    def _nt_atomic_filter(self, img, filter_size):
        x_filters = int(img.shape[0]/filter_size)
        y_filters = int(img.shape[1]/filter_size)
        n_filters = x_filters*y_filters
        img_t = np.transpose(img, (2, 0, 1))
        filters = torch.DoubleTensor(n_filters, img_t.shape[0], filter_size, filter_size).zero_()
        filter_pos = []
        
        for x in range(x_filters):
            for y in range(y_filters):
                filter_pos.append((x, y))
                i = x*y_filters + y
                slice_img = img_t[:,x*filter_size:(x+1)*filter_size, y*filter_size:(y+1)*filter_size]
                
                #for j in range(slice_img.shape[0]):
                #    slice_img[j] = np.flipud(np.fliplr(slice_img[j]))

                filters[i,:,:,:] = torch.from_numpy(slice_img)

        return filters, filter_pos

    def _t_grt_correlation_map(self, img, filters):
        image = np.transpose(img, (2, 0, 1))
        image = torch.from_numpy(image)
        tensor = image.clone()
        tensor.resize_(1, image.shape[0], image.shape[1], image.shape[2])
        correlation_map = F.conv2d(Variable(tensor), Variable(filters), padding=1)
        return correlation_map.data


    