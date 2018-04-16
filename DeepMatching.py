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
        activation_map = self._t_get_correlation_map(des_search, filters)

        return des_base, filters, activation_map.cpu()


    def _nt_atomic_filter(self, img, filter_size):
        x_filters = int(img.shape[0]/filter_size)
        y_filters = int(img.shape[1]/filter_size)
        n_filters = x_filters*y_filters
        img_t = np.transpose(img, (2, 0, 1))
        filters = torch.FloatTensor(n_filters, img_t.shape[0], filter_size, filter_size).zero_()
        filter_pos = np.zeros((x_filters, y_filters))
        
        for x in range(x_filters):
            for y in range(y_filters):
                i = x*y_filters + y
                filter_pos[x,y] = i
                slice_img = img_t[:,x*filter_size:(x+1)*filter_size, y*filter_size:(y+1)*filter_size]
                
                #for j in range(slice_img.shape[0]):
                #    slice_img[j] = np.flipud(np.fliplr(slice_img[j]))

                filters[i,:,:,:] = torch.from_numpy(slice_img)

        return filters, filter_pos

    def _t_get_correlation_map(self, img, filters):
        image = np.transpose(img, (2, 0, 1))
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        tensor = image.clone()
        tensor.resize_(1, image.shape[0], image.shape[1], image.shape[2])
        correlation_map = F.conv2d(Variable(tensor.cuda()), Variable(filters.cuda()), padding=2)
        #correlation_map = F.conv2d(Variable(tensor), Variable(filters), padding=1)        
        return correlation_map.data

    def _t_patch_aggregation(self, correlation_map, filter_position, power_correct):
        pooled_map = F.max_pool2d(correlation_map, 3, stride=2, return_indices=True)
        pooled_map = pooled_map.cpu()
        for x in filter_position.shape[0]:
            for y in filter_position.shape[1]:
                sub_map_1 = pooled_map[0,filter_position[x, y],:,:]
                sub_map_2 = pooled_map[0,filter_position[x, y+1],:,:]
                sub_map_3 = pooled_map[0,filter_position[x+1, y],:,:]
                sub_map_4 = pooled_map[0,filter_position[x+1, y+1],:,:]

                map_size_x, map_size_y = sub_map_1.shape[0], sub_map_1.shape[1]
                sub_map = torch.FloatTensor(4, map_size_x-1, map_size_y-1)
                #shift
                sub_map[0] = sub_map_1[:map_size_x-1, :map_size_y-1]
                sub_map[1] = sub_map_2[:map_size_x-1, 1:]
                sub_map[2] = sub_map_3[1:, :map_size_y-1]
                sub_map[3] = sub_map_4[1:,1:]
                #average and recitification
                sub_map[0] = torch.sum(sub_map, 0)
                sub_map[0] = torch.div(sub_map[0], 0.25)
                sub_map[0] = torch.pow(sub_map[0], power_correct)




        


    