import numpy as np

def sparse_conv2(data, reduce_func, weights, dilation):
    '''
    High order sparse convolution function, used to implement 2-deminational sparse, with normalization
    convolution with kernel size 2x2 and reduce_func
    Input:
        data: tuple with three elements (h, w, data). data is 1-d array with data.shape[0] = h*w and row-major order(C-Style)
        reduce_func: the function which will be applied on corresponding position
        weights: normalization weights, 1-d array with data.shape[0] = h*w
        dilation: Real kernel size
    Output:
        tuple with three elements (h, w, data). data is 1-d array with data.shape[0] = h*w and row-major order(C-Style)
    '''
    h, w, i_data = data[0], data[1], data[2]
    #print('resieve maps shape:', i_data.shape)
    
    if dilation != 2:
        step = int(dilation/2)
        o_data = np.zeros(i_data.shape)
        o_weights = np.zeros(i_data.shape[0])
        o_w, o_h = w, h
    else:
        step = 1
        shape = list(i_data.shape)
        shape[0] = (w+1)*(h+1)
        shape = tuple(shape)
        o_data = np.zeros(shape)
        o_weights = np.zeros(o_data.shape[0])
        o_w, o_h = w+1, h+1

    print(o_weights.shape)
    get_loc = lambda idx, offset: ((idx[0]+offset[0])*w + (idx[1]+offset[1])) if (h > (idx[0]+offset[0]) >= 0 and w > (idx[1]+offset[1]) >= 0 ) else -1
    for row in range(o_h):
        for column in range(o_w):
            if dilation != 2:
                loc =  [get_loc((row, column), (-step, -step)), 
                        get_loc((row, column), (-step, step)), 
                        get_loc((row, column), (step, -step)),
                        get_loc((row, column), (step, step))]
            else:
                loc =  [get_loc((row, column), (-step, -step)), 
                        get_loc((row, column), (-step, 0)), 
                        get_loc((row, column), (0, -step)),
                        get_loc((row, column), (0, 0))]
                        
            o_loc = row*o_w + column

            if len(i_data.shape) > 1:
                shape = list(i_data.shape)
                shape[0] = 4
                shape = tuple(shape)
                selected = np.zeros(shape)
            else:
                selected = np.zeros(4)
            
            reduce_weights = np.zeros(4)

            for i in range(4):
                if loc[i] != -1:
                    selected[i] = i_data[loc[i]]
                    reduce_weights[i] = weights[loc[i]]
            
            o_weights[o_loc] = np.sum(reduce_weights)

            o_data[o_loc] = reduce_func(selected, (reduce_weights/o_weights[o_loc]))
            #print(o_data[get_loc((row, column), (0, 0))])
    
    return o_h, o_w, o_data, o_weights
            