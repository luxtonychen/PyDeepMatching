import numpy as np


def get_entry(pyramid, top_n=100):
    entry = []
    top = np.sort(np.array(list(set(pyramid[-1]['response_maps'].flatten()))))[::-1][:top_n]
    for score in top:
        idx = np.array(np.where(pyramid[-1]['response_maps'] == score)).transpose()
        for i in idx:
            entry.append((i, score))
        
    return entry

def match_next(entry, current_level, next_level):
    entry_idx, entry_score = entry
    patch_idx = current_level['reverse_idx'][entry_idx[0]]
    pool_idx = current_level['pool_idx']
    current_response_size = current_level['response_maps'].shape[1], current_level['response_maps'].shape[2]
    next_response = next_level['response_maps']
    h, w = next_response.shape[1], next_response.shape[2]
    match_points = []
    for i in range(4):
        if patch_idx[i] != -1:
            if i == 0:  #top left
                max_idx = (entry_idx[1]-1, entry_idx[2]-1) 
            elif i == 1:    #top right
                max_idx = (entry_idx[1]-1, entry_idx[2]+1)
            elif i == 2:    #bottom left
                max_idx = (entry_idx[1]+1, entry_idx[2]-1)
            elif i == 3:    #bottom right
                max_idx = (entry_idx[1]+1, entry_idx[2]+1)
            
            if (max_idx[0] < 0) or (max_idx[1] < 0) or \
            (max_idx[0] >= current_response_size[0]) or (max_idx[1] >= current_response_size[1]):
                continue
            
            response_map = next_response[patch_idx[i]]
            origion_pos = pool_idx[0, patch_idx[i], max_idx[0], max_idx[1]]
            next_pos = int(origion_pos/w), int(origion_pos%w)
            next_score = response_map[next_pos] + entry_score
            next_point = ((patch_idx[i], next_pos[0], next_pos[1]), next_score)
            match_points.append(next_point)

    return match_points

def match_entry(entry, pyramid, dir=0):
    depth = len(pyramid)
    match_points = entry
    for i in range(1, depth):
        current_level = pyramid[depth-i]
        next_level = pyramid[depth-(i+1)]
        tmp_entry = []
        for e in match_points:
            tmp_entry.extend(match_next(e, current_level, next_level))
        #match_points = merge_points(tmp_entry, dir=dir)
        match_points = tmp_entry
    
    return match_points

def merge_points(match_points, dir=0, step=1):
    found = []
    merged_points = []
    for p in match_points:
        loc = lambda x: x[0][0] if dir == 0 else (int(x[0][1]/step), int(x[0][2]/step))
        if loc(p) in found:
            idx = found.index(loc(p))
            if merged_points[idx][1] < p[1]:
                merged_points[idx] = p
        else:
            found.append(loc(p))
            merged_points.append(p)
    return merged_points

def convert2coord(match_points, shape, kernel_size):
    h, w = shape
    match_points_c = []
    for p in match_points:
        x0, y0 = (p[0][0] % w), int(p[0][0]/w)
        x0, y0 = x0*kernel_size+int(kernel_size/2), y0*kernel_size+int(kernel_size/2)
        x1, y1 = p[0][2]*2, p[0][1]*2
        match_points_c.append(((x0, y0), (x1, y1), p[1]))
    return match_points_c


def matching(pyramid, top_n=-1):
    entry = get_entry(pyramid, top_n)
    match_points = match_entry(entry, pyramid, dir=0)
    match_points12 = sorted(merge_points(match_points, 0), key=lambda x: x[1], reverse=True)
    match_points21 = sorted(merge_points(match_points, 1, 4), key=lambda x: x[1], reverse=True)
    t = []
    score = lambda x: x[1]
    loc = lambda x: x[0]
    i21 = 0
    i12 = 0
    while i12 < len(match_points12):
        if score(match_points12[i12]) < score(match_points21[i21]):
            i21 += 1
        elif score(match_points12[i12]) > score(match_points21[i21]):
            i12 += 1
        else: 
            if loc(match_points12[i12]) == loc(match_points21[i21]):
                t.append(match_points12[i12])
                i12 += 1
                i21 += 1
            else:
                i12 += 1
    
    match_points = convert2coord(t, pyramid[0]['size'], pyramid[0]['kernel_size'])

    return match_points