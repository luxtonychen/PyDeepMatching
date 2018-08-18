import numpy as np
import multiprocessing as mp
from timeit import default_timer
from functools import partial


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

    max_idx = [(patch_idx[0], (entry_idx[1]-1, entry_idx[2]-1)),
               (patch_idx[1], (entry_idx[1]-1, entry_idx[2]+1)),
               (patch_idx[2], (entry_idx[1]+1, entry_idx[2]-1)), 
               (patch_idx[3], (entry_idx[1]+1, entry_idx[2]+1))]
    
    
    for maxima in max_idx:
        if (maxima[0] == -1) or (maxima[1][0] < 0) or (maxima[1][1] < 0) or \
            (maxima[1][0] >= current_response_size[0]) or (maxima[1][1] >= current_response_size[1]):
            continue
        
        response_map = next_response[maxima[0]]
        origion_pos = pool_idx[0, maxima[0], maxima[1][0], maxima[1][1]]
        next_pos = int(origion_pos/w), int(origion_pos%w)
        next_score = response_map[next_pos] + entry_score
        next_point = ((maxima[0], next_pos[0], next_pos[1]), next_score)
        match_points.append(next_point)
    
    return match_points

def match_atomic_patch(pyramid, current_level, entry):
    if current_level == 0:
        return [entry]
    else:
        next_level = current_level - 1
        atomic_patchs = []
        match_points = match_next(entry, pyramid[current_level], pyramid[next_level])
        for p in match_points:
            atomic_patchs.extend(match_atomic_patch(pyramid, next_level, p))
        return atomic_patchs


def match_all_entry(entrys, pyramid):
    match_points = []
    match_points_idx = []
    match_points_reverse = []
    match_points_reverse_idx = []
    idx_hash = lambda x, y: int(x/4)*16384 + int(y/4)
    get_patch = lambda x: match_atomic_patch(pyramid, len(pyramid)-1, x)
    #get_patch2 = partial(match_atomic_patch, pyramid, len(pyramid)-1)
    #pool = mp.pool.Pool(processes=8)
    #ap = pool.imap(get_patch2, entrys)
    #atomic_patchs = []
    #for p in ap:
    #    atomic_patchs.extend(p)
    t1 = default_timer()
    for e in entrys:
        atomic_patchs = get_patch(e)#match_atomic_patch(e, pyramid, len(pyramid)-1)
        #print(atomic_patchs)
        for p in atomic_patchs:
            if p[0][0] in match_points_idx:
                idx = match_points_idx.index(p[0][0])
                if match_points[idx][1] < p[1]:
                    match_points[idx] = p
            else:
                match_points.append(p)
                match_points_idx.append(p[0][0])

            t = idx_hash(p[0][1], p[0][2])
            if t in match_points_reverse_idx:
                idx = match_points_reverse_idx.index(t)
                if match_points_reverse[idx][1] < p[1]:
                    match_points_reverse[idx] = p
            else:
                match_points_reverse.append(p)
                match_points_reverse_idx.append(t)
    print(len(entrys), 'entrys, use', default_timer()-t1, 'seconds', (default_timer()-t1)/len(entrys), 'eps')
    return match_points, match_points_reverse

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
    #match_points = match_entry(entry, pyramid)
    t1 = default_timer()
    match_points12, match_points21 = match_all_entry(entry, pyramid)
    match_points12 = sorted(match_points12, key=lambda x: x[1], reverse=True)
    match_points21 = sorted(match_points21, key=lambda x: x[1], reverse=True)
    #match_points12 = sorted(merge_points(match_points, 0), key=lambda x: x[1], reverse=True)
    #match_points21 = sorted(merge_points(match_points, 1, 4), key=lambda x: x[1], reverse=True)
    t2 = default_timer()
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
    t3 = default_timer()
    match_points = convert2coord(t, pyramid[0]['size'], pyramid[0]['kernel_size'])
    print(t2-t1, t3-t2)
    return match_points