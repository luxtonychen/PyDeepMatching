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

def match_next(entry, current_layer, next_layer):
    entry_idx, entry_score = entry
    patch_idx = current_layer['reverse_idx'][entry_idx[0]]
    pool_idx = next_layer['pool_idx']
    current_response_size = current_layer['map_size']
    next_response = next_layer['response_maps']
    h, w = next_layer['map_size']
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
        next_score = response_map[maxima[1]] + entry_score
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

def match_atomic_patch_loop(pyramid, entry):
    depth = len(pyramid)
    entries = [entry]
    for i in range(1, depth):
        tmp_entries = []
        for e in entries:
            tmp_entries.extend(match_next(e, pyramid[depth-i], pyramid[depth-(i+1)]))
        entries = tmp_entries
    return entries

def match_all_entry(pyramid, entries):
    match_points = []

    get_patch = lambda x: match_atomic_patch(pyramid, len(pyramid)-1, x)
    
    atomic_patchs = []
    for e in entries:
        atomic_patchs.extend(get_patch(e))

    match_points = merge_points(lambda x: x[0][0], atomic_patchs)
    match_points_reverse = merge_points(lambda x: int(x[0][1]/4)*16384 + int(x[0][2]/4), atomic_patchs)

    return match_points, match_points_reverse

def match_all_entry_parallel(entries, pyramid):
    n_processors = 6
    n_entries = len(entries)
    print('num of entries', n_entries)
    step = int(n_entries/n_processors)
    entries_slice = []
    for i in range(n_processors-1):
        entries_slice.append(entries[i*step:(i+1)*step])
    entries_slice.append(entries[7*step:])

    match_partial_entry = partial(match_all_entry, pyramid)
    pool = mp.pool.Pool(processes=n_processors)
    points = pool.imap(match_partial_entry, entries_slice)
    match_points, match_points_reverse = points.next()
    for e in points:
        match_points.extend(e[0])
        match_points_reverse.extend(e[1])
    match_points = merge_points(lambda x: x[0][0], match_points)
    match_points_reverse = merge_points(lambda x: int(x[0][1]/4)*16384 + int(x[0][2]/4), match_points_reverse)
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

def merge_points(idx_func, points):
    score_dict = {i: ((0, 0, 0), -1) for i in map(idx_func, points)}
    for p in points:
        if p[1] > score_dict[idx_func(p)][1]:
            score_dict[idx_func(p)] = p
    merged_points = list(score_dict.values())
    return merged_points

def convert2coord(match_points, shape, kernel_size):
    h, w = shape
    match_points_c = []
    for p in match_points:
        x0, y0 = (p[0][0] % w), int(p[0][0]/w)
        x0, y0 = x0*kernel_size, y0*kernel_size
        x1, y1 = p[0][2]*2, p[0][1]*2
        match_points_c.append(((x0, y0), (x1, y1), p[1]))
    return match_points_c


def matching(pyramid, top_n=-1):
    entry = get_entry(pyramid, top_n)
    t1 = default_timer()
    match_points12, match_points21 = match_all_entry(pyramid, entry)
    #match_points12 = sorted(match_points12, key=lambda x: x[1], reverse=True)
    #match_points21 = sorted(match_points21, key=lambda x: x[1], reverse=True)
    t2 = default_timer()
    match_points = list(set(match_points12) & set(match_points21))
#    t = []
#    score = lambda x: x[1]
#    loc = lambda x: x[0]
#    i21 = 0
#    i12 = 0
#    while (i12 < len(match_points12)) and i21 < len(match_points21):
#        if score(match_points12[i12]) < score(match_points21[i21]):
#            i21 += 1
#        elif score(match_points12[i12]) > score(match_points21[i21]):
#            i12 += 1
#        else: 
#            if loc(match_points12[i12]) == loc(match_points21[i21]):
#                t.append(match_points12[i12])
#                i12 += 1
#                i21 += 1
#            else:
#                i12 += 1
    match_points = convert2coord(match_points, pyramid[0]['size'], pyramid[0]['kernel_size'])
    print(t2-t1)
    return match_points