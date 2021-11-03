import numpy as np
import itertools
import random

def find_nearest(array, value):
    value = int(value)
    array = np.asarray(array).astype(int)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
def norm_arr(a):
    a = a-a.min()
    a = a/a.max()
    return a

def dist(a1, a2):
    return np.linalg.norm(a1-a2)

max_iter = 2000
def dist_for_different_len(a1, a2):
    assert len(a1) >= len(a2), f'{len(a1)}, {len(a2)}' # make sure the number of detected helmets <= the number of player 
    len_diff = len(a1) - len(a2)
    a2 = norm_arr(a2)
    if len_diff == 0:
        a1 = norm_arr(a1)
        return dist(a1,a2), ()
    else:
        min_dist = 10000
        min_detete_idx = None
        cnt = 0
        del_list = list(itertools.combinations(range(len(a1)),len_diff)) # get all possible wrong players' combinations 
        # if the combinations is a lot. the speed will be slow down. hence we will sample max_iter combinations
        if len(del_list) > max_iter:
            del_list = random.sample(del_list, max_iter)
        # forloop the combinations, choose the one with the least distance, which mean the NGS and video is closer
        for detete_idx in del_list:
            this_a1 = np.delete(a1, detete_idx)
            this_a1 = norm_arr(this_a1)
            this_dist = dist(this_a1, a2)
            #print(len(a1), len(a2), this_dist)
            if min_dist > this_dist:
                min_dist = this_dist
                min_detete_idx = detete_idx
                
        return min_dist, min_detete_idx
    
def rotate_arr(u, t, deg=True):
    if deg == True:
        t = np.deg2rad(t) # degree to radians(radian = degree*pi/180) ref: https://www.cnblogs.com/yyy6/p/8110206.html
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t),  np.cos(t)]])
    return  np.dot(R, u) # https://www.cnblogs.com/zhoug2020/p/7842808.html


def dist_rot(tracking_df, a2):
    tracking_df = tracking_df.sort_values('x') # sort by x, to match the order of a2
    x = tracking_df['x']
    y = tracking_df['y']
    min_dist = 10000
    min_idx = None
    min_x = None
    dig_step = 3
    dig_max = dig_step*10
    for dig in range(-dig_max,dig_max+1,dig_step):
        arr = rotate_arr(np.array((x,y)), dig) # rotated x and y array
        this_dist, this_idx = dist_for_different_len(np.sort(arr[0]), a2) # get the distance and idx of the best combinations
        if min_dist > this_dist:
            min_dist = this_dist
            min_idx = this_idx
            min_x = arr[0]
    tracking_df['x_rot'] = min_x
    player_arr = tracking_df.sort_values('x_rot')['player'].values
    players = np.delete(player_arr,min_idx) # delete the best combinations, which make the score best
    return min_dist, players


def dist_rot_x(tracking_df, a2, DIG_MAX, DIG_STEP):
    tracking_df = tracking_df.sort_values('x')
    x = tracking_df['x']
    y = tracking_df['y']
    min_dist = 10000
    min_idx = None
    min_x = None
    for dig in range(-DIG_MAX,DIG_MAX+1,DIG_STEP):
        arr = rotate_arr(np.array((x,y)), dig)
        this_dist, this_idx = dist_for_different_len(np.sort(arr[0]), a2)
        if min_dist > this_dist:
            min_dist = this_dist
            min_idx = this_idx
            min_x = arr[0]
    tracking_df['x_rot'] = min_x
    player_arr = tracking_df.sort_values('x_rot')['player'].values
    players = np.delete(player_arr,min_idx)
    return min_dist, players

def dist_rot_y(tracking_df, a2, DIG_MAX, DIG_STEP):
    tracking_df = tracking_df.sort_values('y')
    x = tracking_df['x']
    y = tracking_df['y']
    min_dist = 10000
    min_idx = None
    min_x = None
    for dig in range(-DIG_MAX,DIG_MAX+1,DIG_STEP):
        arr = rotate_arr(np.array((x,y)), dig)
        this_dist, this_idx = dist_for_different_len(np.sort(arr[1]), a2)
        if min_dist > this_dist:
            min_dist = this_dist
            min_idx = this_idx
            min_y = arr[1]
    tracking_df['y_rot'] = min_y
    player_arr = tracking_df.sort_values('y_rot')['player'].values
    players = np.delete(player_arr,min_idx)
    return min_dist, players


def crop_img_edge(tgt_helmet_df, tgt_ngs_df, crop_dist = 40, crop_max_iter = 4):
    '''裁剪图片边缘的头盔
        crop_dist = 40 # 每次裁剪40pixel
        crop_max_iter = 4 # 最大裁剪次数
    '''
    tgt_imgh, tgt_imgw = 720, 1280

    i = 1
    while len(tgt_helmet_df) > len(tgt_ngs_df):
        # 若超过最大裁剪次数，跳出训练
        if i > crop_max_iter:
            break
        
        # 准备裁剪上下左右界限
        y_upper_limit = tgt_imgh - crop_dist * i
        y_bottom_limit = crop_dist * i
        x_upper_limit = tgt_imgw - crop_dist * i
        x_bottom_limit = crop_dist * i
        
        tmp_helmet_df = tgt_helmet_df.copy()
        drop_idxs = []
        for j in range(len(tgt_helmet_df)):
            tmp_x = tmp_helmet_df['center_x_orig'].iloc[j]
            tmp_y = tmp_helmet_df['center_y_orig'].iloc[j]
            # 如果在裁剪范围，则剔除球员
            if (tmp_x < x_bottom_limit) or (tmp_x > x_upper_limit) or (tmp_y < y_bottom_limit) or (tmp_y > y_upper_limit):
                tmp_idx = tgt_helmet_df.index[j]
                drop_idxs.append(tmp_idx)
        
        tmp_helmet_df.drop(drop_idxs, inplace = True)
        tgt_helmet_df = tmp_helmet_df.copy()
        i += 1 

    # 若超过最大裁剪次数，跳出后仍有多余球员，则按置信度剔除
#     if len(tgt_helmet_df) > len(tgt_ngs_df):  
#         tgt_helmet_df = tgt_helmet_df.tail(len(tgt_ngs_df))
    
    return tgt_helmet_df