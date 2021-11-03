from shutil import Error
from helmet_assignment.score import NFLAssignmentScorer, check_submission
from helmet_assignment.features import add_track_features
from helmet_assignment.utils import *
import os
from tqdm import tqdm
import cv2
import pandas as pd
import torch
from sklearn.neighbors import KDTree
import numpy as np


tgt_imgh, tgt_imgw = 720, 1280


def mapping_df(args):
    video_frame, df, tracking, CONF_THRE_END, CONF_THRE_SIDE, endview_loc_dict, sideline_loc_dict, DIG_MAX, DIG_STEP = args
    gameKey,playID,view,frame = video_frame.split('_')
    # need for camera prediction
    video_name = gameKey + '_' + playID + '_' + view
    gameKey = int(gameKey)
    playID = int(playID)
    frame = int(frame)
    this_tracking = tracking[(tracking['gameKey']==gameKey) & (tracking['playID']==playID)]
    est_frame = find_nearest(this_tracking.est_frame.values, frame)
    this_tracking = this_tracking[this_tracking['est_frame']==est_frame]
    len_this_tracking = len(this_tracking)
    df['center_h_p'] = (df['left']+df['width']/2).astype(int)
    df['center_h_m'] = (df['left']+df['width']/2).astype(int)*-1
    
    df['center_v_p'] = tgt_imgh - (df['top']+df['height']/2).astype(int)
    df['center_v_m'] = (tgt_imgh - (df['top']+df['height']/2).astype(int))*-1
    
    if view == 'Endzone':
        df = df[df['conf']>CONF_THRE_END].copy()
    else:
        df = df[df['conf']>CONF_THRE_SIDE].copy()
     # cut out not inside the match   
    
    df['center_x_rotate'] = tgt_imgw - (df['left'] + df['width']/2).astype(int)
    df['center_y_rotate'] = tgt_imgh - (df['left'] + df['width']/2).astype(int)
    # keep the original coordinate
    df['center_x_orig'] = (df['left'] + df['width']/2).astype(int) # helmet CENTER position of x axis
    df['center_y_orig'] = (df['top'] + df['height']/2).astype(int) # helmet CENTER position of y axis
    
    if len(df) > len_this_tracking and len(df) > 30:
        df = crop_img_edge(df.copy(),this_tracking)
#     else:
#         df = df.tail(len_this_tracking)
#     print(df['video'])
    if len(df) > len_this_tracking:
        df = df.tail(len_this_tracking)

    df_h_p = df.sort_values('center_h_p').copy()
    df_h_m = df.sort_values('center_h_m').copy()
    
    df_v_p = df.sort_values('center_v_p').copy()
    df_v_m = df.sort_values('center_v_m').copy()
    
    if view == 'Endzone':
        this_tracking['x'], this_tracking['y'] = this_tracking['y'].copy(), this_tracking['x'].copy()
    a2_h_p = df_h_p['center_h_p'].values
    a2_h_m = df_h_m['center_h_m'].values
    
    a2_v_p = df_v_p['center_v_p'].values
    a2_v_m = df_v_m['center_v_m'].values
    ######################## new code ########################
    # first mapping based on x
    if view == 'Endzone':   #same as before
        if endview_loc_dict[video_name] =='R':
            min_dist_p, min_detete_idx_p = dist_rot_x(this_tracking ,a2_h_p, DIG_MAX, DIG_STEP)
            min_dist = min_dist_p
            min_detete_idx = min_detete_idx_p
            tgt_df = df_h_p
        elif endview_loc_dict[video_name] =='L':
            min_dist_m, min_detete_idx_m = dist_rot_x(this_tracking ,a2_h_m, DIG_MAX, DIG_STEP)
            min_dist = min_dist_m
            min_detete_idx = min_detete_idx_m
            tgt_df = df_h_m
        else:
            min_dist_p, min_detete_idx_p = dist_rot_x(this_tracking ,a2_h_p, DIG_MAX, DIG_STEP)
            min_dist_m, min_detete_idx_m = dist_rot_x(this_tracking ,a2_h_m, DIG_MAX, DIG_STEP)
            if min_dist_p < min_dist_m:
                min_dist = min_dist_p
                min_detete_idx = min_detete_idx_p
                tgt_df = df_h_p
            else:
                min_dist = min_dist_m
                min_detete_idx = min_detete_idx_m
                tgt_df = df_h_m
    else:                   # using the camera prediciton here
        if sideline_loc_dict[video_name] == 0:
            min_dist_p, min_detete_idx_p = dist_rot_x(this_tracking ,a2_h_p, DIG_MAX, DIG_STEP)
            min_dist = min_dist_p
            min_detete_idx = min_detete_idx_p
            tgt_df = df_h_p
        else:
            min_dist_m, min_detete_idx_m = dist_rot_x(this_tracking ,a2_h_m, DIG_MAX, DIG_STEP)
            min_dist = min_dist_m
            min_detete_idx = min_detete_idx_m
            tgt_df = df_h_m
    #storing the result from x
    
    min_dist_x = min_dist
    min_detete_idx_x = min_detete_idx
    tgt_df_x = tgt_df
    
    ######################## new code ########################
    # then mapping based on y
    if view == 'Endzone':   #same as before
        if endview_loc_dict[video_name] =='L':
            min_dist_p, min_detete_idx_p = dist_rot_y(this_tracking ,a2_v_p, DIG_MAX, DIG_STEP)
            min_dist = min_dist_p
            min_detete_idx = min_detete_idx_p
            tgt_df = df_v_p
        elif endview_loc_dict[video_name] =='R':
            min_dist_m, min_detete_idx_m = dist_rot_y(this_tracking ,a2_v_m, DIG_MAX, DIG_STEP)
            min_dist = min_dist_m
            min_detete_idx = min_detete_idx_m
            tgt_df = df_v_m
        else:
            min_dist_p, min_detete_idx_p = dist_rot_y(this_tracking ,a2_v_p, DIG_MAX, DIG_STEP)
            min_dist_m, min_detete_idx_m = dist_rot_y(this_tracking ,a2_v_m, DIG_MAX, DIG_STEP)
            if min_dist_p < min_dist_m:
                min_dist = min_dist_p
                min_detete_idx = min_detete_idx_p
                tgt_df = df_v_p
            else:
                min_dist = min_dist_m
                min_detete_idx = min_detete_idx_m
                tgt_df = df_v_m
    else:                   # using the camera prediciton here
        if sideline_loc_dict[video_name] == 0:
            min_dist_p, min_detete_idx_p = dist_rot_y(this_tracking ,a2_v_p, DIG_MAX, DIG_STEP)
            min_dist = min_dist_p
            min_detete_idx = min_detete_idx_p
            tgt_df = df_v_p
        else:
            min_dist_m, min_detete_idx_m = dist_rot_y(this_tracking ,a2_v_m, DIG_MAX, DIG_STEP)
            min_dist = min_dist_m
            min_detete_idx = min_detete_idx_m
            tgt_df = df_v_m
    #storing the result from y
    
    min_dist_y = min_dist
    min_detete_idx_y = min_detete_idx
    tgt_df_y = tgt_df
    
    ######################## original code ########################
#     min_dist_p, min_detete_idx_p = dist_rot(this_tracking ,a2_p)
#     min_dist_m, min_detete_idx_m = dist_rot(this_tracking ,a2_m)
#     if min_dist_p < min_dist_m:
#         min_dist = min_dist_p
#         min_detete_idx = min_detete_idx_p
#         tgt_df = df_p
#     else:
#         min_dist = min_dist_m
#         min_detete_idx = min_detete_idx_m
#         tgt_df = df_m    
    #print(video_frame, len(this_tracking), len(df), len(df[df['conf']>CONF_THRE]), this_tracking['x'].mean(), min_dist_p, min_dist_m, min_dist)
    # take the better value from x or y mapping
    if min_dist_x <= min_dist_y:
        tgt_df = tgt_df_x
        min_detete_idx = min_detete_idx_x
    else:
        tgt_df = tgt_df_y
        min_detete_idx = min_detete_idx_y
    
    tgt_df['label'] = min_detete_idx
    return tgt_df[['video_frame','left','width','top','height','label']]

