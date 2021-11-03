import os
from tqdm.auto import tqdm
import cv2
import pandas as pd
from PIL import Image
from helmet_assignment.utils import *


def camera_postion_sideline(VIDEO_DIR: str,   # directory of videos
        tgt_vname: str,  # target video name
        tgt_frame: int,  # frame of the target video name
        sideline_loc_dict: dict,  # writable sideline loc dictionary
        helmet_df: pd.DataFrame,   # helmet dataframe
        ngs_df: pd.DataFrame,  # tracks ngs dataframe
        CONF_THRESH_SIDE: float  # confidence threshold of sideline
        ):
    # Target Video: obtain the targeted game info
    tgt_vpath = VIDEO_DIR + tgt_vname + '.mp4'

    tgt_helmet_df = helmet_df[helmet_df['video_frame'] == tgt_vname + '_{}'.format(tgt_frame)]
    tgt_gamekey = int(tgt_helmet_df['gameKey'].values[0])
    tgt_playid = int(tgt_helmet_df['playID'].values[0])
    tgt_ngs_df = ngs_df[(ngs_df['gameKey'] == tgt_gamekey) & (ngs_df['playID'] == tgt_playid)]

    # Target Frame: given a Helmet frame. find the nearest frame in NGS
    est_frame = find_nearest(tgt_ngs_df['est_frame'].values, tgt_frame)
    tgt_ngs_df = tgt_ngs_df[tgt_ngs_df['est_frame'] == est_frame]

    # print("tgt_vname:{}\n\
    # tgt_gamekey:{}\n\
    # tgt_playid:{}\n\
    # len of helmets:{}\n\
    # len of NGS:{}".format(tgt_vname, tgt_gamekey, tgt_playid, len(tgt_helmet_df), len(tgt_ngs_df)))

    # filter out the box with low confidence
    tgt_helmet_df = tgt_helmet_df[tgt_helmet_df['conf'] > CONF_THRESH_SIDE]

    # Load the target frame into an image
    cap = cv2.VideoCapture(tgt_vpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(tgt_frame) - 1) # optional, frame is 0-based indexed
    success, image = cap.read()
    tgt_imgw = image.shape[1]
    tgt_imgh = image.shape[0]

    # define the helmet location
    # rotate the coordinate with 180 degree
    tgt_helmet_df['center_x_rotate'] = tgt_imgw - (tgt_helmet_df['left'] + tgt_helmet_df['width']/2).astype(int)
    tgt_helmet_df['center_y_rotate'] = tgt_imgh - (tgt_helmet_df['left'] + tgt_helmet_df['width']/2).astype(int)
    # keep the original coordinate
    tgt_helmet_df['center_x_orig'] = (tgt_helmet_df['left'] + tgt_helmet_df['width']/2).astype(int) # helmet CENTER position of x axis
    tgt_helmet_df['center_y_orig'] = (tgt_helmet_df['top'] + tgt_helmet_df['height']/2).astype(int) # helmet CENTER position of y axis
    
    # # if helmets' number > NGS number, 
    # if len(tgt_helmet_df) > len(tgt_ngs_df):
    #     tgt_helmet_df = tgt_helmet_df.tail(len(tgt_ngs_df))

    # 修改：若场内头盔数大于30， 改用边缘头盔裁剪
    if len(tgt_helmet_df) > len(tgt_ngs_df) and len(tgt_helmet_df) > 30:
        tgt_helmet_df = crop_img_edge(tgt_helmet_df, tgt_ngs_df, crop_dist = 40, crop_max_iter = 4)
#     else:
#         tgt_helmet_df = tgt_helmet_df.tail(len(tgt_ngs_df))   
        
    
    if len(tgt_helmet_df) > len(tgt_ngs_df):
        tgt_helmet_df = tgt_helmet_df.tail(len(tgt_ngs_df))

    # rotate the ngs and compute the distance between video and roated ngs
    tgt_helmet_df_orig = tgt_helmet_df.sort_values('center_x_orig') 
    center_x_orig = tgt_helmet_df_orig['center_x_orig'].values
    min_dist_orig, min_detete_idx_orig = dist_rot(tgt_ngs_df, center_x_orig)
    tgt_helmet_df_rotate = tgt_helmet_df.sort_values('center_x_rotate') 
    center_x_rotate = tgt_helmet_df_rotate['center_x_rotate'].values
    min_dist_rotate, min_detete_idx_rotate = dist_rot(tgt_ngs_df, center_x_rotate)    

    # choose the camera side
    if min_dist_orig <= min_dist_rotate:
        sideline_loc_dict[tgt_vname] = 0
    else:
        sideline_loc_dict[tgt_vname] = 1


def get_frame_from_video(frame, video, out_path='./frame.png'):
    video_path = video
    frame = frame - 1

    cmd = 'ffmpeg -hide_banner -loglevel fatal -nostats -i {} -vf "select=eq(n\,{})" -vframes 1 {}'.format(video, frame, out_path)
    os.system(cmd)
    img = Image.open(out_path)
    os.remove(out_path)
    return img



def camera_position_endview(
    VIDEO_DIR,
    video,
    tracking,
    helmets,
    CONF_THRE_END,
    endview_loc_dict,
    reader,
    sample_frame_path='frame.png',
    view='Endzone'):

    try_frame = 1
    if video.endswith('.mp4'):
        video_with_extension = video
        tgt_vname = video.rstrp('.mp4')
    else:
        video_with_extension = video + '.mp4'
        tgt_vname = video

    video_path = os.path.join(VIDEO_DIR, video_with_extension)
    
    
    gameKey, playID, _ = tgt_vname.split('_')
    game_play = gameKey + '_' + playID
    sample_tracking = tracking[tracking['game_play'] == game_play]
    sample_helmet = helmets[helmets['video'] == video_with_extension]
    pos = None
    
    while try_frame <= 100:
        video_frame = tgt_vname + '_{}'.format(try_frame)
        gameKey = int(gameKey)
        playID = int(playID)
        frame = int(try_frame)

        df = sample_helmet[sample_helmet['video_frame'] == video_frame]
        if len(df) == 0:
            break
        this_tracking = sample_tracking[(sample_tracking['gameKey']==gameKey) & (sample_tracking['playID']==playID)]
        est_frame = find_nearest(this_tracking.est_frame.values, frame)
        this_tracking = this_tracking[this_tracking['est_frame']==est_frame]
        len_this_tracking = len(this_tracking)

        if len_this_tracking == 0:
            break
        

        df['center_h_p'] = (df['left']+df['width']/2).astype(int)
        df['center_h_m'] = (df['left']+df['width']/2).astype(int)*-1

        df['center_x_rotate'] = 1280 - (df['left'] + df['width']/2).astype(int)
        df['center_y_rotate'] = 720 - (df['left'] + df['width']/2).astype(int)
        # keep the original coordinate
        df['center_x_orig'] = (df['left'] + df['width']/2).astype(int) # helmet CENTER position of x axis
        df['center_y_orig'] = (df['top'] + df['height']/2).astype(int) # helmet CENTER position of y axis

        if view == 'Endzone':
            df = df[df['conf']>CONF_THRE_END].copy()
    
        if len(df) > len_this_tracking and len(df) > 30:
            df = crop_img_edge(df.copy(),this_tracking)
#         else:
#             df = df.tail(len_this_tracking)

        if len(df) > len_this_tracking:
            df = df.tail(len_this_tracking)

        
        frame_end = get_frame_from_video(frame=try_frame, video=video_path, out_path=sample_frame_path)
        frame_end = np.asarray(frame_end, dtype=np.uint8)

        results = reader.readtext(frame_end, detail=1)
        res = set()
        for result in results:
            # result: [[[left top x, y], [right top x, y], [right bottom x, y], [left bottom x, y]], text, conf]
            conf = result[-1]
            if conf < 0.2 or result[0][0][1] < 300: continue

            text = result[1].replace(' ', '')
            if text.isnumeric():
                res.add(text)
        
        this_sorted_tracking_players = this_tracking.sort_values('x')['player'].values
        pos = set()
        for t in res:
            for i, player in enumerate(this_sorted_tracking_players):
                if t in player:
                    pos.add(i+1)
                    break
        if len(pos) > 0:
            break
        try_frame += 30

    if pos is None or len(pos) == 0:
        endview_loc_dict[tgt_vname] = None
    elif sum(pos) > len(pos) * len(this_sorted_tracking_players) // 2:
        endview_loc_dict[tgt_vname] = 'R'
    else:
        endview_loc_dict[tgt_vname] = 'L'


def camera_position(VIDEO_DIR, helmet_df, ngs_df, CONF_THRESH_SIDE, CONF_THRESH_END, reader, out_frame_path='frame.png'):
    vnames = os.listdir(VIDEO_DIR)
    vnames = [v[:-4] for v in vnames]
    tgt_frame  = '1' # choose the first frame to decide the location of the camera

    sideline_loc_dict = {} # save the sideline camera's location
    endview_loc_dict = {}


    for tgt_vname in tqdm(vnames):
        # Sideline
        if tgt_vname.find('Sideline') != -1:
            camera_postion_sideline(VIDEO_DIR, tgt_vname, tgt_frame, sideline_loc_dict, helmet_df, ngs_df, CONF_THRESH_SIDE)
        
        # EndView
        else:
            camera_position_endview(VIDEO_DIR, tgt_vname, ngs_df, helmet_df, CONF_THRESH_END, endview_loc_dict, reader, sample_frame_path=out_frame_path)

    return sideline_loc_dict, endview_loc_dict