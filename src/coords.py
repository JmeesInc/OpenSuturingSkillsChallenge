import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pycocotools import mask
import json
import glob

from postprocess import get_edge_point, get_contours
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

input_dir = sys.argv[1]

def choose_indice(json_paths):
    '''
    scoresがthresholdを超えた回数から、利用する3つのindiceを選択する
    '''
    indice_li = []
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
        scores = data['scores']
        for i, scr in enumerate(scores):
            if scr < 0.3:
                continue
            indice_li.append(i)
    indice, count = np.unique(indice_li, return_counts=True)
    
    return list(np.sort(indice[np.argsort(count)[::-1][:3]]))

def postprocess(json_path, indices):
    '''image pathとjson pathから、indicesに対応するmaskのedge pointを取得する'''
    # Load json
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Load masks
    scores = data['scores']
    boxes = data['bboxes']
    masks = data['masks']
    # Postprocess
    pts = []
    assert len(boxes) == len(masks) == 100
    for i in indices:
        box = boxes[i]
        msk = masks[i]
        msk = mask.decode(msk)
        if len(get_contours(msk)) == 0:
            pts.append(None)
            continue
        # get edge of mask
        pts.append(get_edge_point(box, msk, (1080, 1920)))
    return pts

#input_dirの親dir../seg_jsonに対してglob.glob
video_paths = sorted(glob.glob(f"{input_dir}/**/"))

print(video_paths)

pts_dict = {}
indices = [8, 10, 74, 81]
for video_path in tqdm(video_paths):

    json_paths = sorted(glob.glob(video_path + "/preds/*.json"))
    json_idx = [int(x.split('/')[-1].split('.')[0]) for x in json_paths]
    max_idx, min_idx = max(json_idx), min(json_idx)

    json_paths = [video_path + "preds/{}.json".format(i) for i in range(min_idx, max_idx+1)]
    pts_li = process_map(postprocess, json_paths, [indices]*len(json_paths), max_workers=128)
    new_li = []
    for pts in pts_li:
        pts = [pt if pt is not None else [540, 920] for pt in pts]
        new_li.append(pts)
    pts_li = np.array(new_li)

    video_title = video_path[:-1].replace("seg_json", "coord") + ".npy"
    os.makedirs(os.path.dirname(video_title), exist_ok=True)
    np.save(video_title, pts_li)