'''先端認識などの後処理'''
import numpy as np
import cv2
import json
from PIL import Image
from pycocotools import mask

class INPUT_SIZE:
    x = 1920
    y = 1080


def modify_point(point, out_size):
    o_w, o_h = out_size
    x, y = point
    y = y * o_h // INPUT_SIZE.x
    x = x * o_w // INPUT_SIZE.y
    return x, y


def modify_bbox(bbox, out_size):
    x1, y1, x2, y2 = bbox
    x1, y1 = modify_point([x1, y1], out_size)
    x2, y2 = modify_point([x2, y2], out_size)
    return x1, y1, x2, y2


def modify_mask(mask, out_size):
    return cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)


def calc_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return w * h


def is_small_instance(bbox, th_rate=0.005):
    # 閾値以下の面積のinstanceかどうかを判定する
    area = calc_bbox_area(bbox)
    if area / (INPUT_SIZE.x * INPUT_SIZE.y) < th_rate:
        return True
    return False


def bitwise(mask):
    bw = np.zeros((mask.shape), np.uint8)
    bw[mask > 0] = 255
    return bw


def get_contours(mask, find_contours_flag=cv2.RETR_EXTERNAL):
    # 輪郭点取得
    bw = bitwise(mask)
    contours, hierarchy = cv2.findContours(
        bw, find_contours_flag, cv2.CHAIN_APPROX_NONE)
    return contours


def get_max_contour(mask):
    # 最も大きい輪郭を取得
    contours = get_contours(mask)
    max_area = 0
    max_idx = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_idx = i
    return contours[max_idx]


def distance(pt1, pt2, axis=0):
    # pt1, pt2 :tuple | list[tuple] | np.ndarray: (x,y)
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.linalg.norm(pt1 - pt2, axis=axis)


def get_edge_point(bbox, mask, img_size, hdelta=0.13):
    # ランドマーク座標の取得
    x1, y1, x2, y2 = bbox
    center = (img_size[0] // 2, img_size[1] // 2)
    # 直線近似
    cnt = get_max_contour(mask)
    vx, vy, x, y = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    a = vy / vx  # 傾き
    # print('delta', a[0])
    # maskの向きとbboxから術具の先端を割り出す
    if abs(a[0]) < hdelta:
        # 傾きがほぼ水平の場合、
        # bboxの2点の中点を先端とする
        left = (x1, (y1 + y2) // 2)
        right = (x2, (y1 + y2) // 2)
    # TODO: 理論的には傾きは正だが、符号を反転させないとうまくいかない
    elif a[0] < 0:
        # 傾きが正の場合、
        # bboxの左下点(x1,y2)付近か右上点(x2,y1)付近のどちらかが先端
        left = (x1, y2)
        right = (x2, y1)
    else:
        # 傾きが負の場合、
        # bboxの左上点(x1,y1)付近か右下点(x2,y2)付近が先端
        left = (x1, y1)
        right = (x2, y2)
    # より画面中心に近い方を先端とする
    if distance(left, center) < distance(right, center):
        edge_x, edge_y = left
    else:
        edge_x, edge_y = right
    return int(edge_x), int(edge_y)


def ignore_instance(edge,  # (x,y)
                    img_size,  # (width, height)
                    margin=50  # [px]
                    ):
    # 先端点が画面端にある場合はこのインスタンスは無視する
    w, h = img_size
    # 画面端領域を作成
    area = np.ones((h, w), np.uint8)
    area[margin:h - margin, margin:w - margin] = 0
    # 画面端領域に点があるか判断
    if area[edge[1], edge[0]] > 0:
        # 点が画面端にある場合
        return True
    else:
        # 点が画面端にない場合
        return False


def filter_center_points(pts: list, img_size: tuple, n_sample: int = 2):
    pts = np.array(pts)
    # ptsの数分中心座標を作る
    cx, cy = img_size[0] // 2, img_size[1] // 2
    centers = [(cx, cy)] * len(pts)
    # ptsと中心座標の距離を一気に計算し、距離の近い順を取得
    order = np.argsort(distance(pts, centers, axis=1))
    # 近い点をn_sample分抽出する
    sampled = pts[order][:n_sample]
    # list(tuple)の形に戻す
    sampled = [(x, y) for x, y in sampled]
    return sampled

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

def postprocess(image_path, json_path, indices):
    '''image pathとjson pathから、indicesに対応するmaskのedge pointを取得する'''
    # Load image
    image = Image.open(image_path)
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
            #print('No contours at ', i, ' ', image_path)
            pts.append(None)
            continue
        # get edge of mask
        pts.append(get_edge_point(box, msk, image.size))
    return pts
