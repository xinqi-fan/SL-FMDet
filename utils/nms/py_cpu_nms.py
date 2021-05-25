# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np


# compress similar boxes
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]  # 1st index (in order[1:]) whose iou <= threshold
        order = order[inds + 1]

    return keep


def py_cpu_nms_cross_class(dets_face, dets_mask, thresh, margin=0.1):
    f = dets_face.shape[0]
    m = dets_mask.shape[0]

    x1_face = dets_face[:, 0]
    y1_face = dets_face[:, 1]
    x2_face = dets_face[:, 2]
    y2_face = dets_face[:, 3]
    scores_face = dets_face[:, 4]
    areas_face = (x2_face - x1_face + 1) * (y2_face - y1_face + 1)

    x1_mask = dets_mask[:, 0]
    y1_mask = dets_mask[:, 1]
    x2_mask = dets_mask[:, 2]
    y2_mask = dets_mask[:, 3]
    scores_mask = dets_mask[:, 4]
    areas_mask = (x2_mask - x1_mask + 1) * (y2_mask - y1_mask + 1)

    rm_face = []
    rm_mask = []



    for i in range(f):
        for j in range(m):
            xx1 = np.maximum(x1_face[i], x1_mask[j])
            yy1 = np.maximum(y1_face[i], y1_mask[j])
            xx2 = np.minimum(x2_face[i], x2_mask[j])
            yy2 = np.minimum(y2_face[i], y2_mask[j])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            iou = inter / (areas_face[i] + areas_mask[j] - inter)
            if iou >= thresh:
                if scores_face[i] >= scores_mask[j] + margin:
                    pass
                    # rm_mask.append(j)
                else:
                    if scores_mask[j] >= scores_face[i] + margin:
                        rm_face.append(i)
    dets_face = np.delete(arr=dets_face, obj=rm_face, axis=0)
    dets_mask = np.delete(arr=dets_mask, obj=rm_mask, axis=0)
    # if len(rm_face) != 0:
    #     print('face is fused')
    # if len(rm_mask) != 0:
    #     print('mask is fused')

    return dets_face, dets_mask


if __name__ == '__main__':
    dets = np.array([[0.1, 0.2, 0.3, 0.4, 0.1],[0.05, 0.15, 0.5, 0.7, 0.3],[0.15, 0.25, 0.35, 0.45, 0.8],[0.18, 0.28, 0.32, 0.42, 0.5],[0.01,0.01,0.02,0.02,0.2]])
    print(dets)
    dets = np.delete(arr=dets, obj=[0, 2], axis=0)
    print(dets)
