"""Independent, standard-library reference for Unit 04. Not copied from YOLOX.

The teaching simulation uses the current geometry-only center constraint (radius=1.5).
It does not implement an end-to-end detector or trained model.
"""
from __future__ import annotations
import math


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def bce_with_logits(z: float, target: float) -> float:
    return max(z, 0) + math.log1p(math.exp(-abs(z))) - z * target


def decode_box(raw: tuple[float, float, float, float], gx: int, gy: int, stride: int):
    tx, ty, tw, th = raw
    return ((gx + tx) * stride, (gy + ty) * stride,
            math.exp(tw) * stride, math.exp(th) * stride)


def to_xyxy(cxcywh):
    cx, cy, w, h = cxcywh
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0])*max(0,a[3]-a[1])
    area_b = max(0, b[2]-b[0])*max(0,b[3]-b[1])
    denominator = area_a+area_b-intersection
    return intersection / denominator if denominator > 0 else 0.0


def geometry(gx, gy, stride, gt_xywh, radius=1.5):
    cx, cy = (gx + 0.5) * stride, (gy + 0.5) * stride
    return (abs(cx - gt_xywh[0]) < stride*radius
            and abs(cy - gt_xywh[1]) < stride*radius)


def dynamic_k(ious):
    top10 = sorted(ious, reverse=True)[:10]
    return max(1, int(sum(top10))) if top10 else 0


def classification_cost(obj_logit, cls_logits, gt_class):
    p = [math.sqrt(sigmoid(z) * sigmoid(obj_logit)) for z in cls_logits]
    return sum(-math.log(max(1e-12,v)) if i == gt_class else -math.log(max(1e-12,1-v))
               for i,v in enumerate(p))


def simota(cost_matrix, iou_matrix):
    """Dynamic-K + minimum cost conflict resolution for a pre-filtered candidate pool.

    cost_matrix and iou_matrix have shape [num_gt][num_candidate]. Input costs should
    already contain a large penalty for per-GT geometry violations.
    """
    if len(cost_matrix) != len(iou_matrix):
        raise ValueError('GT count mismatch')
    if not cost_matrix:
        return [], []
    n = len(cost_matrix[0])
    if any(len(c)!=n or len(i)!=n for c,i in zip(cost_matrix, iou_matrix)):
        raise ValueError('nonrectangular cost/iou matrix')
    if n == 0:
        return [0]*len(cost_matrix), []
    ks = [dynamic_k(row) for row in iou_matrix]
    selected = [sorted(range(n), key=lambda j:(cost_matrix[g][j],j))[:k] for g,k in enumerate(ks)]
    owner = [-1]*n
    for g,row in enumerate(selected):
        for j in row:
            if owner[j]<0 or cost_matrix[g][j]<cost_matrix[owner[j]][j]:
                owner[j]=g
    return ks,owner


def nms(candidates, iou_threshold, score_threshold=0, class_aware=True):
    """Candidates are (id, xyxy, score, class). Returns kept ids in score order."""
    accepted=[]
    for c in sorted((c for c in candidates if c[2]>=score_threshold),key=lambda c:(-c[2],str(c[0]))):
        if not any((not class_aware or c[3]==prev[3])
                   and iou_xyxy(c[1],prev[1])>iou_threshold for prev in accepted):
            accepted.append(c)
    return [x[0] for x in accepted]
