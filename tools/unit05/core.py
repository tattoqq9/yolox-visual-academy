"""One-class YOLOX-Nano educational training: genuine autograd and geometry SimOTA.

Designed for synthetic rectangles and CPU. This is NOT full official YOLOX trainer.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

STRIDES=(8,16,32)

def flatten(raw_maps):
    """Return (N,6) raw predictions, int grid coords and their stride."""
    outputs=[];grids=[];strides=[]
    for z,stride in zip(raw_maps,STRIDES):
        assert z.shape[0]==1 and z.shape[1]==6
        _,_,h,w=z.shape
        outputs.append(z.permute(0,2,3,1).reshape(-1,6))
        gy,gx=torch.meshgrid(torch.arange(h,device=z.device),torch.arange(w,device=z.device),indexing='ij')
        grids.append(torch.stack((gx.flatten(),gy.flatten()),dim=1).float())
        strides.append(torch.full((h*w,1),stride,dtype=torch.float32,device=z.device))
    return torch.cat(outputs),torch.cat(grids),torch.cat(strides)


def decode(raw,grid,stride):
    """YOLOX xywh output in absolute image pixels; clamp for extreme logits."""
    return torch.cat(((raw[:,:2]+grid)*stride,raw[:,2:4].clamp(-8,8).exp()*stride),dim=1)


def xyxy(boxes):
    return torch.cat((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),dim=-1)


def pairwise_iou(boxes,gt):
    """boxes (N,4) center-format, gt (4,) center-format, differentiable w.r.t boxes."""
    a=xyxy(boxes);b=xyxy(gt[None])
    wh=(torch.minimum(a[:,2:],b[:,2:])-torch.maximum(a[:,:2],b[:,:2])).clamp(min=0)
    inter=wh[:,0]*wh[:,1]
    area_a=(a[:,2]-a[:,0]).clamp(min=0)*(a[:,3]-a[:,1]).clamp(min=0)
    area_b=(b[:,2]-b[:,0]).clamp(min=0)*(b[:,3]-b[:,1]).clamp(min=0)
    return inter/(area_a+area_b-inter+1.e-9)


@torch.no_grad()
def assign(raw,grid,stride,gt):
    """One-GT geometry-constrained SimOTA. Dynamic K from top10 candidate IoUs.

    Full multi-GT conflict handling is a Unit 04 exercise, not needed for one-GT data.
    """
    centers=(grid+.5)*stride
    near=((centers-gt[:2]).abs()<stride*1.5).all(dim=1)
    candidates=torch.nonzero(near,as_tuple=False).squeeze(1)
    if candidates.numel()==0:
        d=((centers-gt[:2])**2).sum(dim=1);candidates=d.argmin()[None]
    p=decode(raw[candidates],grid[candidates],stride[candidates])
    ious=pairwise_iou(p,gt)
    # The same positive selection principles as YOLOX, reduced to a single GT/class.
    cls_obj=(raw[candidates,4].sigmoid()*raw[candidates,5].sigmoid()).clamp_min(1e-8).sqrt()
    cls_cost=-torch.log(cls_obj)
    cost=cls_cost+3*(-torch.log(ious.clamp_min(1e-8)))
    k=min(candidates.numel(),max(1,int(ious.topk(min(10,len(ious))).values.sum().item())))
    selected=candidates[cost.topk(k,largest=False).indices]
    return selected,k,int(candidates.numel())


def train_step_loss(raw_maps,gt):
    raw,grid,stride=flatten(raw_maps)
    positives,k,nc=assign(raw.detach(),grid,stride,gt.detach())
    count=positives.numel()
    output_boxes=decode(raw[positives],grid[positives],stride[positives])
    ious=pairwise_iou(output_boxes,gt)
    reg=(1-ious.square()).sum()/count*5
    labels=torch.zeros_like(raw[:,4]);labels[positives]=1
    obj=F.binary_cross_entropy_with_logits(raw[:,4],labels,reduction='sum')/count
    quality=ious.detach()
    cls=F.binary_cross_entropy_with_logits(raw[positives,5],quality,reduction='sum')/count
    return reg+obj+cls,{'reg':float(reg.detach()),'obj':float(obj.detach()),'cls':float(cls.detach()),
                        'positives':int(count),'candidate_count':nc,'dynamic_k':k,'mean_iou':float(ious.mean().detach())}

@torch.no_grad()
def predict(model,image,score_threshold=.00001,nms_threshold=.5,topk=50):
    """Single-class predictions: (xyxy,score). Doesn't alter model training mode."""
    model.eval()
    raw,grid,stride=flatten(model(image))
    boxes=xyxy(decode(raw,grid,stride))
    scores=raw[:,4].sigmoid()*raw[:,5].sigmoid()
    order=torch.argsort(scores,descending=True)
    keep=[]
    for i in order.tolist():
        score=float(scores[i]);
        if score<score_threshold:break
        box=boxes[i]
        if all(float(_iou_xyxy(box,boxes[j]))<nms_threshold for j in keep): keep.append(i)
        if len(keep)>=topk:break
    return [(boxes[i].clamp(-256,384).tolist(),float(scores[i])) for i in keep]


def _iou_xyxy(a,b):
    wh=(torch.minimum(a[2:],b[2:])-torch.maximum(a[:2],b[:2])).clamp(min=0)
    inter=wh[0]*wh[1]
    aa=(a[2]-a[0]).clamp(min=0)*(a[3]-a[1]).clamp(min=0)
    bb=(b[2]-b[0]).clamp(min=0)*(b[3]-b[1]).clamp(min=0)
    return inter/(aa+bb-inter+1e-9)


def evaluate_predictions(predictions,ground_truths, threshold=.25):
    """Single-class continuous AP50 using sorted detections and one GT per image."""
    total=len(ground_truths)
    pool=[];tp_at_threshold=fp_at_threshold=0
    for image_id,(preds,gt) in enumerate(zip(predictions,ground_truths)):
        g=[float(gt[0]-gt[2]/2),float(gt[1]-gt[3]/2),float(gt[0]+gt[2]/2),float(gt[1]+gt[3]/2)]
        for box,score in preds:
            iou=_iou_xyxy_plain(box,g)
            pool.append((score,image_id,iou))
    pool.sort(key=lambda x:x[0],reverse=True)
    seen=set();tps=fps=0;precisions=[];recalls=[]
    for score,image_id,iou in pool:
        hit=iou>=.5 and image_id not in seen
        if hit:seen.add(image_id);tps+=1
        else:fps+=1
        precisions.append(tps/(tps+fps));recalls.append(tps/total)
        if score>=threshold:
            tp_at_threshold+=int(hit);fp_at_threshold+=int(not hit)
    ap=0.;last=0.
    # All-points interpolated AP (VOC/COCO-like PR area at IoU=.5, single class).
    for r,p in zip(recalls,precisions):
        if r>last:
            idx=recalls.index(r)
            ap+=(r-last)*max(precisions[idx:]);last=r
    return {'ap50':float(ap),'tp':tp_at_threshold,'fp':fp_at_threshold,'fn':total-tp_at_threshold,
            'precision':tp_at_threshold/max(1,tp_at_threshold+fp_at_threshold),
            'recall':tp_at_threshold/total,'num_gt':total}


def _iou_xyxy_plain(a,b):
    wh=[max(0.,min(float(a[i+2]),b[i+2])-max(float(a[i]),b[i])) for i in range(2)]
    inter=wh[0]*wh[1];aa=max(0.,float(a[2])-float(a[0]))*max(0.,float(a[3])-float(a[1]));bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb-inter>0 else 0.
