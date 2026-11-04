"""Train a genuine one-class 896754-parameter YOLOX-Nano on synthetic rectangles.

No official checkpoint or private dataset is read. Saves weights ONLY locally.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.unit03b.nano_reference import YOLOX,count_params
from tools.unit05.dataset import sample,boxes_for_split,preview_png,SEED
from tools.unit05.core import train_step_loss,predict,evaluate_predictions


def make_model(seed=SEED):
    torch.manual_seed(seed)
    model=YOLOX(num_classes=1)
    assert count_params(model)==896754,count_params(model)
    return model


def eval_split(model,split,train_count=8,preview_count=10):
    collection=boxes_for_split(split,train_count)
    preds=[];gts=[];views=[]
    for index,(image,gt) in enumerate(collection):
        p=predict(model,torch.from_numpy(image[None]),score_threshold=1.e-5,nms_threshold=.5,topk=50)
        preds.append(p);gts.append(gt)
        if index<1:views=[{'image_index':index,'predictions':[
                    {'box':[round(float(q),3) for q in coords],'score':round(float(sc),8)} for coords,sc in p[:preview_count]]}]
    return eval_predictions_compat(preds,gts),views


def eval_predictions_compat(preds,gts):return evaluate_predictions(preds,gts,threshold=.25)


def state_sha(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(train_count:int,steps:int,seed:int=SEED,lr:float=.001,output:Path|None=None,checkpoint:Path|None=None):
    if train_count not in {1,8} or steps<1:raise ValueError('train_count must be 1 or 8 and steps positive')
    torch.set_num_threads(2)
    model=make_model(seed);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0005)
    data=[(torch.from_numpy(im[None]),torch.from_numpy(box)) for im,box in boxes_for_split('train',train_count)]
    snaps={0,1,5,25,50,100,200,400,600,steps};snaps={s for s in snaps if s<=steps}
    events=[];start=time.monotonic()
    def checkpoint_trace(step,loss_parts):
        model.eval()
        train_metrics,_=eval_split(model,'train',train_count)
        val_metrics,val_views=eval_split(model,'val',train_count)
        sha=state_sha(model)
        events.append({'step':step,'elapsed_s':round(time.monotonic()-start,2),'loss':loss_parts,
                       'training':train_metrics,'validation':val_metrics,'preview':val_views,
                       'weight_sha256':sha})
        print(f'train={train_count} step={step:3d}: train AP50={train_metrics["ap50"]:.4f},'
              f' val AP50={val_metrics["ap50"]:.4f}, SHA={sha[:10]}',flush=True)
    checkpoint_trace(0,None)
    for step in range(1,steps+1):
        im,gt=data[(step-1)%train_count]
        model.train();opt.zero_grad(set_to_none=True)
        total,parts=train_step_loss(model(im),gt)
        if not torch.isfinite(total):raise RuntimeError(f'Non-finite loss at step {step}: {parts}')
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=10.0)
        opt.step()
        parts={'total':round(float(total.detach()),6),**{k:(round(v,6) if isinstance(v,float) else v) for k,v in parts.items()}}
        if step in snaps:checkpoint_trace(step,parts)
    result={'schema':'yva.unit05.training.v1','provenance':{'kind':'real-pytorch-weight-updates',
         'task':'one-class synthetic rectangles, NOT real heads','architecture':'YOLOX-Nano topology',
         'seed':seed,'train_count':train_count,'val_count':6,'input':128,'steps':steps,'optimizer':'AdamW',
         'learning_rate':lr,'parameter_count':count_params(model),'iou_threshold':.5,
         'metric':'continuous interpolated AP50, single class, 1 GT per image',
         'training_simplifications':['no Mosaic','no EMA','no warmup/scheduler','batch 1','one GT/image','geometry-only SimOTA','synthetic images']},
         'snapshots':events}
    if output:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if checkpoint:
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state':model.state_dict(),'train_count':train_count,'seed':seed,'steps':steps,
                    'source_state_sha256':state_sha(model)},checkpoint)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--train-count',type=int,choices=[1,8],default=8)
    ap.add_argument('--seed',type=int,default=SEED);ap.add_argument('--trace',type=Path)
    ap.add_argument('--checkpoint',type=Path);ap.add_argument('--preview-dir',type=Path)
    a=ap.parse_args();s=train(a.train_count,a.steps,a.seed,output=a.trace,checkpoint=a.checkpoint)
    if a.preview_dir:
        a.preview_dir.mkdir(parents=True,exist_ok=True)
        for i in range(6):(a.preview_dir/f'val-{i}.png').write_bytes(preview_png(i,'val'))
    print(json.dumps({'train_count':a.train_count,'steps':a.steps,'final':s['snapshots'][-1]['validation']},ensure_ascii=False))
if __name__=='__main__':main()
