#!/usr/bin/env python3
"""Export authentic *computed* YOLOPAFPN + decoupled head tensors for Unit03-C.

Uses the independently written YOLOX Nano-compatible inference topology from
Unit03-B. Default model weights are randomly initialized with a fixed seed.
For educational inspection only: NOT learned object/head detection.

Model architecture based on https://github.com/Megvii-BaseDetection/YOLOX
(Apache-2.0), especially yolox/models/yolo_pafpn.py and yolo_head.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'/'unit03b'))
from nano_reference import YOLOX, count_params
from export_trace import prepare_image

IMG = ROOT/'site/assets/003Z-selfie.png'
OUTPUT = ROOT/'site/lab/03-fpn-head/trace.json'
SEED=20261104

# Source names, operations and the actual module paths in Unit03-B reference.
NODES=OrderedDict([
('dark3',('Backbone dark3','dark3 → PAFPN入力','backbone.backbone.dark3','backbone',[1,64,52,52],[],'1/8 解像度の特徴。高解像度経路と結合します。')),
('dark4',('Backbone dark4','dark4 → PAFPN入力','backbone.backbone.dark4','backbone',[1,128,26,26],[],'1/16 解像度。Top-downで深い特徴と結合します。')),
('dark5',('Backbone dark5','dark5 → PAFPN入力','backbone.backbone.dark5','backbone',[1,256,13,13],[],'1/32 解像度。最も深い入力特徴です。')),
('lateral',('Lateral 1×1 Conv','256 → 128 ch','backbone.lateral_conv0','topdown',[1,128,13,13],['dark5'],'dark5のチャネル数を半分にします。')),
('up0',('Nearest Upsample ×2','13×13 → 26×26',None,'topdown',[1,128,26,26],['lateral'],'最近傍補間で各値を縦横2マスに複製します。学習パラメータなし。')),
('cat0',('Concat + dark4','128 + 128 → 256 ch',None,'topdown',[1,256,26,26],['up0','dark4'],'同じ空間座標の2テンソルをチャネル方向に結合します。加算しません。')),
('tdp4',('C3_p4 / CSP','256 → 128 ch','backbone.C3_p4','topdown',[1,128,26,26],['cat0'],'CSPの分岐とConvにより結合した特徴を変換します。')),
('reduce',('Reduce 1×1 Conv','128 → 64 ch','backbone.reduce_conv1','topdown',[1,64,26,26],['tdp4'],'高解像度へ伝える特徴を64chに縮約します。')),
('up1',('Nearest Upsample ×2','26×26 → 52×52',None,'topdown',[1,64,52,52],['reduce'],'64chのまま最近傍補間で52×52へ拡大します。')),
('cat1',('Concat + dark3','64 + 64 → 128 ch',None,'topdown',[1,128,52,52],['up1','dark3'],'浅い位置情報を、上位から伝わった特徴と結合します。')),
('p3',('C3_p3 → P3','64ch × 52×52','backbone.C3_p3','topdown',[1,64,52,52],['cat1'],'Top-downの終端。後段のBottom-upとHeadに渡されます。')),
('down0',('Bottom-up stride 2','52×52 → 26×26','backbone.bu_conv2','bottomup',[1,64,26,26],['p3'],'DWConvのstride2で空間解像度を半分にします。')),
('cat2',('Concat + reduce','64 + 64 → 128 ch',None,'bottomup',[1,128,26,26],['down0','reduce'],'浅い解像度の特徴とTop-downの中間特徴を結合します。')),
('p4',('C3_n3 → P4','128ch × 26×26','backbone.C3_n3','bottomup',[1,128,26,26],['cat2'],'Bottom-upで再統合した中解像度特徴です。')),
('down1',('Bottom-up stride 2','26×26 → 13×13','backbone.bu_conv1','bottomup',[1,128,13,13],['p4'],'stride2でP4から深い特徴マップへ縮小します。')),
('cat3',('Concat + lateral','128 + 128 → 256 ch',None,'bottomup',[1,256,13,13],['down1','lateral'],'Bottom-upとLateralをチャネル方向で結合。')),
('p5',('C3_n4 → P5','256ch × 13×13','backbone.C3_n4','bottomup',[1,256,13,13],['cat3'],'Bottom-upの終端。P3/P4/P5が出揃います。')),
])

HEAD_META={
'stem':('1×1 Stem','64チャネルへ揃える','stems',64),
'cls_feat':('Classification Convs','分類側3×3 Conv ×2','cls_convs',64),
'reg_feat':('Regression Convs','回帰側3×3 Conv ×2','reg_convs',64),
'reg':('BBox Raw 4ch','tx,ty,tw,th (未デコード)','reg_preds',4),
'obj':('Objectness Raw 1ch','sigmoid前のlogit','obj_preds',1),
'cls':('Class Raw 80ch','sigmoid前の各クラスlogit','cls_preds',80),
'raw':('Concat Raw 85ch','[4 bbox, 1 obj, 80 cls]','',85),
}


def _sample_indices(t:torch.Tensor, ident:str, family:str, split:int|None=None):
    c=t.shape[1]
    if family=='reg': return list(range(4))
    if family=='obj': return [0]
    if family=='cls': return [0,1,7,79]
    if family=='raw': return [0,1,2,3,4,5,6,12,84]
    if ident in ('cat0','cat1','cat2','cat3'):
        assert split is not None
        return list(dict.fromkeys([0,split-1,split,c-1]))
    var=t[0].float().flatten(1).var(dim=1,unbiased=False)
    return list(dict.fromkeys([0,int(var.argmax()),c-1]))[:3]


def _stat_map(t:torch.Tensor,channel:int):
    arr=t[0,channel].detach().cpu().float().numpy()
    # Significant digits preserve very small near-zero random activations.
    # Fixed decimal rounding would turn deeper feature maps into all-zero arrays.
    f=lambda v:float(f'{float(v):.7g}')
    return dict(index=channel,min=f(arr.min()),max=f(arr.max()),
                mean=f(arr.mean()),values=[f(v) for v in arr.reshape(-1)])


def _serialize_node(ident:str, tensor:torch.Tensor, meta, model, split=None):
    name,subpath,modulepath,group,expected,inputs,description=meta
    shape=list(tensor.shape)
    if shape!=expected:raise AssertionError(f'{ident}: expected={expected} actual={shape}')
    module=model.get_submodule(modulepath) if modulepath else None
    indices=_sample_indices(tensor,ident,ident.rsplit('.',1)[-1],split)
    return dict(id=ident,name=name,subpath=subpath,description=description,
                group=group,shape=shape,params=count_params(module) if module else 0,
                modulePath=modulepath or 'torch.Tensor operation',inputs=inputs,
                sampleIndices=indices,
                samples=[_stat_map(tensor,k) for k in indices])


def export(img:Path=IMG,output:Path=OUTPUT,seed:int=SEED):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model=YOLOX(80).eval()
    input_tensor,original,resized=prepare_image(img,ROOT/'site/lab/03-nano/input416.jpg')
    collected:dict[str,torch.Tensor]={}
    input_to_csp={}
    hooks=[]
    for ident,meta in NODES.items():
        path=meta[2]
        if path:
            hooks.append(model.get_submodule(path).register_forward_hook(lambda m,args,out,k=ident:collected.__setitem__(k,out.detach().clone())))
    for ident in ('cat0','cat1','cat2','cat3'):
        path={'cat0':'backbone.C3_p4','cat1':'backbone.C3_p3','cat2':'backbone.C3_n3','cat3':'backbone.C3_n4'}[ident]
        hooks.append(model.get_submodule(path).register_forward_pre_hook(lambda m,args,k=ident:input_to_csp.__setitem__(k,args[0].detach().clone())))
    for i,level in enumerate(('p3','p4','p5')):
        for suffix,(_,_,modkey,_) in HEAD_META.items():
            if suffix=='raw': continue
            path=f'head.{modkey}.{i}'
            hooks.append(model.get_submodule(path).register_forward_hook(lambda m,args,out,k=f'{level}.{suffix}':collected.__setitem__(k,out.detach().clone())))
    with torch.inference_mode():outs=model(input_tensor)
    for h in hooks: h.remove()
    # Observe graph joins from true C3 inputs. Verify that every piece matches
    # the exact originating tensor; no hand-painted feature maps.
    with torch.inference_mode():
        collected['up0']=torch.nn.functional.interpolate(collected['lateral'],scale_factor=2,mode='nearest')
        collected['cat0']=input_to_csp['cat0']
        collected['up1']=torch.nn.functional.interpolate(collected['reduce'],scale_factor=2,mode='nearest')
        collected['cat1']=input_to_csp['cat1']
        collected['cat2']=input_to_csp['cat2']
        collected['cat3']=input_to_csp['cat3']
        expected_concat={
          'cat0':(collected['up0'],collected['dark4']),
          'cat1':(collected['up1'],collected['dark3']),
          'cat2':(collected['down0'],collected['reduce']),
          'cat3':(collected['down1'],collected['lateral']),
        }
        for key,(a,b) in expected_concat.items():
            torch.testing.assert_close(collected[key],torch.cat((a,b),dim=1),rtol=0,atol=0)
        for k,feat in enumerate(outs):collected[f'{("p3","p4","p5")[k]}.raw']=feat.detach().clone()
        for k,level in enumerate(('p3','p4','p5')):
            s=52//(2**k)
            assert tuple(collected[level].shape[-2:]) == (s,s)
            assert tuple(outs[k].shape)==(1,85,s,s)
            torch.testing.assert_close(outs[k],torch.cat((collected[level+'.reg'],collected[level+'.obj'],collected[level+'.cls']),dim=1),rtol=0,atol=0)
    data=OrderedDict()
    for ident,meta in NODES.items():
        split={'cat0':128,'cat1':64,'cat2':64,'cat3':128}.get(ident)
        data[ident]=_serialize_node(ident,collected[ident],meta,model,split)
    for k,level in enumerate(('p3','p4','p5')):
        h,w=collected[level].shape[-2:]
        for suffix,(name,sub,modkey,cout) in HEAD_META.items():
            ident=f'{level}.{suffix}'
            shape=[1,cout,h,w]
            module_path=f'head.{modkey}.{k}' if modkey else None
            inputs=[level] if suffix=='stem' else [ident.split('.')[0]+'.stem'] if suffix in ('cls_feat','reg_feat') else [level+'.cls_feat'] if suffix=='cls' else [level+'.reg_feat'] if suffix in ('reg','obj') else [level+'.reg',level+'.obj',level+'.cls']
            d=(name,sub,module_path,'head',shape,inputs,'80クラス構成の未学習重みを用いた実forward出力。')
            data[ident]=_serialize_node(ident,collected[ident],d,model)
    # Arithmetic proof at selected spatial positions, independent of visualization.
    checks={}
    for kind,source,joined,cut in [('up0','lateral','cat0',128),('up1','reduce','cat1',64)]:
        x,y=(3,5)
        source_val=float(collected[source][0,0,y,x])
        output_val=float(collected[kind][0,0,y*2,x*2])
        joined_val=float(collected[joined][0,0,y*2,x*2])
        assert source_val==output_val==joined_val
        checks[kind]={'srcXY':[x,y],'targetXY':[x*2,y*2],'value':round(source_val,6)}
    model_info={
      'schema':'yolox-academy.neck-head-trace.v1',
      'model':'YOLOX-Nano compatible inference topology, 80 classes',
      'reference':'https://github.com/Megvii-BaseDetection/YOLOX',
      'provenance':{'weights':'fixed-seed random initialization, NOT TRAINED','seed':seed,'checkpoint':None,'learned':False,
                    'warning':'These are real PyTorch forward tensors from UNTRAINED weights, not model predictions with semantic meaning.'},
      'input':{'image':'../../assets/003Z-selfie.png','preprocessed':'../03-nano/input416.jpg','size':[1,3,416,416],
               'originalWH':original,'resizedWH':resized,'letterbox':'top-left, RGB padding 114',
               'channelOrder':'BGR float32 0..255, no normalization'},
      'config':{'depth':.33,'width':.25,'depthwise':True,'numClasses':80,'headOutputsPerCell':85},
      'params':{'total':count_params(model),'backbone':count_params(model.backbone.backbone),'neck':count_params(model.backbone)-count_params(model.backbone.backbone),'head':count_params(model.head)},
      'counts':{'p3':52*52,'p4':26*26,'p5':13*13,'total':52*52+26*26+13*13,'rawScalars':85*(52*52+26*26+13*13)},
      'nodes':data,'checks':checks,
      'note':'Only a small subset of each feature tensor channel is serialized; raw logits are NOT decoded or sigmoid-transformed. In official eval mode, objectness/classes are sigmoid and BBoxes are decoded downstream.'
    }
    assert model_info['params']['total']==912159, model_info['params']
    assert model_info['counts']['total']==3549
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(model_info,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    print('saved',output,'bytes',output.stat().st_size,'nodes',len(data))
    print('param counts',model_info['params'],'head outputs',model_info['counts'])
    print('concat proofs',checks)
    return model_info

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--image',type=Path,default=IMG)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--seed',type=int,default=SEED)
    args=p.parse_args()
    export(args.image,args.output,args.seed)
