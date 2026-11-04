#!/usr/bin/env python3
"""Export real torch tensors from the Nano topology to a static offline web lab.

Default uses fixed-seed RANDOM weights to demonstrate genuine layer arithmetic.
Never describe random activations as learned head/person features.
For an official 80-class checkpoint, add --weights /path/to/yolox_nano.pth
and verify its provenance/LICENSE before publishing the derived trace.

Run from repo root:
python3 tools/unit03b/export_trace.py
python3 tools/unit03b/export_trace.py --weights ./yolox_nano.pth \
    --output /tmp/trained_trace.json --preview /tmp/trained_preproc.jpg
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from collections import OrderedDict
import numpy as np
from PIL import Image
import torch

from nano_reference import YOLOX, count_params

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_IMAGE=ROOT/'site/assets/003Z-selfie.png'
DEFAULT_OUTPUT=ROOT/'site/lab/03-nano/trace.json'
DEFAULT_PREVIEW=ROOT/'site/lab/03-nano/input416.jpg'
IMAGE_SIZE=416
STAGES=OrderedDict([
  ('focus',('backbone.backbone.stem','Focus（空間→チャネル）＋Conv / stride2','入力RGBを偶奇位置4組に並び替え、12chを3×3 Convで16chへ変換。')),
  ('dark2',('backbone.backbone.dark2','dark2 / stride4','Depthwise Conv＋CSPLayer、特徴は32ch。')),
  ('dark3',('backbone.backbone.dark3','dark3 / stride8','1/8解像度の特徴を作る。後続PAFPNの入力となる。')),
  ('dark4',('backbone.backbone.dark4','dark4 / stride16','1/16解像度、CSP Bottleneckを3個持つ。')),
  ('dark5',('backbone.backbone.dark5','dark5 / stride32','1/32解像度、SPPとCSPLayerを含む。')),
])
BRANCHES=OrderedDict([
  ('branch1',('backbone.backbone.dark3.1.conv1','Conv1 → Bottleneckへ','入力64chを1×1 Convで32chへ縮小')),
  ('branch1_out',('backbone.backbone.dark3.1.m','Bottleneck群の出力','3個のBottleneckを通過した32chの特徴')),
  ('branch2',('backbone.backbone.dark3.1.conv2','Shortcut側のConv2','別の1×1 Convで32chを生成')),
  ('branch_concat',('concat','Concat（実測の2経路を結合）','Bottleneck側32ch＋Shortcut側32ch＝64ch')),
  ('branch_output',('backbone.backbone.dark3.1.conv3','最後の1×1 Conv','Concat後64chから出力64chへ変換')),
])


def prepare_image(path:Path,preview:Path):
    im=Image.open(path).convert('RGB')
    original=list(im.size)
    scale=min(IMAGE_SIZE/im.width,IMAGE_SIZE/im.height)
    rw=max(1,int(im.width*scale));rh=max(1,int(im.height*scale))
    resized=im.resize((rw,rh),Image.Resampling.BILINEAR)
    padded=Image.new('RGB',(IMAGE_SIZE,IMAGE_SIZE),(114,114,114))
    padded.paste(resized,(0,0)) # reproduce the top-left letterbox location
    preview.parent.mkdir(parents=True,exist_ok=True)
    padded.save(preview,quality=90,optimize=True)
    # Reference YOLOX demo supplies BGR in [0,255], no mean/std normalization.
    a=np.asarray(padded,dtype=np.uint8)[:,:,::-1].copy()
    t=torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0).float()
    return t,original,[rw,rh]


def sample_channels(value:torch.Tensor,max_count=3):
    value=value.detach().cpu().float().squeeze(0)
    C,H,W=value.shape
    v=value.view(C,-1)
    variances=v.var(dim=1,unbiased=False)
    idx=[0,int(variances.argmax()),C//2]
    if C>3: idx+=[C-1]
    indices=list(dict.fromkeys(idx))[:max_count]
    chans=[]
    for ci in indices:
        t=value[ci].numpy()
        chans.append({
            'index':ci,
            'min':round(float(t.min()),6),
            'max':round(float(t.max()),6),
            'mean':round(float(t.mean()),6),
            'std':round(float(t.std()),6),
            'values':np.round(t.astype(np.float64),6).reshape(-1).tolist(),
        })
    return chans


def describe_stage(module, val, label, info, stride=None):
    t=val.detach().cpu().float()
    return {'name':label,'description':info,'shape':list(t.shape),
        'stride':stride,'paramCount':count_params(module) if module is not None else 0,
        'channels':sample_channels(t),'min':round(float(t.min()),6),
        'max':round(float(t.max()),6),
        'mean':round(float(t.mean()),6),
        'std':round(float(t.std(unbiased=False)),6)}


def load_checkpoint(model,weights):
    # weights_only=True: don't unpickle arbitrary Python objects.
    ckpt=torch.load(weights,map_location='cpu',weights_only=True)
    if not isinstance(ckpt,dict):raise ValueError('Checkpoint must be a dictionary')
    state=ckpt.get('model',ckpt)
    if not isinstance(state,dict):raise ValueError('No model state_dict')
    missing,unexpected=model.load_state_dict(state,strict=False)
    if missing or unexpected:
        raise ValueError(f'Checkpoint not compatible with official 80-class Nano model. '
                         f'Missing {missing[:8]} / unexpected {unexpected[:8]}')
    return hashlib.sha256(Path(weights).read_bytes()).hexdigest()


def export(image:Path, output:Path, preview:Path, weights:Path|None,seed:int=20261104, checkpoint_status:str='unverified'):
    torch.manual_seed(seed)
    torch.set_num_threads(min(2,torch.get_num_threads()))
    model=YOLOX(num_classes=80).eval()
    if not weights and checkpoint_status!='unverified':
        raise ValueError('Cannot declare verified-pretrained without a checkpoint')
    ck_sha=load_checkpoint(model,weights) if weights else None
    inp,orig,resized=prepare_image(image,preview)
    collected={}
    hooks=[]
    for stage,path_details in list(STAGES.items())+list(BRANCHES.items()):
        module_path=path_details[0]
        if module_path=='concat':continue
        module=model.get_submodule(module_path)
        hooks.append(module.register_forward_hook(lambda m,args,out,k=stage:collected.__setitem__(k,out.detach().clone())))
    with torch.inference_mode():
        outputs=model(inp)
    for h in hooks:h.remove()
    expected={"focus":[1,16,208,208],"dark2":[1,32,104,104],"dark3":[1,64,52,52],"dark4":[1,128,26,26],"dark5":[1,256,13,13]}
    for key,shape in expected.items():
        assert list(collected[key].shape)==shape, (key,collected[key].shape)
    collected['branch_concat']=torch.cat((collected['branch1_out'],collected['branch2']),dim=1)
    for i,output_t in enumerate(outputs):
        assert list(output_t.shape)==[1,85,416//[8,16,32][i],416//[8,16,32][i]]
    stages={}
    backbone=model.backbone.backbone
    for i,(name,(module_path,label,info)) in enumerate(STAGES.items()):
        stages[name]=describe_stage(model.get_submodule(module_path),collected[name],label,info,2**(i+1))
        stages[name]['code']=module_path
    branches={}
    for name,(module_path,label,info) in BRANCHES.items():
        module=model.get_submodule(module_path) if module_path!='concat' else None
        branches[name]=describe_stage(module,collected[name],label,info,8)
        branches[name]['code']=module_path
    feature_paths=["backbone.C3_p3","backbone.C3_n3","backbone.C3_n4"]
    head_params=count_params(model.head)
    bb_params=count_params(model.backbone.backbone)
    neck_params=count_params(model.backbone)-bb_params
    meta={
      'schema':'yolox-academy.nano-trace.v1',
      'model':'YOLOX-Nano (YOLOX public experiment topology)',
      'source':'https://github.com/Megvii-BaseDetection/YOLOX',
      'experimental_config':{'depth':0.33,'width':0.25,'depthwise':True,'classes':80,'input':[1,3,416,416]},
      'params':{'total':count_params(model),'backbone':bb_params,'neck':neck_params,'head':head_params},
      'provenance':{
          'type':('verified-pretrained' if checkpoint_status=='verified-pretrained' else 'checkpoint-loaded-unverified') if weights else 'random-initialized',
          'trained':True if weights and checkpoint_status=='verified-pretrained' else (None if weights else False),
          'seed':None if weights else seed,
          'checkpoint_sha256':ck_sha,
          'warning':('This is an actual forward pass with randomly initialized weights. '
                     'It is NOT a trained detector and its feature maps do NOT represent learned semantics.' if not weights else
                     'This is a forward pass using an externally supplied compatible checkpoint. Training status is unverified unless explicitly attested by the exporter. Verify license and provenance before redistribution.')},
      'preprocess':{'originalWH':orig,'resizedWH':resized,'inputWH':[416,416],
                    'padded':'top-left; pad remaining pixels with RGB 114',
                    'channel_order':'BGR (for official YOLOX demo compatibility)',
                    'input_range':'0..255 float32, no /255 scaling',
                    'resize':'Pillow bilinear; results may differ slightly from OpenCV demo preprocessing',
                    'preview':'input416.jpg'},
      'stages':stages,'branches':branches,
      'note':'P3/P4/P5 become final FPN outputs only after PAFPN; dark3/dark4/dark5 are backbone features.',
      'fpn':{'shapes':[list(t.shape) for t in [model.backbone(inp)[i] for i in range(3)]],
             'strides':[8,16,32]},
      'head':{'rawShapes':[list(o.shape) for o in outputs],
              'meaning':'raw regression 4, objectness 1, classification 80. No postprocessing performed.'}
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(meta,separators=(',',':'),ensure_ascii=False),encoding='utf-8')
    print('Model params:',meta['params'])
    print('Stages:')
    for k,v in stages.items():print(' ',k,v['shape'],'params',v['paramCount'])
    print('Output:',output,'size',output.stat().st_size,'bytes')
    return meta


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--image',type=Path,default=DEFAULT_IMAGE)
    a.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    a.add_argument('--preview',type=Path,default=DEFAULT_PREVIEW)
    a.add_argument('--weights',type=Path)
    a.add_argument('--seed',type=int,default=20261104)
    a.add_argument('--checkpoint-status',choices=['unverified','verified-pretrained'],default='unverified',
                   help='Only assert verified-pretrained after independently checking origin and training status')
    args=a.parse_args()
    export(args.image,args.output,args.preview,args.weights,args.seed,args.checkpoint_status)

if __name__=='__main__':main()
