"""Verify local checkpoints, export immutable public metrics + bbox previews only.

The .pt checkpoints are intentionally NOT part of the published website.
"""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from tools.unit05.train import make_model,eval_split,state_sha
from tools.unit05.dataset import sample,preview_png,SEED
from tools.unit05.core import predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'out'/'unit05')
    p.add_argument('--site',type=Path,default=ROOT/'site'/'lab'/'05-training');a=p.parse_args()
    torch.set_num_threads(2)
    a.site.mkdir(parents=True,exist_ok=True)
    models={}
    for key,stem,count in [('one','overfit',1),('eight','smallset',8)]:
        d=json.loads((a.input/f'{stem}.json').read_text('utf-8'))
        c=torch.load(a.input/f'{stem}.pt',map_location='cpu',weights_only=True)
        m=make_model(SEED);m.load_state_dict(c['state']);m.eval()
        assert c['steps']==600 and c['train_count']==count and d['provenance']['parameter_count']==896754
        assert state_sha(m)==c['source_state_sha256']==d['snapshots'][-1]['weight_sha256']
        tr,_=eval_split(m,'train',count);va,_=eval_split(m,'val',count)
        assert math.isclose(tr['ap50'],d['snapshots'][-1]['training']['ap50'],abs_tol=1e-12)
        assert math.isclose(va['ap50'],d['snapshots'][-1]['validation']['ap50'],abs_tol=1e-12)
        all_preds=[]
        for i in range(6):
            img,_=sample(i,'val')
            preds=predict(m,torch.from_numpy(img[None]),score_threshold=1.e-5,topk=50)
            all_preds.append({'image_index':i,'predictions':[{'box':[round(float(x),4) for x in b],'score':round(float(s),9)} for b,s in preds[:10]]})
        d['final_predictions']=all_preds
        models[key]=d
    examples=[]
    for i in range(6):
        (a.site/f'val-{i}.png').write_bytes(preview_png(i,'val'))
        _,gt=sample(i,'val');examples.append({'index':i,'image':f'val-{i}.png','gt':[float(x) for x in gt]})
    data={'schema':'yva.unit05.public.v1','provenance':{
        'training':'actual CPU PyTorch updates','seed':SEED,'dataset':'synthetic rectangle only',
        'architecture':'YOLOX-Nano compatible 1-class 896754 parameters','input_size':128,
        'trained_weights_public':False,'checkpoint_reloaded_and_verified':True,
        'caution':'The original v0.5 work area was lost before packaging; these results are from a reconstructed synthetic dataset and are NOT the earlier recorded AP50 values.',
        'validation_count':6,'not_head_detector':True},'validation_examples':examples,'experiments':models}
    (a.site/'trace05.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print('PASS: publication JSON generated; checkpoint hash and train/val AP50 independently re-evaluated')
    for key,m in models.items():
        print(key,m['snapshots'][-1]['training']['ap50'],m['snapshots'][-1]['validation']['ap50'])
if __name__=='__main__':main()
