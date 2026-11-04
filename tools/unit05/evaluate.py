"""Reload independently stored local model weights and recompute AP50."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from tools.unit05.train import make_model,eval_split,state_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--split',choices=['train','val'],default='val');args=p.parse_args()
    torch.set_num_threads(2)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=make_model(ckpt['seed']).eval()
    model.load_state_dict(ckpt['state'],strict=True)
    sha=state_sha(model)
    if sha!=ckpt['source_state_sha256']:raise RuntimeError('Weights SHA mismatch')
    result,_=eval_split(model,args.split,ckpt['train_count'])
    print(json.dumps({'split':args.split,'steps':ckpt['steps'],'train_count':ckpt['train_count'],
                      'sha256':sha,'metrics':result},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
