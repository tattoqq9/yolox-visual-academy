"""Reproducible synthetic single-rectangle examples. No real-person images or external files."""
from __future__ import annotations
import numpy as np
from PIL import Image

IMAGE_SIZE=128
SEED=20261104


def sample(index:int, split:str='train'):
    """Return normalized CHW float32 image and center/width/height pixel GT."""
    assert split in {'train','val'}
    rng=np.random.default_rng(SEED + (0 if split=='train' else 100000) + index*137)
    # Positions vary; the single-image baseline observes only index zero.
    w=int(rng.integers(20,39));h=int(rng.integers(20,39))
    x1=int(rng.integers(14, IMAGE_SIZE-14-w));y1=int(rng.integers(14, IMAGE_SIZE-14-h))
    bg=np.array([.16,.20,.25],dtype=np.float32)
    canvas=np.empty((IMAGE_SIZE,IMAGE_SIZE,3),dtype=np.float32);canvas[:]=bg
    # A local, reproducible texture encourages general rather than one-position matching.
    canvas+=rng.normal(0,.015,size=canvas.shape).astype('float32')
    canvas=np.clip(canvas,0,1)
    color=np.array([.92,.72,.27],dtype=np.float32)
    canvas[y1:y1+h,x1:x1+w]=color
    canvas[y1:y1+h,x1:x1+2]=[.98,.94,.82]
    canvas[y1:y1+h,x1+w-2:x1+w]=[.98,.94,.82]
    canvas[y1:y1+2,x1:x1+w]=[.98,.94,.82]
    canvas[y1+h-2:y1+h,x1:x1+w]=[.98,.94,.82]
    box=np.array([x1+w/2,y1+h/2,w,h],dtype=np.float32)
    return np.ascontiguousarray(canvas.transpose(2,0,1)),box


def preview_png(index:int,split:str='val')->bytes:
    import io
    x,_=sample(index,split)
    image=Image.fromarray(np.uint8(np.rint(x.transpose(1,2,0)*255)).clip(0,255),'RGB')
    out=io.BytesIO();image.save(out,format='PNG',optimize=True);return out.getvalue()


def boxes_for_split(split:str,train_count:int=8):
    n=train_count if split=='train' else 6
    return [sample(i,split) for i in range(n)]
