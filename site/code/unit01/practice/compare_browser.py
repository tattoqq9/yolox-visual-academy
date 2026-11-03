"""ブラウザから書き出した48x32 RGB CSVを使用して出力を再現する。"""
from pathlib import Path
import argparse,csv,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solutions'))
from cnn_reference import rgb_to_gray, conv2d_single

FILTERS={
 'blur': np.ones((3,3),dtype=np.float64)/9,
 'sx': np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float64),
 'sy': np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float64),
 'sharp': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float64)
}

def load_rgb(path):
    out=np.zeros((32,48,3),dtype=np.uint8)
    with open(path,encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            x=int(row['x']);y=int(row['y']);out[y,x]=[int(row[k]) for k in ('R','G','B')]
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('x',type=int);p.add_argument('y',type=int);p.add_argument('kernel',choices=FILTERS)
    a=p.parse_args();assert 0<=a.x<48 and 0<=a.y<32
    gray=rgb_to_gray(load_rgb(a.csv));out=conv2d_single(gray,FILTERS[a.kernel],1,1)
    print(f'position=({a.x},{a.y}), kernel={a.kernel}, output={out[a.y,a.x]:.9f}')
    print('shape=',out.shape,'min=',out.min(),'max=',out.max())
if __name__=='__main__':main()
