"""ブラウザで保存したunit02_gray16.csvを学生の畳み込み関数で再計算。"""
import argparse
from pathlib import Path
import numpy as np
from student import conv2d_same

KERNELS={
 'edge':np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=float),
 'blur':np.ones((3,3),dtype=float)/9,
 'identity':np.array([[0,0,0],[0,1,0],[0,0,0]],dtype=float),
}

def main():
    p=argparse.ArgumentParser(description='Unit02 のブラウザ16x16画像でCNNを再現する')
    p.add_argument('csv',type=Path,help='Unit02実験室から保存したunit02_gray16.csv')
    p.add_argument('--stride',type=int,default=1,choices=[1,2,4])
    p.add_argument('--kernel',choices=KERNELS,default='edge')
    p.add_argument('--ox',type=int,default=0)
    p.add_argument('--oy',type=int,default=0)
    args=p.parse_args()
    src=np.loadtxt(args.csv,delimiter=',',dtype=float)
    if src.shape!=(16,16):raise ValueError(f'CSV shape must be 16x16, got {src.shape}')
    out=conv2d_same(src,KERNELS[args.kernel],args.stride)
    if not(0<=args.oy<out.shape[0] and 0<=args.ox<out.shape[1]):raise ValueError('指定座標が出力サイズを超えています')
    print('入力',src.shape,'kernel',args.kernel,'stride',args.stride,'出力',out.shape)
    print(f'出力 (ox={args.ox}, oy={args.oy}) = {out[args.oy,args.ox]:.6f}')
    print('ブラウザで同じstride・Kernel・出力座標を選んで比較してください。')

if __name__=='__main__':main()
