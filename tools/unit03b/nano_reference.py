"""Educational, independently written YOLOX-Nano-compatible *inference topology*.

Mirrors the public 2021 Megvii YOLOX Nano experiment architecture:
  depth=0.33, width=0.25, depthwise=True, image size=416, COCO=80 classes.
This is not a replacement for the original training/inference toolchain.
Reference: https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0).

All tensors are NCHW. A random-initialized model has NOT learned to detect heads.
Only use a compatible 80-class official checkpoint for trained activations.
"""
from __future__ import annotations
import math
import torch
from torch import nn

class BaseConv(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, (kernel-1)//2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DWConv(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1):
        super().__init__()
        self.dconv = BaseConv(cin, cin, kernel, stride, groups=cin)
        self.pconv = BaseConv(cin, cout, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pconv(self.dconv(x))

class Bottleneck(nn.Module):
    def __init__(self, cin: int, cout: int, shortcut: bool = True, expansion: float = .5, depthwise: bool = False):
        super().__init__()
        hidden = int(cout*expansion)
        self.conv1 = BaseConv(cin, hidden, 1)
        self.conv2 = (DWConv if depthwise else BaseConv)(hidden, cout, 3)
        self.use_add = shortcut and cin == cout
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x+y if self.use_add else y

class CSPLayer(nn.Module):
    def __init__(self, cin: int, cout: int, n: int = 1, shortcut: bool = True, expansion: float = .5, depthwise: bool = False):
        super().__init__()
        hidden = int(cout*expansion)
        self.conv1 = BaseConv(cin, hidden, 1)
        self.conv2 = BaseConv(cin, hidden, 1)
        self.conv3 = BaseConv(hidden*2, cout, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden,hidden,shortcut,1.,depthwise) for _ in range(n)))
    def forward(self,x):
        a = self.m(self.conv1(x))
        b = self.conv2(x)
        return self.conv3(torch.cat((a,b),1))

class Focus(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int = 3):
        super().__init__()
        self.conv = BaseConv(cin*4, cout, kernel)
    def forward(self,x):
        # Preserve official channel ordering: TL, BL, TR, BR.
        x = torch.cat((x[...,::2,::2], x[...,1::2,::2], x[...,::2,1::2], x[...,1::2,1::2]), dim=1)
        return self.conv(x)

class SPPBottleneck(nn.Module):
    def __init__(self,cin:int,cout:int,kernel_sizes=(5,9,13)):
        super().__init__()
        hidden=cin//2
        self.conv1=BaseConv(cin,hidden,1)
        self.m=nn.ModuleList(nn.MaxPool2d(k, stride=1, padding=k//2) for k in kernel_sizes)
        self.conv2=BaseConv(hidden*(len(kernel_sizes)+1),cout,1)
    def forward(self,x):
        x=self.conv1(x)
        return self.conv2(torch.cat([x]+[pool(x) for pool in self.m],1))

class CSPDarknet(nn.Module):
    def __init__(self,depth:float=.33,width:float=.25,depthwise:bool=True):
        super().__init__()
        C=int(64*width)
        D=max(round(3*depth),1)
        Conv=DWConv if depthwise else BaseConv
        self.stem=Focus(3,C,3)
        self.dark2=nn.Sequential(Conv(C,2*C,3,2),CSPLayer(2*C,2*C,D,depthwise=depthwise))
        self.dark3=nn.Sequential(Conv(2*C,4*C,3,2),CSPLayer(4*C,4*C,D*3,depthwise=depthwise))
        self.dark4=nn.Sequential(Conv(4*C,8*C,3,2),CSPLayer(8*C,8*C,D*3,depthwise=depthwise))
        self.dark5=nn.Sequential(Conv(8*C,16*C,3,2),SPPBottleneck(16*C,16*C),CSPLayer(16*C,16*C,D,shortcut=False,depthwise=depthwise))
    def forward(self,x):
        x=self.stem(x)
        d2=self.dark2(x)
        d3=self.dark3(d2)
        d4=self.dark4(d3)
        d5=self.dark5(d4)
        return {"dark3":d3,"dark4":d4,"dark5":d5}

class YOLOPAFPN(nn.Module):
    def __init__(self,depth:float=.33,width:float=.25,depthwise:bool=True):
        super().__init__()
        c3,c4,c5=(int(v*width) for v in (256,512,1024))
        Conv=DWConv if depthwise else BaseConv
        self.backbone=CSPDarknet(depth,width,depthwise)
        self.upsample=nn.Upsample(scale_factor=2,mode="nearest")
        self.lateral_conv0=BaseConv(c5,c4,1)
        self.C3_p4=CSPLayer(2*c4,c4,round(3*depth),shortcut=False,depthwise=depthwise)
        self.reduce_conv1=BaseConv(c4,c3,1)
        self.C3_p3=CSPLayer(2*c3,c3,round(3*depth),shortcut=False,depthwise=depthwise)
        self.bu_conv2=Conv(c3,c3,3,2)
        self.C3_n3=CSPLayer(2*c3,c4,round(3*depth),shortcut=False,depthwise=depthwise)
        self.bu_conv1=Conv(c4,c4,3,2)
        self.C3_n4=CSPLayer(2*c4,c5,round(3*depth),shortcut=False,depthwise=depthwise)
    def forward(self,x):
        b=self.backbone(x)
        x2,x1,x0=(b[k] for k in ("dark3","dark4","dark5"))
        fpn0=self.lateral_conv0(x0)
        upper0=self.C3_p4(torch.cat([self.upsample(fpn0),x1],dim=1))
        fpn1=self.reduce_conv1(upper0)
        p3=self.C3_p3(torch.cat([self.upsample(fpn1),x2],dim=1))
        p4=self.C3_n3(torch.cat([self.bu_conv2(p3),fpn1],dim=1))
        p5=self.C3_n4(torch.cat([self.bu_conv1(p4),fpn0],dim=1))
        return p3,p4,p5

class YOLOXHead(nn.Module):
    def __init__(self,num_classes:int=80,width:float=.25,depthwise:bool=True):
        super().__init__()
        self.num_classes=num_classes
        self.stems=nn.ModuleList()
        self.cls_convs=nn.ModuleList()
        self.reg_convs=nn.ModuleList()
        self.cls_preds=nn.ModuleList()
        self.reg_preds=nn.ModuleList()
        self.obj_preds=nn.ModuleList()
        Conv=DWConv if depthwise else BaseConv
        features=int(256*width)
        for cin0 in (256,512,1024):
            cin=int(cin0*width)
            self.stems.append(BaseConv(cin,features,1))
            self.cls_convs.append(nn.Sequential(Conv(features,features,3),Conv(features,features,3)))
            self.reg_convs.append(nn.Sequential(Conv(features,features,3),Conv(features,features,3)))
            self.cls_preds.append(nn.Conv2d(features,num_classes,1))
            self.reg_preds.append(nn.Conv2d(features,4,1))
            self.obj_preds.append(nn.Conv2d(features,1,1))
        self.strides=(8,16,32)
    def initialize_biases(self,prior=.01):
        with torch.no_grad():
            v=-math.log((1-prior)/prior)
            for m in list(self.cls_preds)+list(self.obj_preds):m.bias.fill_(v)
    def forward(self,feats):
        # Return raw feature maps, not decoded detections. Layout matches official.
        out=[]
        for i,f in enumerate(feats):
            x=self.stems[i](f)
            cls=self.cls_preds[i](self.cls_convs[i](x))
            r=self.reg_convs[i](x)
            reg=self.reg_preds[i](r)
            obj=self.obj_preds[i](r)
            out.append(torch.cat((reg,obj,cls),dim=1))
        return tuple(out)

class YOLOX(nn.Module):
    def __init__(self,num_classes:int=80):
        super().__init__()
        self.backbone=YOLOPAFPN(.33,.25,True)
        self.head=YOLOXHead(num_classes,.25,True)
        for layer in self.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.eps=1e-3
                layer.momentum=.03
        self.head.initialize_biases(.01)
    def forward(self,x):
        return self.head(self.backbone(x))


def count_params(model:nn.Module)->int:
    return sum(v.numel() for v in model.parameters())


def verify_shapes(model:YOLOX, image_size:int=416) -> dict[str,list[int]]:
    stages={}
    def observe(name):
        def hook(module,inputs,output):stages[name]=list(output.shape)
        return hook
    names=["backbone.backbone.stem","backbone.backbone.dark2","backbone.backbone.dark3","backbone.backbone.dark4","backbone.backbone.dark5"]
    hooks=[]
    for n in names: hooks.append(model.get_submodule(n).register_forward_hook(observe(n)))
    model.eval()
    with torch.inference_mode(): model(torch.zeros(1,3,image_size,image_size))
    for h in hooks:h.remove()
    return stages

if __name__=="__main__":
    torch.set_num_threads(2)
    m=YOLOX(80).eval()
    print("YOLOX Nano compatible params (80-class):",count_params(m))
    print("stage shapes:",verify_shapes(m))
