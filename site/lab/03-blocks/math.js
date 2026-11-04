/* Unit03-A deterministic educational math. No YOLOX weights, no inference.
 * Tensors are HWC Float64Array to keep spatial indexing explicit.
 * Exported both to browser and Node.js for numerical consistency tests.
 */
(function (global) {
  'use strict';
  const newTensor = (h,w,c,values=null) => ({h,w,c,values: values || new Float64Array(h*w*c)});
  const offset = (t,y,x,ch) => (y*t.w+x)*t.c+ch;
  const at = (t,y,x,ch) => (y<0||x<0||y>=t.h||x>=t.w) ? 0 : t.values[offset(t,y,x,ch)];
  const put = (t,y,x,ch,v) => {t.values[offset(t,y,x,ch)] = v;};
  const BASE = {
    blur: [1,1,1,1,1,1,1,1,1].map(v=>v/9),
    edgeX: [-1,0,1,-2,0,2,-1,0,1].map(v=>v/4),
    edgeY: [-1,-2,-1,0,0,0,1,2,1].map(v=>v/4),
    sharpen: [0,-1,0,-1,5,-1,0,-1,0]
  };
  const GAINS = [[.34,.46,.20],[.299,.587,.114],[.25,.50,.25],[.55,.25,.20]];
  const kernelBank = (preset='mixed') => {
    const bank=[];
    const kinds=preset==='blur'?['blur','blur','blur','blur'] : preset==='edge'?['edgeX','edgeY','edgeX','edgeY'] : ['blur','edgeX','edgeY','sharpen'];
    for(let oc=0;oc<4;oc++){
      const icarr=[];
      for(let ic=0;ic<3;ic++)icarr.push(BASE[kinds[oc]].map(v=>v*GAINS[oc][ic]));
      bank.push(icarr);
    }
    return bank;
  };
  // weights[outChannel][inChannel][kernelY*kw+kernelX]. Conv2d convention: cross correlation.
  function conv2d(input,weights,{stride=1,pad=0,k=3}={}){
    const h=Math.floor((input.h+2*pad-k)/stride)+1;
    const w=Math.floor((input.w+2*pad-k)/stride)+1;
    if(h<=0||w<=0||!Number.isInteger(h)||!Number.isInteger(w))throw Error('invalid convolution shape');
    const out=newTensor(h,w,weights.length);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++)for(let oc=0;oc<out.c;oc++){
      let sum=0;
      for(let ic=0;ic<input.c;ic++)for(let ky=0;ky<k;ky++)for(let kx=0;kx<k;kx++){
        sum+=at(input,y*stride+ky-pad,x*stride+kx-pad,ic)*weights[oc][ic][ky*k+kx];
      }
      put(out,y,x,oc,sum);
    }
    return out;
  }
  function batchNormSpatial(input,{gamma=1,beta=0,eps=1e-5}={}){
    if(!(eps>0))throw Error('eps must be positive');
    const out=newTensor(input.h,input.w,input.c),stats=[];
    const count=input.h*input.w;
    for(let ch=0;ch<input.c;ch++){
      let mu=0,variance=0;
      for(let y=0;y<input.h;y++)for(let x=0;x<input.w;x++)mu+=at(input,y,x,ch);
      mu/=count;
      for(let y=0;y<input.h;y++)for(let x=0;x<input.w;x++){const d=at(input,y,x,ch)-mu;variance+=d*d;}
      variance/=count; // training forward uses biased variance estimator, B=1.
      const inv=1/Math.sqrt(variance+eps);
      for(let y=0;y<input.h;y++)for(let x=0;x<input.w;x++){
        put(out,y,x,ch, gamma*(at(input,y,x,ch)-mu)*inv+beta);
      }
      stats.push({mean:mu,var:variance,invstd:inv});
    }
    return {out,stats};
  }
  const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
  const silu=x=>x*sigmoid(x);
  const activate=(t,fn=silu)=>{const o=newTensor(t.h,t.w,t.c);for(let i=0;i<t.values.length;i++)o.values[i]=fn(t.values[i]);return o;};
  const cspWeights=()=>{
    const projectA=[[.50,.25,-.10,.25],[-.10,.55,.30,.25]];
    const projectB=[[.20,.20,.35,.25],[.40,-.25,.30,.10]];
    const inner1=[[.75,.15],[-.12,.67]];
    const inner3=[];
    for(let oc=0;oc<2;oc++){
      const a=[];
      for(let ic=0;ic<2;ic++){
        const scale=oc===ic?.55:.12;
        a.push(BASE.blur.map(v=>v*scale));
      }
      inner3.push(a);
    }
    const fuse=[[.55,.20,.10,.15],[.15,.55,.20,.10],[.10,.20,.55,.15],[.20,.10,.15,.55]];
    return {projectA,projectB,inner1,inner3,fuse};
  };
  function pointwise(t,matrix){return conv2d(t,matrix.map(row=>row.map(v=>[v])),{stride:1,pad:0,k:1});}
  function add(a,b){if(a.values.length!==b.values.length)throw Error('shape mismatch'); const out=newTensor(a.h,a.w,a.c);for(let i=0;i<a.values.length;i++)out.values[i]=a.values[i]+b.values[i];return out;}
  function concatChannels(a,b){if(a.h!==b.h||a.w!==b.w)throw Error('spatial mismatch');const out=newTensor(a.h,a.w,a.c+b.c);for(let y=0;y<a.h;y++)for(let x=0;x<a.w;x++){for(let c=0;c<a.c;c++)put(out,y,x,c,at(a,y,x,c));for(let c=0;c<b.c;c++)put(out,y,x,a.c+c,at(b,y,x,c));}return out;}
  /* Educational CSP-like split/transform/shortcut/concat/fuse.
     It intentionally omits BN inside the CSP branches, unlike the official YOLOX CSPLayer. */
  function cspToy(input,{shortcut=true}={}){
    const w=cspWeights();
    const branch0=activate(pointwise(input,w.projectA));
    const branch1=activate(pointwise(input,w.projectB));
    const inner1=activate(pointwise(branch0,w.inner1));
    const inner2=activate(conv2d(inner1,w.inner3,{k:3,pad:1,stride:1}));
    const main=shortcut?add(branch0,inner2):inner2;
    const joined=concatChannels(main,branch1);
    const fused=activate(pointwise(joined,w.fuse));
    return {branch0,branch1,inner1,inner2,main,joined,fused};
  }
  function convTerms(input,weights,y,x,oc,{stride=2,pad=1}={}){
    let terms=[];
    for(let ic=0;ic<input.c;ic++)for(let ky=0;ky<3;ky++)for(let kx=0;kx<3;kx++){
      const iy=y*stride+ky-pad,ix=x*stride+kx-pad;
      const v=at(input,iy,ix,ic),w=weights[oc][ic][ky*3+kx];
      terms.push({channel:ic,iy,ix,weight:w,input:v,product:v*w});
    }
    return terms;
  }
  function run(input,{preset='mixed',gamma=1,beta=0,shortcut=true}={}){
    const weights=kernelBank(preset);
    const conv=conv2d(input,weights,{stride:2,pad:1,k:3});
    const bn=batchNormSpatial(conv,{gamma,beta});
    const act=activate(bn.out);
    const csp=cspToy(act,{shortcut});
    return {input,weights,conv,bn:bn.out,stats:bn.stats,act,csp};
  }
  const API={newTensor,offset,at,put,kernelBank,conv2d,batchNormSpatial,sigmoid,silu,activate,cspWeights,pointwise,add,concatChannels,cspToy,convTerms,run};
  if(typeof module!=='undefined'&&module.exports)module.exports=API;
  global.YVA03Math=API;
})(typeof window!=='undefined'?window:globalThis);
