/* YOLOX Visual Academy Unit 06. Synthetic analytical calculations; not empirical head detection. */
(function (root,factory) { 'use strict'; const mod=factory(); if(typeof module==='object'&&module.exports) module.exports=mod; else root.YvaMath06=mod; })(typeof window!=='undefined'?window:globalThis,function () {
'use strict';
const sigmoid=z=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));
const softplus=z=>Math.max(z,0)+Math.log1p(Math.exp(-Math.abs(z)));
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const xyxy=({cx,cy,w,h})=>[cx-w/2,cy-h/2,cx+w/2,cy+h/2];
function iou(a,b){
 const x1=Math.max(a[0],b[0]), y1=Math.max(a[1],b[1]);
 const x2=Math.min(a[2],b[2]), y2=Math.min(a[3],b[3]);
 const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1);
 const A=Math.max(0,a[2]-a[0])*Math.max(0,a[3]-a[1]);
 const B=Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]);
 const u=A+B-inter;return u>0?inter/u:0;
}
function decodeYolox(raw,gx,gy,stride){
 if(!Number.isFinite(stride)||stride<=0||raw.length!==4||raw.some(x=>!Number.isFinite(x))) throw new Error('invalid YOLOX raw/stride');
 return xyxy({cx:(gx+raw[0])*stride,cy:(gy+raw[1])*stride,w:Math.exp(raw[2])*stride,h:Math.exp(raw[3])*stride});
}
function encodeYolox(box,gx,gy,stride){
 const [x1,y1,x2,y2]=box,w=x2-x1,h=y2-y1;
 if(w<=0||h<=0||stride<=0)throw new Error('GT must have positive size');
 return [((x1+x2)/2)/stride-gx,((y1+y2)/2)/stride-gy,Math.log(w/stride),Math.log(h/stride)];
}
function decodeSignedLTRB(raw,refx,refy){
 const [l,t,r,b]=raw;
 return [refx-l,refy-t,refx+r,refy+b];
}
function encodeSignedLTRB(box,refx,refy){return [refx-box[0],refy-box[1],box[2]-refx,box[3]-refy];}
function validBox(b){return b.length===4&&b.every(Number.isFinite)&&b[2]>b[0]&&b[3]>b[1];}
/* Ordered alternative: signed unconstrained center offsets but strictly positive size via softplus. */
function decodeOrdered(raw,refx,refy,epsilon=1e-4){
 const [dx,dy,uw,uh]=raw; const w=softplus(uw)+epsilon,h=softplus(uh)+epsilon;
 return [refx+dx-w/2,refy+dy-h/2,refx+dx+w/2,refy+dy+h/2];
}
function dynamicK(ious){return ious.length?Math.max(1,Math.floor(ious.slice().sort((a,b)=>b-a).slice(0,10).reduce((s,x)=>s+x,0))):0;}
function geometryCenter(gx,gy,stride,gt,radius=1.5){const cx=(gx+.5)*stride,cy=(gy+.5)*stride;return Math.abs(cx-gt.cx)<radius*stride&&Math.abs(cy-gt.cy)<radius*stride;}
function simulate({gt, stride=8, level='low', threshold=.25, imageSize=128}){
 if(!gt||!(gt.w>0&&gt.h>0)||![4,8,16,32].includes(stride)||!['low','high'].includes(level))throw new Error('invalid demo parameters');
 const w=Math.ceil(imageSize/stride), h=Math.ceil(imageSize/stride), rows=[],pool=[];
 const target=xyxy(gt);
 for(let gy=0;gy<h;gy++)for(let gx=0;gx<w;gx++){
  const px=(gx+.5)*stride,py=(gy+.5)*stride;
  const inside=geometryCenter(gx,gy,stride,gt), distance=Math.hypot(px-gt.cx,py-gt.cy)/stride;
  const jitterX=.16*Math.sin(gx*1.9+gy*.7+1)*Math.max(1,gt.w);
  const jitterY=.16*Math.cos(gx*.8-gy*1.6)*Math.max(1,gt.h);
  const pred={cx:gt.cx+jitterX,cy:gt.cy+jitterY,w:gt.w*(1+.15*Math.sin(gx+gy)),h:gt.h*(1+.15*Math.cos(gx-gy))};
  const overlap=iou(xyxy(pred),target);
  // Synthetic score is independent from bbox geometry; it is NOT a learned prediction.
  const base=level==='low'?-4.1:2.4;
  const objLogit=base-.32*distance+.2*Math.sin(gx+gy),clsLogit=base-.15*distance;
  const score=sigmoid(objLogit)*sigmoid(clsLogit);
  const clsCost=-Math.log(Math.max(1e-12,Math.sqrt(score)));
  const cost=clsCost+3*(-Math.log(overlap+1e-8));
  const candidate={gx,gy,x:px,y:py,inside,score,objLogit,clsLogit,pred,box:xyxy(pred),iou:overlap,cost,id:`${gx}:${gy}`};
  rows.push(candidate);if(inside)pool.push(candidate);
 }
 const k=dynamicK(pool.map(x=>x.iou));
 const selected=pool.slice().sort((a,b)=>a.cost-b.cost||a.gy-b.gy||a.gx-b.gx).slice(0,k);
 const passed=pool.filter(x=>x.score>=threshold);
 const matchedPassed=selected.filter(x=>x.score>=threshold);
 const best=pool.reduce((a,b)=>!a||b.iou>a.iou?b:a,null);
 return {rows,pool,selected,passed,matchedPassed,k,geometryCount:pool.length,bestIoU:best?.iou??0,topScore:Math.max(0,...pool.map(x=>x.score)),best,threshold,level,stride,gt};
}
function diagnosis(s){
 if(!s.geometryCount)return '幾何候補が0件です。まずGT中心・stride・候補範囲の条件を調べてください。';
 if(s.bestIoU<.5)return '幾何候補はありますが、BBox品質が低く、IoU ≥ 0.5 の候補がありません。';
 if(!s.passed.length)return 'BBox品質のよい候補がありますが、Objectness×Classification Scoreが閾値未満です。';
 if(!s.matchedPassed.length)return '閾値を超える候補はありますが、正例に選ばれた候補は通過していません。';
 return '幾何候補・BBox品質・Scoreの条件を満たす候補があります。ただし実検出の成功は意味しません。';
}
return {sigmoid,softplus,clamp,xyxy,iou,decodeYolox,encodeYolox,decodeSignedLTRB,encodeSignedLTRB,validBox,decodeOrdered,geometryCenter,dynamicK,simulate,diagnosis};
});
