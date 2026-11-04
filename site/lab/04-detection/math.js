/* Unit 04 exact primitives. Independent educational rewrite, not YOLOX source code. */
(function(root,factory){'use strict';const x=factory();if(typeof module==='object'&&module.exports)module.exports=x;else root.YvaMath04=x;})(typeof window!=='undefined'?window:globalThis,function(){
'use strict';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
const softplus=x=>Math.max(x,0)+Math.log1p(Math.exp(-Math.abs(x)));
const bce=(z,t)=>softplus(z)-t*z;
const xyxy=b=>[b.cx-b.w/2,b.cy-b.h/2,b.cx+b.w/2,b.cy+b.h/2];
const decode=(raw,gx,gy,stride)=>({cx:(Number(gx)+raw[0])*stride,cy:(Number(gy)+raw[1])*stride,w:Math.exp(raw[2])*stride,h:Math.exp(raw[3])*stride});
function iou(a,b){
 const A=Array.isArray(a)?a:xyxy(a),B=Array.isArray(b)?b:xyxy(b);
 const iw=Math.max(0,Math.min(A[2],B[2])-Math.max(A[0],B[0]));
 const ih=Math.max(0,Math.min(A[3],B[3])-Math.max(A[1],B[1]));
 const areaA=Math.max(0,A[2]-A[0])*Math.max(0,A[3]-A[1]);
 const areaB=Math.max(0,B[2]-B[0])*Math.max(0,B[3]-B[1]);
 const inter=iw*ih,union=areaA+areaB-inter;
 return union>0?inter/union:0;
}
function nms(candidates,threshold,scoreMin=0,aware=true){
 const sorted=candidates.filter(c=>c.score>=scoreMin).sort((a,b)=>b.score-a.score||String(a.id).localeCompare(String(b.id)));
 const keep=[],rejected=[];
 for(const c of sorted){let suppress=null;
  for(const k of keep){const overlap=iou(c.box,k.box);if((!aware||c.cls===k.cls)&&overlap>threshold){suppress={by:k.id,overlap};break;}}
  if(suppress)rejected.push({...c,why:suppress});else keep.push(c);
 }
 return {keep,rejected,below:candidates.filter(c=>c.score<scoreMin)};
}
const geometry=(cx,cy,gt,stride,radius=1.5)=>Math.abs(cx-gt.cx)<radius*stride&&Math.abs(cy-gt.cy)<radius*stride;
function classifyCost(objLogit,clsLogits,gtClass){const ps=clsLogits.map(z=>Math.sqrt(sigmoid(z)*sigmoid(objLogit)));return ps.reduce((s,p,i)=>s-(i===gtClass?Math.log(Math.max(1e-12,p)):Math.log(Math.max(1e-12,1-p))),0);}
function simota(candidates,gts,params={}){
 const radius=params.radius??1.5;
 const cand=candidates.filter(c=>gts.some(gt=>geometry(c.gx,c.gy,gt,c.stride,radius)));
 const rows=gts.map((gt,gi)=>cand.map((c,ci)=>{
  const inside=geometry(c.gx,c.gy,gt,c.stride,radius);
  const overlap=iou(c.box,gt);
  const clsCost=classifyCost(c.objLogit,c.clsLogits,gt.cls);
  const iouCost=-Math.log(overlap+1e-8);
  return {gt:gi,cand:ci,id:c.id,inside,iou:overlap,clsCost,iouCost,cost:clsCost+3*iouCost+(inside?0:1e6)};
 }));
 const choices=rows.map(row=>{
  if(!row.length)return {k:0,selected:[],topIous:[]};
  const topIous=row.map(r=>r.iou).sort((a,b)=>b-a).slice(0,10);
  const k=Math.max(1,Math.floor(topIous.reduce((s,x)=>s+x,0)));
  const selected=row.slice().sort((a,b)=>a.cost-b.cost||a.cand-b.cand).slice(0,k).map(r=>r.cand);
  return {k,selected,topIous};
 });
 const owner=new Map(),conflicts=[];
 for(let gi=0;gi<choices.length;gi++)for(const ci of choices[gi].selected){
  if(!owner.has(ci))owner.set(ci,gi);
  else{const prev=owner.get(ci);const better=rows[prev][ci].cost<=rows[gi][ci].cost?prev:gi;
   owner.set(ci,better);conflicts.push({candidate:ci,previous:prev,new:gi,winner:better});}
 }
 const byGT=gts.map((gt,gi)=>[...owner].filter(([ci,assigned])=>assigned===gi).map(([ci])=>cand[ci].id));
 return {candidates:cand,rows,choices,owner,conflicts,byGT};
}
function lossTerms(pred,gt,params){
 const q=iou(pred,gt),liou=1-q*q;
 return {iou:q,liou,weightedIoU:5*liou,
  posObj:bce(params.obj,1),negObj:bce(params.negObj??-2,0),cls:bce(params.cls,q),
  total:5*liou+bce(params.obj,1)+bce(params.negObj??-2,0)+bce(params.cls,q)};
}
const finiteDiff=(fn,x,h=1e-3)=>(fn(x+h)-fn(x-h))/(2*h);
return {clamp,sigmoid,softplus,bce,xyxy,decode,iou,nms,geometry,classifyCost,simota,lossTerms,finiteDiff};
});
