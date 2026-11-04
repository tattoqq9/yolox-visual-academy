'use strict';
/* Unit 04 is a static client-only learning experience. Actual trace = untrained model. GT/candidate illustrations are explicitly synthetic. */
(()=>{
 const $=id=>document.getElementById(id);
 const M=window.YvaMath04;
 if(!M){throw Error('Unit 04 math.js failed to load');}
 const state={step:'decode',trace:null,img:null,edited:false,level:'p3',x:24,y:14,selectedCandidate:null,gts:[{id:'A',cx:184,cy:116,w:64,h:64,cls:0},{id:'B',cx:232,cy:126,w:66,h:68,cls:1}],matching:null};
 const fmt=(v,d=3)=>Number.isFinite(v)?v.toFixed(d):'—';
 const metric=(name,value)=>`<div class="metric"><small>${name}</small><strong>${value}</strong></div>`;
 const readNumber=(id,defaultValue=0)=>{const v=Number($(id)?.value);return Number.isFinite(v)?v:defaultValue;};
 const cap=(x,a,b)=>M.clamp(x,a,b);
 const drawLine=(ctx,points,color,width=1,dashed=false)=>{ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dashed?[6,5]:[]);ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();ctx.restore()};
 function bg(ctx,w=416,h=416){
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#e3e9e3';ctx.fillRect(0,0,w,h);
  if(state.img&&state.img.complete&&state.img.naturalWidth)ctx.drawImage(state.img,0,0,w,h);
  else{ctx.fillStyle='#607c71';ctx.textAlign='center';ctx.font='16px sans-serif';ctx.fillText('入力写真を読み込み中',w/2,h/2)}
 }
 function rect(ctx,b,color,label,opts={}){
  const [x1,y1,x2,y2]=M.xyxy(b);ctx.save();ctx.lineWidth=opts.width??2.5;ctx.strokeStyle=color;ctx.fillStyle=opts.fill??'transparent';ctx.setLineDash(opts.dash?[6,4]:[]);ctx.strokeRect(x1,y1,Math.max(0,x2-x1),Math.max(0,y2-y1));
  if(label){ctx.setLineDash([]);ctx.font='bold 12px sans-serif';const text=String(label);const t=ctx.measureText(text).width+12;let lx=cap(x1,1,416-t-1),ly=y1>17?y1-20:y1+2;ctx.fillStyle=color;ctx.fillRect(lx,ly,t,18);ctx.fillStyle='#fff';ctx.fillText(text,lx+6,ly+13);}ctx.restore();
 }
 function circle(ctx,x,y,r,color,fill=false){ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);if(fill){ctx.fillStyle=color;ctx.fill()}else{ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();}}
 function showStep(step){
  state.step=step;
  if(window.history&&window.history.replaceState)window.history.replaceState(null,'','#'+step);
  for(const button of document.querySelectorAll('[data-step]')){const active=button.dataset.step===step;button.classList.toggle('active',active);if(active)button.setAttribute('aria-current','step');else button.removeAttribute('aria-current');}
  for(const panel of document.querySelectorAll('[data-panel]'))panel.hidden=panel.dataset.panel!==step;
  render();
 }
 function getActual(){
  const level=state.trace?.levels?.[state.level];if(!level)return null;
  const index=state.y*level.width+state.x;
  const a=level.raw;
  return {raw:[a.tx[index],a.ty[index],a.tw[index],a.th[index]],obj:a.obj[index],cls0:a.cls0[index],stride:level.stride};
 }
 function rawControls(actual){
  state.edited=false;
  ['Tx','Ty','Tw','Th'].forEach((key,i)=>{const el=$('raw'+key);let n=actual.raw[i];if(n<Number(el.min))el.min=String(Math.floor(n-1));if(n>Number(el.max))el.max=String(Math.ceil(n+1));el.value=String(n);});
  $('originFlag').textContent='実PyTorch値';$('originFlag').classList.remove('edited');
 }
 function refreshActual(){
  const actual=getActual();if(!actual)return;
  rawControls(actual);renderDecode();
 }
 function renderDecode(){
  const d=getActual();if(!d)return;
  const vals=['Tx','Ty','Tw','Th'].map(s=>readNumber('raw'+s));
  const b=M.decode(vals,state.x,state.y,d.stride);const gt=state.gts[0];const overlap=M.iou(b,gt);
  const c=$('decodeCanvas'),ctx=c.getContext('2d');bg(ctx);
  rect(ctx,gt,'#d1982f','演習用GT');rect(ctx,b,'#1e75bd',state.edited?'仮想BBox':'実rawからのBBox');
  const [x1,y1,x2,y2]=M.xyxy(b);
  ctx.save();ctx.strokeStyle='#1e75bd';ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.strokeRect(state.x*d.stride,state.y*d.stride,d.stride,d.stride);ctx.restore();
  for(let i=0;i<4;i++)$(''+['txOut','tyOut','twOut','thOut'][i]).textContent=fmt(vals[i],4);
  $('decodedMetrics').innerHTML=metric('BBox中心（cx,cy）',`${fmt(b.cx,1)}, ${fmt(b.cy,1)}`)+metric('幅 × 高さ（px）',`${fmt(b.w,1)} × ${fmt(b.h,1)}`)+metric('予測xyxy',`${fmt(x1,0)},${fmt(y1,0)}…`)+metric('仮想GTとのIoU',fmt(overlap,4));
  $('headLogits').innerHTML=`<strong>この格子の未学習Head出力（80クラス中class0のみ表示）</strong><br>obj_raw = ${fmt(d.obj,5)} → sigmoid = ${fmt(M.sigmoid(d.obj),5)}<br>cls0_raw = ${fmt(d.cls0,5)} → sigmoid = ${fmt(M.sigmoid(d.cls0),5)}<br>obj×cls0 = ${fmt(M.sigmoid(d.obj)*M.sigmoid(d.cls0),7)}（頭部の認識確率ではない）`;
 }
 function renderIoU(){
  const size=readNumber('sizeExample',4),offset=readNumber('pixelShift',1),zoom=size===4?10:size===16?5:1.55;
  const cv=$('iouCanvas'),ctx=cv.getContext('2d');ctx.clearRect(0,0,416,250);ctx.fillStyle='#eff4ef';ctx.fillRect(0,0,416,250);
  const centX=210,centY=125,s=size*zoom;
  rect(ctx,{cx:centX,cy:centY,w:s,h:s},'#d1982f','GT',{});
  rect(ctx,{cx:centX+offset*zoom,cy:centY,w:s,h:s},'#1e75bd','予測',{});
  ctx.fillStyle='#304e40';ctx.textAlign='left';ctx.font='12px sans-serif';ctx.fillText(`表示倍率：${fmt(zoom,1)}倍（面積は実pxで計算）`,12,234);
  const iou=M.iou({cx:0,cy:0,w:size,h:size},{cx:offset,cy:0,w:size,h:size});
  const inter=Math.max(0,size-offset)*size,union=2*size*size-inter;
  $('shiftOut').textContent=`${offset}px`;
  $('iouReadout').innerHTML=`IoU <strong>${fmt(iou,4)}</strong><div>共通部分 ${fmt(inter,1)} px² / 和集合 ${fmt(union,1)} px² · ${size}pxのBBoxを${offset}pxずらした例</div>`;
  renderNMS();
 }
 function demoBoxes(){const S=[
  [0,186,124,78,68,.95,0],[1,191,126,79,68,.89,0],[2,197,123,77,64,.81,0],[3,184,128,72,72,.67,1],
  [4,260,178,47,59,.78,0],[5,263,177,44,60,.64,0],[6,86,80,40,55,.42,1],[7,187,124,82,67,.12,0]
 ];return S.map(([id,cx,cy,w,h,score,cls])=>({id:String(id),box:{cx,cy,w,h},score,cls}));}
 function renderNMS(){
  const candidates=demoBoxes(),threshold=readNumber('nmsThreshold',.5),score=readNumber('nmsScore',.2),aware=$('nmsClassAware').checked;
  $('nmsThresholdLabel').textContent=fmt(threshold,2);$('nmsScoreLabel').textContent=fmt(score,2);
  const result=M.nms(candidates,threshold,score,aware);$('nmsSummary').innerHTML=`<b>${result.keep.length}件が残る</b> · ${result.rejected.length}件はIoUで抑制 · ${result.below.length}件はスコア不足`;
  const ctx=$('nmsCanvas').getContext('2d');bg(ctx);
  const colors=['#1b72b6','#b16e22'];
  result.below.forEach(c=>rect(ctx,c.box,'#9aa5a3',`#${c.id} score不足`,{dash:true,width:1}));
  result.rejected.forEach(c=>rect(ctx,c.box,colors[c.cls],`#${c.id} 抑制`,{dash:true,width:1.5}));
  result.keep.forEach(c=>rect(ctx,c.box,colors[c.cls],`#${c.id} 残る`,{width:2.5}));
  const sorted=candidates.slice().sort((a,b)=>b.score-a.score);
  $('nmsRows').innerHTML=sorted.map(c=>{const reject=result.rejected.find(x=>x.id===c.id),below=result.below.some(x=>x.id===c.id);const verdict=below?'Score不足':reject?`#${reject.why.by}とIoU=${fmt(reject.why.overlap,2)}で抑制`:'残る';return `<tr><td><b>#${c.id}</b> / ${c.cls===0?'A':'B'}</td><td>${fmt(c.score,2)}</td><td class="${below||reject?'fail':'pass'}">${verdict}</td></tr>`}).join('');
 }
 // Synthetic predictions are generated from lesson GTs for didactic clarity, never from real model weights.
 function syntheticCandidates(gts){
  const cand=[];
  for(const [level,stride,size] of [['p3',8,52],['p4',16,26],['p5',32,13]]){
   for(let gy=0;gy<size;gy++)for(let gx=0;gx<size;gx++){
    const cx=(gx+.5)*stride,cy=(gy+.5)*stride;
    if(!gts.some(gt=>M.geometry(cx,cy,gt,stride)))continue;
    let chosen=gts[0],best=Infinity;
    for(const gt of gts){const dist=(cx-gt.cx)**2+(cy-gt.cy)**2;if(dist<best){chosen=gt;best=dist;}}
    const phase=(gx*31+gy*17+stride*3)*.14;
    const box={cx:chosen.cx+(cx-chosen.cx)*.34+Math.sin(phase)*2.2,cy:chosen.cy+(cy-chosen.cy)*.34+Math.cos(phase*1.4)*2.2,w:chosen.w*(.88+.19*Math.sin(phase*1.8)),h:chosen.h*(.90+.17*Math.cos(phase*.8))};
    cand.push({id:`${level}:${gx},${gy}`,level,stride,gx:cx,gy:cy,gridX:gx,gridY:gy,box,objLogit:1+.5*Math.cos(phase),clsLogits:[chosen.cls===0?1.4:-.9,chosen.cls===1?1.4:-.9].map((n,i)=>n+.34*Math.sin(phase+i*2.1))});
   }
  }return cand;
 }
 function computeSimOTA(){const gts=state.gts;state.matching=M.simota(syntheticCandidates(gts),gts);}
 function renderSimOTA(){
  computeSimOTA();const {candidates,rows,choices,owner,conflicts,byGT}=state.matching;
  const focus=Number($('simotaGT').value)||0,gt=state.gts[focus],other=state.gts[1-focus],level=$('simotaLevel').value;
  $('simotaX').value=gt.cx;$('simotaY').value=gt.cy;
  $('simotaXOut').textContent=`${gt.cx}px`;$('simotaYOut').textContent=`${gt.cy}px`;
  $('simotaCandidateCount').textContent=`Union候補 ${candidates.length} · 対象GT: ${gt.id} · 上位候補を表示`;
  $('simotaMetrics').innerHTML=metric(`GT ${gt.id} のDynamic K`,choices[focus].k)+metric('GT A/B 割当数',`${byGT[0].length} / ${byGT[1].length}`)+metric('候補Union',candidates.length)+metric('多重割当の解消',conflicts.length);
  const cv=$('simotaCanvas'),ctx=cv.getContext('2d');bg(ctx);
  const colors=['#2868ad','#bd751e'];state.gts.forEach((box,i)=>rect(ctx,box,colors[i],`GT ${box.id}`));
  for(let ci=0;ci<candidates.length;ci++){
   const c=candidates[ci];if(level!=='all'&&c.level!==level)continue;
   const selected=owner.get(ci),valid=rows[focus][ci].inside;
   const clr=selected!==undefined?colors[selected]:(valid?'#4d8875':'#aaaaaa');
   circle(ctx,c.gx,c.gy,selected!==undefined?3.7:2.1,clr,selected!==undefined);
   if(state.selectedCandidate===c.id){circle(ctx,c.gx,c.gy,7,'#e73268',false);rect(ctx,c.box,'#dc3561','選択候補');}
  }
  const indexes=candidates.map((c,ci)=>({c,ci,r:rows[focus][ci]})).filter(({c})=>level==='all'||c.level===level);
  indexes.sort((a,b)=>a.r.cost-b.r.cost||a.ci-b.ci);
  const selected=id=>id===state.selectedCandidate;
  const finalSet=new Set(byGT[focus]);
  const displayed=indexes.slice(0,15),extra=indexes.filter(it=>finalSet.has(it.c.id)&&!displayed.includes(it)).slice(0,5);
  $('simotaRows').innerHTML=[...displayed,...extra].map(({c,ci,r})=>{
   const winner=owner.get(ci);const verdict=winner!==undefined?`GT ${state.gts[winner].id}`:'非正例';
   const penalty=r.inside?'':` + 1e6 penalty`;
   return `<tr data-candidate="${c.id}" class="${selected(c.id)?'selected':''}"><td>${c.level.toUpperCase()} (${c.gridX},${c.gridY})${r.inside?'':' <small>中心範囲外</small>'}</td><td>${fmt(r.iou,3)}</td><td>${fmt(r.clsCost,2)}</td><td>${fmt(r.iouCost*3,2)}${penalty}</td><td class="${winner!==undefined?'pass':'fail'}">${verdict}</td></tr>`;
  }).join('');
  for(const tr of $('simotaRows').querySelectorAll('tr[data-candidate]'))tr.addEventListener('click',()=>{state.selectedCandidate=tr.dataset.candidate;renderSimOTA()});
  const item=state.selectedCandidate?indexes.find(x=>x.c.id===state.selectedCandidate):indexes[0];
  if(item){const {c,r,ci}=item;
   $('simotaExplanation').innerHTML=`<b>候補 ${c.id}</b>：GT ${gt.id} とのIoU=${fmt(r.iou,4)}。classification cost=${fmt(r.clsCost,4)}、3×IoU cost=${fmt(3*r.iouCost,4)}、${r.inside?'幾何条件を満たす':'GT中心の範囲外（+1,000,000）'}。GT ${gt.id} のdynamic K=${choices[focus].k}。最終割当=${owner.has(ci)?'GT '+state.gts[owner.get(ci)].id:'なし'}。`;
  }
 }
 function drawLossCurve(){
  const cv=$('lossCanvas'),ctx=cv.getContext('2d'),size=readNumber('lossBoxSize',40),dx=readNumber('lossShift',6);ctx.clearRect(0,0,520,320);ctx.fillStyle='#f2f8f1';ctx.fillRect(0,0,520,320);
  const left=50,right=505,top=22,bottom=274,minX=-20,maxX=20,maxY=5;
  ctx.font='11px sans-serif';ctx.textAlign='right';ctx.strokeStyle='#dce9df';ctx.fillStyle='#51695c';
  for(let j=0;j<=5;j++){const y=bottom-j/5*(bottom-top);drawLine(ctx,[[left,y],[right,y]],'#cfdfd2');ctx.fillText(String(j),left-8,y+4);}
  for(let x=-20;x<=20;x+=10){const px=left+(x-minX)/(maxX-minX)*(right-left);drawLine(ctx,[[px,top],[px,bottom]],'#e5ede5');ctx.textAlign='center';ctx.fillText(String(x),px,bottom+18)}
  const loss=dist=>5*(1-M.iou({cx:0,cy:0,w:size,h:size},{cx:dist,cy:0,w:size,h:size})**2);
  ctx.save();ctx.beginPath();for(let i=0;i<=400;i++){const x=minX+(maxX-minX)*i/400,y=loss(x),px=left+i/400*(right-left),py=bottom-(y/maxY)*(bottom-top);i?ctx.lineTo(px,py):ctx.moveTo(px,py)}ctx.strokeStyle='#11765d';ctx.lineWidth=3;ctx.stroke();ctx.restore();
  const px=left+(dx-minX)/(maxX-minX)*(right-left),py=bottom-loss(dx)/maxY*(bottom-top);circle(ctx,px,py,5,'#be762d',true);
  ctx.textAlign='left';ctx.fillStyle='#365545';ctx.fillText('Y: 5 × (1 − IoU²)',55,14);ctx.fillText('X: prediction center shift (px)',190,308);
 }
 function renderLoss(){
  const dx=readNumber('lossShift',6),size=readNumber('lossBoxSize',40),obj=readNumber('lossObj',-1.5),cls=readNumber('lossCls',-1.2);
  $('lossShiftOut').textContent=`${fmt(dx,2)}px`;$('lossObjOut').textContent=fmt(obj,1);$('lossClsOut').textContent=fmt(cls,1);
  const gt={cx:0,cy:0,w:size,h:size},pred={cx:dx,cy:0,w:size,h:size};
  const l=M.lossTerms(pred,gt,{obj,cls,negObj:-3});
  const dBox=M.finiteDiff(x=>5*(1-M.iou({cx:x,cy:0,w:size,h:size},gt)**2),dx,0.001);
  $('lossMetrics').innerHTML=metric('IoU',fmt(l.iou,4))+metric('5 × IoU Loss',fmt(l.weightedIoU,4))+metric('Obj BCE（正＋負）',fmt(l.posObj+l.negObj,4))+metric('Cls BCE（IoU target）',fmt(l.cls,4))+metric('Loss合計（num_fg=1）',fmt(l.total,4))+metric('分類品質ターゲット',fmt(l.iou,4));
  $('lossGradients').innerHTML=`<div><span>d L_obj / d obj_logit（正例）</span><code>${fmt(M.sigmoid(obj)-1,5)}</code></div><div><span>d L_cls / d cls_logit（正例）</span><code>${fmt(M.sigmoid(cls)-l.iou,5)}</code></div><div><span>d (5 × L_iou) / d 水平ズレ（数値微分）</span><code>${fmt(dBox,5)} / px</code></div><div><span>負例obj logit（固定値）</span><code>-3.0 / target 0</code></div>`;
  drawLossCurve();
 }
 function render(){if(state.step==='decode'){renderDecode()}else if(state.step==='nms'){renderIoU()}else if(state.step==='simota'){renderSimOTA()}else if(state.step==='loss'){renderLoss()}}
 function init(){
  document.querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>showStep(b.dataset.step)));
  $('decodeLevel').addEventListener('change',()=>{state.level=$('decodeLevel').value;const level=state.trace?.levels?.[state.level];if(!level)return;state.x=cap(state.x,0,level.width-1);state.y=cap(state.y,0,level.height-1);$('decodeX').max=String(level.width-1);$('decodeY').max=String(level.height-1);$('gridTotal').textContent=`${level.width}×${level.height}`;$('decodeX').value=String(state.x);$('decodeY').value=String(state.y);refreshActual()});
  for(const [id,key] of [['decodeX','x'],['decodeY','y']])$(id).addEventListener('change',()=>{const level=state.trace?.levels?.[state.level];if(!level)return;state[key]=cap(Math.round(readNumber(id)),0,level.width-1);$(id).value=String(state[key]);refreshActual()});
  for(const key of ['Tx','Ty','Tw','Th'])$('raw'+key).addEventListener('input',()=>{state.edited=true;$('originFlag').textContent='仮想編集（実forwardではない）';$('originFlag').classList.add('edited');renderDecode()});
  $('restoreRaw').addEventListener('click',refreshActual);
  for(const id of ['sizeExample','pixelShift','nmsThreshold','nmsScore','nmsClassAware'])$(id).addEventListener('input',renderIoU);
  $('simotaGT').addEventListener('change',renderSimOTA);
  $('simotaLevel').addEventListener('change',renderSimOTA);
  for(const [id,key] of [['simotaX','cx'],['simotaY','cy']])$(id).addEventListener('input',()=>{const gt=state.gts[Number($('simotaGT').value)];gt[key]=Number($(id).value);state.selectedCandidate=null;renderSimOTA()});
  for(const id of ['lossBoxSize','lossShift','lossObj','lossCls'])$(id).addEventListener('input',renderLoss);
  const img=new Image();img.onload=()=>{state.img=img;render()};img.onerror=()=>{console.error('Could not load input photo');render()};img.src='../03-nano/input416.jpg';
  fetch('trace04.json',{cache:'no-cache'}).then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json()}).then(data=>{
   if(data.schema!=='yva.unit04.untrained-head-raw.v1'||data.provenance.learned!==false)throw Error('unexpected trace provenance');
   state.trace=data;state.level='p3';$('decodeX').max='51';$('decodeY').max='51';refreshActual();
  }).catch(err=>{$('headLogits').textContent=`trace04.jsonの読込エラー: ${String(err)}。python3 -m http.serverでプレビューしてください。`;console.error(err)});
  renderIoU();renderLoss();
  const requested=window.location.hash.slice(1);if(['decode','nms','simota','loss'].includes(requested))showStep(requested);
 }
 init();
})();
