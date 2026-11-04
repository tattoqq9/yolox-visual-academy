'use strict';
/* All candidate data are synthesized client-side. No model inference, user uploads, or network posting. */
(()=>{
const M=window.YvaMath06,$=id=>document.getElementById(id);
if(!M)throw Error('Unit06 math.js unavailable');
const state={gt:{cx:48,cy:64,w:4,h:4},stride:8,threshold:.25,level:'low',reveal:false};
const f=(v,d=3)=>Number.isFinite(v)?v.toFixed(d):'—';
const fmtBox=b=>b.map(x=>f(x,2)).join(', ');
let current=null;
const canvas=$('gridCanvas'),ctx=canvas.getContext('2d'),W=canvas.width,s=W/128;
let img=null;const bg=new Image();bg.onload=()=>{img=bg;render()};bg.src='synthetic-tiny.png';
function draw(data){
 ctx.clearRect(0,0,W,W);ctx.fillStyle='#e5edeb';ctx.fillRect(0,0,W,W);if(img)ctx.drawImage(img,0,0,W,W);
 const space=data.stride*s;ctx.strokeStyle='rgba(61,103,115,.24)';ctx.lineWidth=1;
 for(let x=0;x<=W;x+=space){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,W);ctx.stroke()}
 for(let y=0;y<=W;y+=space){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
 const selected=new Set(data.selected.map(c=>c.id));
 for(const c of data.rows){
  const isSelected=selected.has(c.id),passed=c.score>=data.threshold;
  ctx.beginPath();ctx.arc(c.x*s,c.y*s,c.inside?(isSelected?5.3:3.7):1.7,0,Math.PI*2);
  ctx.fillStyle=!c.inside?'rgba(82,98,94,.45)':isSelected?(passed?'#198752':'#b46a4b'):'#2c83bf';ctx.fill();
  if(isSelected){ctx.strokeStyle='#fff';ctx.lineWidth=1.6;ctx.stroke()}
 }
 const b=M.xyxy(state.gt);ctx.strokeStyle='#edaa19';ctx.lineWidth=3;ctx.setLineDash([]);ctx.strokeRect(b[0]*s,b[1]*s,state.gt.w*s,state.gt.h*s);
 const d=data.selected[0];if(d){const bb=d.box;ctx.strokeStyle='rgba(12,112,75,.83)';ctx.lineWidth=2;ctx.setLineDash([8,6]);ctx.strokeRect(bb[0]*s,bb[1]*s,(bb[2]-bb[0])*s,(bb[3]-bb[1])*s);ctx.setLineDash([])}
 // Zoom inset for the tiny object; to-scale coordinate system enlarged 8x.
 const cx=state.gt.cx,cy=state.gt.cy;const tile=20,source={x:Math.max(0,Math.min(128-tile,cx-tile/2)),y:Math.max(0,Math.min(128-tile,cy-tile/2))};
 const x0=W-164,y0=15,iw=145,scale=iw/tile;
 ctx.fillStyle='rgba(255,255,255,.97)';ctx.fillRect(x0-7,y0-7,iw+14,iw+37);
 ctx.strokeStyle='#b5c9bf';ctx.lineWidth=1.6;ctx.strokeRect(x0-7,y0-7,iw+14,iw+37);
 if(img)ctx.drawImage(img,source.x,source.y,tile,tile,x0,y0,iw,iw);
 const zx=x0+(b[0]-source.x)*scale,zy=y0+(b[1]-source.y)*scale;
 ctx.strokeStyle='#b36c15';ctx.lineWidth=2;ctx.strokeRect(zx,zy,state.gt.w*scale,state.gt.h*scale);
 ctx.fillStyle='#274d40';ctx.font='bold 15px system-ui';ctx.fillText('拡大：20×20px',x0,y0+iw+21);
}
function renderTable(data){
 const rows=data.pool.slice().sort((a,b)=>a.cost-b.cost||a.id.localeCompare(b.id)).slice(0,8),sel=new Set(data.selected.map(x=>x.id));
 $('candidateRows').replaceChildren(...rows.map(c=>{const tr=document.createElement('tr');
  const vals=[`(${c.gx}, ${c.gy})`,f(c.iou),f(M.sigmoid(c.objLogit),5),f(M.sigmoid(c.clsLogit),5),f(c.score,5),f(c.cost,2),sel.has(c.id)?'✓ 正例':'候補'];
  for(const [i,v] of vals.entries()){const cell=document.createElement('td');cell.textContent=v;if(i===6&&sel.has(c.id))cell.className='flag';tr.append(cell)}return tr}));
 $('costNote').textContent=`幾何候補 ${data.pool.length} 個、Top10 IoU和によるDynamic K=${data.k}、表示中 ${rows.length} 候補。Cost = −log(√(p_obj·p_cls)) + 3·(−log(IoU+1e−8))。実YOLOXの全割当処理ではありません。`;
}
function comparison(data){if(!state.reveal)return;
 const gt=M.xyxy(state.gt);const px=(Math.floor(state.gt.cx/state.stride)+.5)*state.stride,py=(Math.floor(state.gt.cy/state.stride)+.5)*state.stride;
 const gx=Math.floor(state.gt.cx/state.stride),gy=Math.floor(state.gt.cy/state.stride);
 const td=Number($('twDelta').value),rd=Number($('rightDelta').value);
 const yr=M.encodeYolox(gt,gx,gy,state.stride);yr[2]+=td;
 const sr=M.encodeSignedLTRB(gt,px,py);sr[2]+=rd;
 const y=M.decodeYolox(yr,gx,gy,state.stride),s=M.decodeSignedLTRB(sr,px,py);
 const valid=M.validBox(s),yIoU=M.iou(y,gt),sIoU=valid?M.iou(s,gt):0;
 $('twOut').textContent=f(td,2);$('rOut').textContent=f(rd,2);
 $('yoloIoU').textContent=`IoU ${f(yIoU)}`;$('signedIoU').textContent=valid?`IoU ${f(sIoU)}`:'INVALID';
 $('signedIoU').style.color=valid?'#13764e':'#a4463a';
 $('yoloBox').textContent=`Box: ${fmtBox(y)}`;
 $('signedBox').textContent=`Box: ${fmtBox(s)}${valid?'':'（x1≥x2 または y1≥y2）'}`;
 $('orderedWidth').textContent='w > 0';
 $('derivatives').textContent=`局所微分（同じGTを再符号化した点）：YOLOXの∂w/∂tw=${f(Math.exp(yr[2])*state.stride,3)} px、Signed LTRBの∂x2/∂r=1。参照点(${f(px,0)},${f(py,0)}) / 予測式とLossの組合せを必ず区別してください。`;
}
function render(){
 const d=M.simulate({gt:state.gt,stride:state.stride,level:state.level,threshold:state.threshold});current=d;
 $('geometryCount').textContent=String(d.geometryCount);
 $('bestIoU').textContent=f(d.bestIoU);
 $('selectedCount').textContent=`${d.selected.length} / ${d.k}`;
 $('scorePass').textContent=`${d.matchedPassed.length} / ${d.selected.length}`;
 $('diagnosis').textContent=M.diagnosis(d);
 $('boxSizeLabel').textContent=`${state.gt.w} px`;
 $('centerXLabel').textContent=`${state.gt.cx} px`;
 $('centerYLabel').textContent=`${state.gt.cy} px`;
 $('thresholdLabel').textContent=f(state.threshold,2);
 draw(d);renderTable(d);comparison(d);
}
function refresh(){const v=Number($('boxSize').value);state.gt.w=v;state.gt.h=v;state.gt.cx=Number($('centerX').value);state.gt.cy=Number($('centerY').value);state.stride=Number($('stride').value);state.level=$('scoreLevel').value;state.threshold=Number($('threshold').value);render()}
for(const id of ['boxSize','centerX','centerY','stride','scoreLevel','threshold'])$(id).addEventListener('input',refresh);
for(const id of ['twDelta','rightDelta'])$(id).addEventListener('input',()=>comparison(current));
$('reveal').addEventListener('click',()=>{const v=$('idea').value.trim();if(v.length<15){$('ideaHelp').textContent='15文字以上で、座標・距離や順序制約の仮説を書いてください。';$('idea').focus();return;}state.reveal=true;$('comparison').hidden=false;$('ideaHelp').textContent='仮説を記入しました。比較式を表示しています。入力はどこにも送信されません。';comparison(current)});
canvas.addEventListener('pointerdown',e=>{const b=canvas.getBoundingClientRect();const x=(e.clientX-b.left)/b.width*128,y=(e.clientY-b.top)/b.height*128;if(x>=0&&x<=128&&y>=0&&y<=128){$('centerX').value=String(Math.round(M.clamp(x,16,112)));$('centerY').value=String(Math.round(M.clamp(y,16,112)));refresh()}});
render();
})();
