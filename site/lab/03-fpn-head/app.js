'use strict';
/* Real PyTorch trace viewer. Absolutely no ML inference or user-image upload. */
(() => {
  const $ = id => document.getElementById(id);
  const state = {trace:null,stage:'topdown',level:'p3',node:'dark5',sample:0,x:6,y:6,rotX:18,rotY:-17,drag:null};
  const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
  const sig=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
  const fmt=(n,places=6)=>{
    if (!Number.isFinite(n))return String(n);
    if (Math.abs(n)<1e-16)return '0';
    if (Math.abs(n)<1e-4||Math.abs(n)>=1e6)return n.toExponential(4);
    return n.toFixed(places);
  };
  const stages={
    topdown:{kicker:'TOP-DOWN FPN',title:'深い特徴を拡大し、浅い特徴と合わせる。',desc:'dark5 → 上方に拡大 → dark4・dark3とのConcat → CSP → P3。',
      ids:['dark5','lateral','up0','dark4','cat0','tdp4','reduce','up1','dark3','cat1','p3'],
      symbol:{dark4:'+',dark3:'+'}},
    bottomup:{kicker:'BOTTOM-UP PAN',title:'高解像度の特徴を再び深い方へ伝える。',desc:'P3からstride2で縮小してTop-downの途中結果と再結合。P4とP5を作る。',
      ids:['p3','down0','reduce','cat2','p4','down1','lateral','cat3','p5'],
      symbol:{reduce:'+',lateral:'+'}},
    head:{kicker:'DECOUPLED HEAD',title:'分類のためのConvと、BBoxのためのConvを分ける。',desc:'P3/P4/P5の各HeadでStemの後に分類枝と回帰枝が分岐。Objectnessは回帰枝から予測します。',
      ids:[]}
  };
  const isHead=()=>state.stage==='head';
  const getNode=id=>state.trace?.nodes?.[id];
  const chosen=()=>getNode(state.node);
  function headIds(){const p=state.level;return [p,p+'.stem',p+'.cls_feat',p+'.cls',p+'.reg_feat',p+'.reg',p+'.obj',p+'.raw'];}
  const shapeText=s=>s.join(' × ');

  function addNode(id,n){
    const btn=document.createElement('button');btn.type='button';btn.className='flow-node';btn.dataset.node=id;
    if(/cat\d/.test(id))btn.classList.add('fusion');
    if(['p3','p4','p5'].includes(id)||id.endsWith('.raw'))btn.classList.add('output');
    if(id.endsWith('.cls')||id.endsWith('.cls_feat'))btn.classList.add('head-cls');
    if(id.endsWith('.reg')||id.endsWith('.obj')||id.endsWith('.reg_feat'))btn.classList.add('head-reg');
    if(id===state.node)btn.classList.add('active');
    const k=document.createElement('span');k.className='node-n';k.textContent=n.group.toUpperCase();
    const title=document.createElement('b');title.textContent=n.name;
    const meta=document.createElement('small');meta.textContent=`${n.shape[1]}ch · ${n.shape[2]}×${n.shape[3]}`;
    btn.append(k,title,meta);btn.addEventListener('click',()=>selectNode(id));return btn;
  }
  function addArrow(text){const a=document.createElement('span');a.className='flow-arrow';a.textContent=text;a.setAttribute('aria-hidden','true');return a;}
  function makeRow(ids,signs={}){
    const row=document.createElement('div');row.className='flow-row';row.style.cssText='width:100%;display:flex;align-items:stretch;flex-wrap:wrap;gap:7px';
    ids.forEach((id,i)=>{if(i)row.append(addArrow(signs[id]||'→'));row.append(addNode(id,getNode(id)));});return row;
  }
  function renderGraph(){
    const st=stages[state.stage];$('stageKicker').textContent=st.kicker;$('stageTitle').textContent=st.title;$('stageDescription').textContent=st.desc;
    $('levelSwitch').classList.toggle('hidden',!isHead());
    document.querySelectorAll('[data-level]').forEach(b=>b.classList.toggle('active',b.dataset.level===state.level));
    document.querySelectorAll('[data-stage]').forEach(b=>b.classList.toggle('active',b.dataset.stage===state.stage));
    const root=$('flowChart');root.replaceChildren();
    if(state.stage==='topdown'){
      root.append(makeRow(['dark5','lateral','up0']));
      root.append(makeRow(['dark4','cat0','tdp4','reduce','up1']));
      root.append(makeRow(['dark3','cat1','p3']));
      $('flowHint').textContent='上段のup0と中段のdark4をcat0で結合。次にup1と下段のdark3をcat1で結合（単純な直列処理ではありません）。';
    }else if(state.stage==='bottomup'){
      root.append(makeRow(['p3','down0']));
      root.append(makeRow(['reduce','cat2','p4','down1']));
      root.append(makeRow(['lateral','cat3','p5']));
      $('flowHint').textContent='down0 + reduce → cat2 → P4、down1 + lateral → cat3 → P5。+は要素加算ではなくConcatへ入る第2の入力です。';
    }else{
      const p=state.level;
      root.append(makeRow([p,p+'.stem']));
      root.append(makeRow([p+'.cls_feat',p+'.cls']));
      root.append(makeRow([p+'.reg_feat',p+'.reg',p+'.obj']));
      root.append(makeRow([p+'.raw']));
      $('flowHint').textContent='Stemから「分類側」と「回帰側」に分岐。RegressionからBBoxとObjectnessが別々の1×1 Convで予測されます。最後に[reg4,obj1,cls80]を結合します。';
    }
  }
  function selectNode(id){
    const last=chosen();const next=getNode(id);if(!next)return;
    if(last){state.x=clamp(Math.floor((state.x+.5)*next.shape[3]/last.shape[3]),0,next.shape[3]-1);state.y=clamp(Math.floor((state.y+.5)*next.shape[2]/last.shape[2]),0,next.shape[2]-1);}
    state.node=id;state.sample=next.sampleIndices.length>1?1:0;
    renderGraph();renderNode();
  }
  function chooseStage(stage){
    state.stage=stage;state.level=stage==='head'?state.level:'p3';
    selectNode(stage==='topdown'?'dark5':stage==='bottomup'?'p3':state.level+'.stem');
  }
  function chooseLevel(level){state.level=level;if(isHead())selectNode(level+'.stem');}

  function sampleData(id,channel){const n=getNode(id);return n&&n.samples.find(s=>s.index===channel);}
  function rawAt(id,ch,x,y){const n=getNode(id),s=sampleData(id,ch);if(!n||!s)return null;
    if(x<0||y<0||x>=n.shape[3]||y>=n.shape[2])return null;
    return s.values[y*n.shape[3]+x];
  }
  const signs=(node,v)=>{
    const m=Math.max(1e-18,...node.values.map(x=>Math.abs(x)));
    const q=clamp(Math.abs(v)/m,0,1),k=Math.pow(q,.55);
    return v>=0?[Math.round(239-145*k),Math.round(246-49*k),Math.round(239-95*k)]:[Math.round(239-40*k),Math.round(245-99*k),Math.round(249-15*k)];
  };
  function makeHeat(sample,w,h){
    const off=document.createElement('canvas');off.width=w;off.height=h;
    const ctx=off.getContext('2d');const bytes=new Uint8ClampedArray(w*h*4);
    const maxabs=Math.max(1e-30,...sample.values.map(Math.abs));
    for(let i=0;i<w*h;i++){
      const z=sample.values[i]||0,k=Math.pow(clamp(Math.abs(z)/maxabs,0,1),.64),rgb=z>=0?
       [Math.round(241-151*k),Math.round(248-55*k),Math.round(243-87*k)]:
       [Math.round(241-32*k),Math.round(247-101*k),Math.round(250-13*k)];
      bytes[4*i]=rgb[0];bytes[4*i+1]=rgb[1];bytes[4*i+2]=rgb[2];bytes[4*i+3]=255;
    }
    ctx.putImageData(new ImageData(bytes,w,h),0,0);return off;
  }
  function drawTensorMap(id,sample,overlay=false){
    const cv=$(id),node=chosen();if(!cv||!node||!sample)return;
    const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
    ctx.imageSmoothingEnabled=false;ctx.drawImage(makeHeat(sample,node.shape[3],node.shape[2]),0,0,cv.width,cv.height);
    if(overlay){const cw=cv.width/node.shape[3],ch=cv.height/node.shape[2];
      ctx.fillStyle='#fff';ctx.strokeStyle='#153c34';ctx.lineWidth=2.5;ctx.strokeRect(state.x*cw+1,state.y*ch+1,Math.max(3,cw-2),Math.max(3,ch-2));
      ctx.beginPath();ctx.arc((state.x+.5)*cw,(state.y+.5)*ch,Math.max(2,Math.min(4,cw*.3)),0,2*Math.PI);ctx.fill();}
  }
  function renderStack(node){
    const root=$('tensorStack');root.replaceChildren();let sel=node.samples;
    const ordered=[sel[state.sample],...sel.filter((s,i)=>i!==state.sample)].slice(0,3);
    ordered.slice().reverse().forEach((s,i)=>{
      const depth=ordered.length-1-i;
      const plane=document.createElement('div');plane.className='tensor-plane';plane.style.transform=`translateZ(${depth*35}px)`;
      const can=document.createElement('canvas');can.width=node.shape[3];can.height=node.shape[2];
      can.getContext('2d').drawImage(makeHeat(s,node.shape[3],node.shape[2]),0,0);
      const lab=document.createElement('span');lab.textContent=`C${s.index}`;plane.append(can,lab);root.append(plane);
    });
    root.style.transform=`translate(-50%,-50%) rotateX(${state.rotY}deg) rotateY(${state.rotX}deg)`;
  }
  function calcText(node){
    const input=node.inputs.join(' + ')||'元Backbone演算';const x=state.x,y=state.y,c=node.samples[state.sample].index;
    let expression=`入力：${input}\n出力形状：${shapeText(node.shape)}\n参照コード：${node.modulePath}`;
    if(node.id==='up0'||node.id==='up1'){
      const origin=node.inputs[0],src=rawAt(origin,c,Math.floor(x/2),Math.floor(y/2));
      expression=`nearest-neighbor ×2\ny_out[y=${y}, x=${x}, C${c}] = x_in[floor(${y}/2),floor(${x}/2),C${c}]\n元座標：(${Math.floor(x/2)}, ${Math.floor(y/2)})\n元データ：${src===null?'保存対象外のチャネル':fmt(src)}\n拡大先の値：${fmt(node.samples[state.sample].values[y*node.shape[3]+x])}\n学習パラメータ：0`;
    }else if(/^cat\d$/.test(node.id)){
      const first=getNode(node.inputs[0]),cut=first.shape[1],src=c<cut?node.inputs[0]:node.inputs[1],srcch=c<cut?c:c-cut;
      const srcval=rawAt(src,srcch,x,y),now=node.samples[state.sample].values[y*node.shape[3]+x];
      expression=`torch.cat([${node.inputs[0]}, ${node.inputs[1]}], dim=1)\nC_out=${cut}+${getNode(node.inputs[1]).shape[1]}=${node.shape[1]}\n選択 C${c} は ${src} の C${srcch} に由来\n入力値=${srcval===null?'保存対象外':fmt(srcval)}\n結合後=${fmt(now)}\n※加算ではなくチャネル方向の連結`;
    }else if(node.id.endsWith('.raw')){
      expression=`torch.cat([reg, obj, cls], dim=1)\nC_out=4 + 1 + 80 = 85\nBBox4値は未デコード\nObjectnessとClassはraw logits\n公式推論時にはsigmoidやBBox Decodeを適用`;
    }else if(node.id.endsWith('.cls'))expression=`分類枝の2層のConv → 1×1 Conv → 80クラスのlogit\n特徴: ${node.id.replace('.cls','.cls_feat')}\nクラスの意味はCOCOラベル順の出力番号\n未学習のため確率に意味はありません。`;
    else if(node.id.endsWith('.reg'))expression=`回帰枝の2層のConv → 1×1 Conv → 4 raw値\n(tx,ty,tw,th) の順\nBBoxデコード・損失計算は次のUnit 04`;
    else if(node.id.endsWith('.obj'))expression=`回帰枝の2層のConv → 1×1 Conv → 1 raw値\nobjectness logit。sigmoidで0..1へ変換するが\n未学習データでは実際の検出信頼度ではない。`;
    else if(node.id==='down0'||node.id==='down1')expression=`DWConv 3×3 / stride2\n空間解像度 H,W をそれぞれ半分にする\n入力: ${input}\n出力：${shapeText(node.shape)}\n学習パラメータあり（${node.params.toLocaleString('ja-JP')}）`;
    else if(node.id.includes('feat')||node.id.includes('stem'))expression=`Head ${node.name}\n入力：${input}\nC×H×W のうち、H×Wは変えずチャネルを変換。\n${node.modulePath}\n結果は実PyTorch forward。`;
    else expression=`${node.name}\n入力：${input}\n${node.description}\n${expression}`;
    return expression;
  }
  function renderHeadValues(){
    const root=$('headReadout');root.replaceChildren();
    const level=isHead()?state.level:/^p[345]/.test(state.node)?state.node.slice(0,2):null;
    if(!level){$('headValues').classList.add('hidden');return;}
    $('headValues').classList.remove('hidden');
    const w=getNode(level).shape[3],x=clamp(state.x,0,w-1),y=clamp(state.y,0,w-1);
    const items=[];
    for(let i=0;i<4;i++)items.push([['tx','ty','tw','th'][i],fmt(rawAt(level+'.reg',i,x,y))]);
    const obj=rawAt(level+'.obj',0,x,y);
    items.push(['obj raw',fmt(obj)]);items.push(['sigmoid(obj)',fmt(sig(obj))]);
    for(const ci of [0,1,7,79])items.push([`class[${ci}] raw`,fmt(rawAt(level+'.cls',ci,x,y))]);
    const title=document.createElement('div');title.style.fontSize='11px';title.style.margin='3px 0';title.textContent=`${level.toUpperCase()} / 格子 (${x},${y})`;
    root.append(title);
    items.forEach(([name,v])=>{const row=document.createElement('div');row.className='head-row';const label=document.createElement('span'),value=document.createElement('strong');label.textContent=name;value.textContent=v;row.append(label,value);root.append(row);});
  }
  function renderNode(){
    const n=chosen();if(!n)return;
    state.sample=clamp(state.sample,0,n.samples.length-1);state.x=clamp(state.x,0,n.shape[3]-1);state.y=clamp(state.y,0,n.shape[2]-1);
    const s=n.samples[state.sample];$('nodeTitle').textContent=n.name;$('nodeSubtitle').textContent=n.description;
    $('nodeShape').textContent=shapeText(n.shape);$('stackShape').textContent=`C=${n.shape[1]} · ${n.shape[2]}×${n.shape[3]}`;
    $('nodeParams').textContent=n.params.toLocaleString('ja-JP');$('modulePath').textContent=n.modulePath;
    $('sampleCount').textContent=`(${n.samples.length} / 全${n.shape[1]}ch)`;
    const root=$('channels');root.replaceChildren();n.samples.forEach((v,i)=>{const b=document.createElement('button');b.type='button';b.classList.toggle('active',i===state.sample);b.textContent=`#${v.index}`;b.addEventListener('click',()=>{state.sample=i;renderNode();});root.append(b);});
    drawTensorMap('heatmap',s,true);renderStack(n);
    for(const [name,k,max] of [['coordX','x',n.shape[3]-1],['coordY','y',n.shape[2]-1]]){const input=$(name);input.max=String(max);input.value=String(state[k]);}
    $('pixelCoord').textContent=`(${state.x},${state.y}) / C${s.index}`;$('pixelValue').textContent=fmt(s.values[state.y*n.shape[3]+state.x]);
    $('minMax').textContent=`${fmt(s.min)} / ${fmt(s.max)}`;$('mean').textContent=fmt(s.mean);
    $('mathTitle').textContent=n.name+'：数式とデータのつながり';$('mathBody').textContent=calcText(n);
    renderHeadValues();
  }
  function updateRotation(){state.rotX=+$('rotationX').value;state.rotY=+$('rotationY').value;
    $('rotXLabel').textContent=state.rotX+'°';$('rotYLabel').textContent=state.rotY+'°';
    $('tensorStack').style.transform=`translate(-50%,-50%) rotateX(${state.rotY}deg) rotateY(${state.rotX}deg)`;
  }
  function bind(){
    document.querySelectorAll('[data-stage]').forEach(b=>b.addEventListener('click',()=>chooseStage(b.dataset.stage)));
    document.querySelectorAll('[data-level]').forEach(b=>b.addEventListener('click',()=>chooseLevel(b.dataset.level)));
    $('heatmap').addEventListener('click',ev=>{const n=chosen(),r=ev.currentTarget.getBoundingClientRect();
      state.x=clamp(Math.floor((ev.clientX-r.left)/r.width*n.shape[3]),0,n.shape[3]-1);
      state.y=clamp(Math.floor((ev.clientY-r.top)/r.height*n.shape[2]),0,n.shape[2]-1);renderNode();});
    for(const [name,k] of [['coordX','x'],['coordY','y']])$(name).addEventListener('change',e=>{const n=chosen();const v=Number(e.target.value);if(Number.isInteger(v))state[k]=clamp(v,0,n.shape[k==='x'?3:2]-1);renderNode();});
    $('rotationX').addEventListener('input',updateRotation);$('rotationY').addEventListener('input',updateRotation);
    const scene=$('stackStage');scene.addEventListener('pointerdown',e=>{state.drag=[e.clientX,e.clientY,state.rotX,state.rotY];scene.setPointerCapture(e.pointerId)});
    scene.addEventListener('pointermove',e=>{if(!state.drag)return;const [x,y,rx,ry]=state.drag;
      $('rotationX').value=String(clamp(rx+(e.clientX-x)*.35,-50,50));
      $('rotationY').value=String(clamp(ry-(e.clientY-y)*.35,-45,45));updateRotation();});
    scene.addEventListener('pointerup',()=>state.drag=null);scene.addEventListener('pointercancel',()=>state.drag=null);
    $('classes').addEventListener('change',e=>{const c=+e.target.value,raw=5+c;
      $('classMath').textContent=`${raw}値 / 位置 · 計${(raw*3549).toLocaleString('ja-JP')}値（計算上）`;
    });
  }
  async function start(){
    bind();let d;
    try{
      if(window.__UNIT03C_TEST_TRACE__)d=window.__UNIT03C_TEST_TRACE__;
      else {const res=await fetch('trace.json',{cache:'no-store'});if(!res.ok)throw Error('trace.json status '+res.status);d=await res.json()}
      if(d.schema!=='yolox-academy.neck-head-trace.v1'||d.provenance?.learned!==false||d.params.total!==912159||d.counts.total!==3549||Object.keys(d.nodes).length!==38)throw Error('trace data validation failed');
    }catch(err){$('stageTitle').textContent='トレースデータを読み込めませんでした';$('stageDescription').textContent=String(err);$('flowHint').textContent='HTTPサーバーから開いてください。';return;}
    state.trace=d;state.node='dark5';state.x=6;state.y=6;state.sample=1;
    renderGraph();renderNode();
    window.Academy03C={getState:()=>({stage:state.stage,node:state.node,level:state.level,x:state.x,y:state.y,sample:chosen()?.sampleIndices?.[state.sample]}),
      chooseStage,chooseNode:selectNode,chooseLevel,chosenSample:()=>chosen()?.samples[state.sample],
      getTraceMeta:()=>({schema:d.schema,params:d.params,counts:d.counts})};
  }
  start();
})();
