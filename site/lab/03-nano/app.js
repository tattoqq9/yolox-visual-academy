/* Unit 03-B / YOLOX-Nano official-topology trace explorer.
   All maps are serialized torch tensors produced by tools/unit03b/export_trace.py.
   Default trace is random-initialized (UNTRAINED) with seeded weights; neither heatmaps nor scores should
   be interpreted as trained head features or detections. Offline, no uploads. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id), $all=s=>[...document.querySelectorAll(s)];
  const STAGE_KEYS=['focus','dark2','dark3','dark4','dark5'];
  const BRANCH_KEYS=['branch1','branch1_out','branch2','branch_concat','branch_output'];
  const state={trace:null,tab:'flow',stage:'focus',channel:0,branch:'branch1',branchChannel:0,yaw:28,pitch:27,
       selected:{x:0,y:0},branchSelected:{x:0,y:0},playing:null,dragging:false,pointer:null};
  const canvases=new Map();
  const commas=n=>Number(n).toLocaleString('ja-JP');
  const fixed=(n,d=3)=>Number(n).toFixed(d);
  const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
  const esc=t=>String(t).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
  function moduleForStage(){return state.trace.stages[state.stage]}
  function moduleForBranch(){return state.trace.branches[state.branch]}
  function getStageChannel(){return moduleForStage().channels[state.channel]}
  function getBranchChannel(){return moduleForBranch().channels[state.branchChannel]}

  function color(v,lim){
    // Each channel is symmetrically mapped about zero. Display only; raw values unchanged.
    const ratio=Math.pow(clamp(Math.abs(v)/(lim||1),0,1),.65);
    const base=[242,246,242],warm=[217,123,61],cool=[59,112,177];const dest=v>=0?warm:cool;
    return base.map((x,i)=>Math.round(x+(dest[i]-x)*ratio));
  }
  function makeHeatImage(ch,w,h){
    const key=[ch.index,w,h,ch.min,ch.max,ch.std].join('|');
    if(canvases.has(key))return canvases.get(key);
    const can=document.createElement('canvas');can.width=w;can.height=h;
    const ctx=can.getContext('2d'),image=ctx.createImageData(w,h),lim=Math.max(.00000001,Math.abs(ch.min),Math.abs(ch.max));
    for(let i=0;i<ch.values.length;i++){
      const [r,g,b]=color(ch.values[i],lim),j=i*4;
      image.data[j]=r;image.data[j+1]=g;image.data[j+2]=b;image.data[j+3]=255;
    }
    ctx.putImageData(image,0,0);canvases.set(key,can);
    return can;
  }
  function heat(canvas,node,channel,selection){
    const shape=node.shape,w=shape[3],h=shape[2],ctx=canvas.getContext('2d');
    const source=makeHeatImage(channel,w,h);
    ctx.clearRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=false;
    ctx.drawImage(source,0,0,canvas.width,canvas.height);
    ctx.strokeStyle='rgba(255,255,255,.8)';ctx.lineWidth=1;
    if(selection){const sx=canvas.width/w,sy=canvas.height/h;ctx.strokeStyle='#172c22';ctx.lineWidth=3;
      ctx.strokeRect(selection.x*sx+1,selection.y*sy+1,Math.max(2,sx-1),Math.max(2,sy-1));}
  }
  function clickPosition(event,canvas,node){
    const r=canvas.getBoundingClientRect(),w=node.shape[3],h=node.shape[2];
    return {x:clamp(Math.floor((event.clientX-r.left)/r.width*w),0,w-1),
            y:clamp(Math.floor((event.clientY-r.top)/r.height*h),0,h-1)};
  }
  function renderDetail(){
    const d=moduleForStage(),ch=getStageChannel();
    $('stageTitle').textContent=d.name;
    $('stageExplain').textContent=d.description;
    $('stageShape').textContent=d.shape.join(' × ');
    $('stageStride').textContent=`${d.stride}px相当`;
    $('stageParams').textContent=commas(d.paramCount)+' 個';
    $('stageMinMax').textContent=`${fixed(d.min)} ～ ${fixed(d.max)}`;
    $('shapeTag').textContent=`${d.shape[1]}ch / ${d.shape[2]}×${d.shape[3]}`;
    $('channelPicker').replaceChildren();
    d.channels.forEach((c,i)=>{
      const b=document.createElement('button');b.textContent=`#${c.index}`;b.className=i===state.channel?'active':'';
      b.setAttribute('aria-pressed',String(i===state.channel));b.onclick=()=>{state.channel=i;state.selected={x:0,y:0};renderDetail();render3d()};
      $('channelPicker').appendChild(b);
    });
    const h=d.shape[2],w=d.shape[3];state.selected.x=clamp(state.selected.x,0,w-1);state.selected.y=clamp(state.selected.y,0,h-1);
    heat($('heatMap'),d,ch,state.selected);
    const v=ch.values[state.selected.y*w+state.selected.x];
    $('pixelReadout').textContent=`channel #${ch.index} | (x=${state.selected.x}, y=${state.selected.y}) → ${fixed(v,6)} | channel mean=${fixed(ch.mean,6)}`;
    $('codeSnippet').textContent=`# nano_reference.py のモジュールにforward_hookを登録\nmodule = model.get_submodule('${d.code}')\nhandle = module.register_forward_hook(\n    lambda m, inp, out: print(tuple(out.shape))\n)\nwith torch.inference_mode():\n    model(image_tensor)\nhandle.remove()`;
  }
  function renderStageList(){
    $('stageList').replaceChildren();STAGE_KEYS.forEach((key,i)=>{
      const d=state.trace.stages[key],b=document.createElement('button');
      b.innerHTML=`<strong>${i+1} · ${esc(key==='focus'?'Focus':key)}</strong><small>${d.shape[1]}ch · ${d.shape[2]}×${d.shape[3]}<br>stride ${d.stride}</small>`;
      b.className=state.stage===key?'active':'';
      b.setAttribute('aria-pressed',String(state.stage===key));
      b.onclick=()=>selectStage(key);
      $('stageList').appendChild(b);
    });
  }
  function selectStage(key){if(!state.trace)return;state.stage=key;state.channel=0;state.selected={x:0,y:0};renderStageList();renderDetail();render3d()}
  function project(x,y,z,cx,cy,scale){
    const a=state.yaw*Math.PI/180,p=state.pitch*Math.PI/180;
    const xx=x*Math.cos(a)+z*Math.sin(a),zz=-x*Math.sin(a)+z*Math.cos(a);
    const yy=y*Math.cos(p)-zz*Math.sin(p);
    return {x:cx+xx*scale,y:cy+yy*scale};
  }
  function render3d(){
    if(!state.trace)return;
    const target=$('stack3d'),ctx=target.getContext('2d'),cw=target.width,ch=target.height;
    ctx.clearRect(0,0,cw,ch);
    const grad=ctx.createLinearGradient(0,0,cw,ch);grad.addColorStop(0,'#edf5ee');grad.addColorStop(1,'#dcebe2');ctx.fillStyle=grad;ctx.fillRect(0,0,cw,ch);
    ctx.fillStyle='#587166';ctx.font='17px system-ui';ctx.fillText(`Backbone → ${state.stage.toUpperCase()}`,34,38);
    const d=moduleForStage(),sourceChannels=d.channels.slice(0,3),width=300,height=300,cx=cw*.5,cy=ch*.53;
    // Draw from far to near. Every board is a true selected channel's saved activation.
    for(let i=sourceChannels.length-1;i>=0;i--){
      const chn=sourceChannels[i],layerSep=i*69,z=layerSep-70;
      const p0=project(-width/2,-height/2,z,cx,cy,1),p1=project(width/2,-height/2,z,cx,cy,1),p2=project(width/2,height/2,z,cx,cy,1),p3=project(-width/2,height/2,z,cx,cy,1);
      const img=makeHeatImage(chn,d.shape[3],d.shape[2]);
      ctx.save();ctx.globalAlpha=i===0?1:.9;
      ctx.setTransform((p1.x-p0.x)/img.width,(p1.y-p0.y)/img.width,
              (p3.x-p0.x)/img.height,(p3.y-p0.y)/img.height,p0.x,p0.y);
      ctx.imageSmoothingEnabled=false;ctx.drawImage(img,0,0,img.width,img.height);ctx.restore();
      ctx.beginPath();ctx.moveTo(p0.x,p0.y);ctx.lineTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);ctx.lineTo(p3.x,p3.y);ctx.closePath();
      ctx.strokeStyle=i===0?'#136d54':'#6ca995';ctx.lineWidth=i===0?4:2;ctx.stroke();
      const lab=project(width/2+15,-height/2,z,cx,cy,1);
      ctx.fillStyle='#134f44';ctx.font='bold 15px ui-monospace,Consolas,monospace';ctx.fillText('C'+chn.index,lab.x+8,lab.y+2);
    }
    ctx.fillStyle='#4a7364';ctx.font='14px system-ui';ctx.fillText(`${sourceChannels.length} / ${d.shape[1]} チャネルのみ表示`,34,ch-24);
  }
  function renderDepthwise(){
    const ci=+ $('cin').value,co=+$('cout').value,k=+$('kernelSize').value;
    const standard=ci*co*k*k,depthwise=ci*k*k+ci*co;
    const pct=100*depthwise/standard;
    $('standardCount').textContent=commas(standard);
    $('depthCount').textContent=commas(depthwise);
    $('standardBar').style.width='100%';$('depthBar').style.width=`${pct}%`;
    $('savingText').textContent=`畳み込み重みを ${((1-depthwise/standard)*100).toFixed(1)}% 削減（${(standard/depthwise).toFixed(2)}倍の差）`;
  }
  function renderBranch(){
    const d=moduleForBranch();state.branchChannel=clamp(state.branchChannel,0,d.channels.length-1);
    $('branchTitle').textContent=d.name;$('branchDescription').textContent=d.description;
    $('branchShape').textContent=d.shape.join(' × ');$('branchParams').textContent=commas(d.paramCount)+' 個';
    $('branchPicker').replaceChildren();
    BRANCH_KEYS.forEach(key=>{
      const b=document.createElement('button');b.textContent=state.trace.branches[key].name;b.className=key===state.branch?'active':'';
      b.setAttribute('aria-pressed',String(key===state.branch));b.onclick=()=>{state.branch=key;state.branchChannel=0;state.branchSelected={x:0,y:0};renderBranch()};
      $('branchPicker').appendChild(b);
    });
    $('branchChannels').replaceChildren();
    d.channels.forEach((ch,i)=>{const b=document.createElement('button');b.textContent='#'+ch.index;b.className=i===state.branchChannel?'active':'';
      b.setAttribute('aria-pressed',String(i===state.branchChannel));b.onclick=()=>{state.branchChannel=i;renderBranch()};$('branchChannels').appendChild(b)});
    const ch=getBranchChannel();
    const w=d.shape[3],h=d.shape[2];state.branchSelected.x=clamp(state.branchSelected.x,0,w-1);state.branchSelected.y=clamp(state.branchSelected.y,0,h-1);
    heat($('cspMap'),d,ch,state.branchSelected);
    $('branchPixel').textContent=`C${ch.index} (${state.branchSelected.x},${state.branchSelected.y}) = ${fixed(ch.values[state.branchSelected.y*w+state.branchSelected.x],6)}`;
  }
  function stopPlay(){if(state.playing){clearInterval(state.playing);state.playing=null}$('playStages').textContent='▶ 層を順に見る'}
  function selectTab(key){state.tab=key;stopPlay();$all('[data-tab]').forEach(b=>{const on=b.dataset.tab===key;b.classList.toggle('active',on);b.setAttribute('aria-selected',String(on))});
    $all('.panel').forEach(p=>p.classList.toggle('hidden',p.id!=='panel-'+key));
    if(key==='flow')render3d();else if(key==='csp')renderBranch();else renderDepthwise();
  }
  function renderAll(){
    const t=state.trace;
    const p=t.params;
    $('modelSummary').textContent=`YOLOX-Nano · ${commas(p.total)} パラメータ · 80クラス構成`;
    $('sourceTitle').textContent=t.provenance.type==='random-initialized'?'実際のPyTorch演算・未学習のランダム重み':(t.provenance.trained?'PyTorch演算・学習済みと検証した重み':'PyTorch演算・学習状態未確認のチェックポイント');
    $('sourceDetails').textContent=t.provenance.type==='random-initialized'?
      `重みは固定seed=${t.provenance.seed}でランダム初期化したものです。特徴マップの数字・形状は実演算ですが、頭部らしさ等を学習した特徴や検出結果ではありません。`:
      `チェックポイントのSHA256: ${t.provenance.checkpoint_sha256}。学習状態はエクスポータの明示的な検証フラグで区別します。出所・学習状態・利用権利を確認してください。`; 
    renderStageList();renderDetail();render3d();renderBranch();renderDepthwise();
  }
  async function init(){
    const response=await fetch('trace.json?v=0.3b',{cache:'no-cache'});
    if(!response.ok)throw new Error(`Trace HTTP ${response.status}`);
    const trace=await response.json();
    if(trace.schema!=='yolox-academy.nano-trace.v1')throw new Error('未対応のtrace形式');
    for(const key of STAGE_KEYS){const s=trace.stages?.[key];if(!s || s.channels.length===0 || s.channels[0].values.length!==s.shape[2]*s.shape[3])throw new Error('trace検証エラー: '+key)}
    state.trace=trace;renderAll();
    $all('[data-tab]').forEach(b=>b.addEventListener('click',()=>selectTab(b.dataset.tab)));
    $('cin').addEventListener('change',renderDepthwise);$('cout').addEventListener('change',renderDepthwise);$('kernelSize').addEventListener('change',renderDepthwise);
    $('yaw').oninput=e=>{state.yaw=+e.target.value;$('yawVal').textContent=state.yaw+'°';render3d()};
    $('pitch').oninput=e=>{state.pitch=+e.target.value;$('pitchVal').textContent=state.pitch+'°';render3d()};
    $('resetView').onclick=()=>{state.yaw=28;state.pitch=27;$('yaw').value=28;$('pitch').value=27;$('yawVal').textContent='28°';$('pitchVal').textContent='27°';render3d()};
    $('playStages').onclick=()=>{
      if(state.playing){stopPlay();return}
      $('playStages').textContent='Ⅱ 一時停止';
      state.playing=setInterval(()=>{
        let idx=STAGE_KEYS.indexOf(state.stage);selectStage(STAGE_KEYS[(idx+1)%STAGE_KEYS.length]);
      },1600);
    };
    $('stack3d').onpointerdown=e=>{state.dragging=true;state.pointer={x:e.clientX,y:e.clientY};$('stack3d').setPointerCapture(e.pointerId)};
    $('stack3d').onpointermove=e=>{if(!state.dragging)return;const dx=e.clientX-state.pointer.x,dy=e.clientY-state.pointer.y;
      state.pointer={x:e.clientX,y:e.clientY};state.yaw=clamp(state.yaw+dx*.25,-75,75);state.pitch=clamp(state.pitch+dy*.25,-25,65);
      $('yaw').value=state.yaw;$('pitch').value=state.pitch;$('yawVal').textContent=Math.round(state.yaw)+'°';$('pitchVal').textContent=Math.round(state.pitch)+'°';render3d()};
    $('stack3d').onpointerup=()=>state.dragging=false;$('stack3d').onpointercancel=()=>state.dragging=false;
    $('stack3d').onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){state.yaw=clamp(state.yaw+(e.key==='ArrowRight'?5:-5),-75,75);$('yaw').value=state.yaw;$('yawVal').textContent=state.yaw+'°';render3d();e.preventDefault();}};
    $('heatMap').onclick=e=>{state.selected=clickPosition(e,$('heatMap'),moduleForStage());renderDetail()};
    $('cspMap').onclick=e=>{state.branchSelected=clickPosition(e,$('cspMap'),moduleForBranch());renderBranch()};
    // Frozen snapshot diagnostic for automated browser tests; no arbitrary evaluation.
    window.AcademyUnit03B={getState:()=>({tab:state.tab,stage:state.stage,channel:getStageChannel().index,trained:state.trace.provenance.trained,params:state.trace.params.total,
      shape:moduleForStage().shape.slice(),value:getStageChannel().values[state.selected.y*moduleForStage().shape[3]+state.selected.x]}),
      selectStage,selectTab,featureValue:(stage,ch,x,y)=>{
         const node=state.trace.stages[stage], c=node.channels.find(k=>k.index===ch);
         if(!c||x<0||y<0||x>=node.shape[3]||y>=node.shape[2])return null;
         return c.values[y*node.shape[3]+x];
      }};
  }
  init().catch(error=>{
    $('sourceTitle').textContent='可視化データを読み込めませんでした';$('sourceDetails').textContent=error.message+'。GitHub Pagesの公開状態やローカルHTTPサーバーを確認してください。';
    $('modelSummary').textContent='TRACE NOT LOADED';
  });
})();
