/* YOLOX Visual Lab Unit 01: dependency-free, offline educational simulator.
   Every displayed numeric convolution uses real pixel samples from the selected picture.
   torch.nn.Conv2d performs cross-correlation (the same convention used here).
   The three filters and the subsequent layer are illustrative fixed weights—not YOLOX weights. */
(() => {
  'use strict';
  const W = 48, H = 32, PAD = 1;
  const IMAGES = {selfie:'assets/person.jpg', indoor:'assets/room.jpg', outdoor:'assets/park.jpg', zselfie:'../../assets/003Z-selfie.png'};
  const FILTERS = {
    blur: {name:'平均化（ぼかし）',short:'Blur',mat:Array(9).fill(1/9),disp:'1/9'},
    sx: {name:'左右の変化（Sobel X）',short:'Edge X',mat:[-1,0,1,-2,0,2,-1,0,1]},
    sy: {name:'上下の変化（Sobel Y）',short:'Edge Y',mat:[-1,-2,-1,0,0,0,1,2,1]},
    sharp: {name:'輪郭を強調',short:'Sharpen',mat:[0,-1,0,-1,5,-1,0,-1,0]}
  };
  const CH = [0.299,0.587,0.114];
  const lessons = [
    {kicker:'LESSON 01 — THE INPUT',title:'画像は、数字の集まり。',lead:'まず「画像」をRGB値の配列として見てみよう。', insight:'ここでの48×32は教材用の縮小サイズ。実際のYOLOXで画像を必ず48×32にするわけではありません。', stories:[
      '写真をクリックしてください。選んだ位置の赤・緑・青の数値を調べられます。',
      '説明しやすくするため、画像を48×32に縮小します。1マスは1つのサンプリング位置です。',
      '1マスの中にはRGBの3つの値があります。「顔」などの意味はまだ入っていません。',
      'これらの値を計算へ渡して、画像から新しい特徴量を作っていきます。'
    ]},
    {kicker:'LESSON 02 — ONE CONVOLUTION',title:'3×3の窓で、1つの値を作る。',lead:'9個の画素と9個の重みを掛け合わせ、合計する。この計算が畳み込みの基本です。',insight:'PyTorchのConv2dは、厳密にはKernelを反転しない相互相関（cross-correlation）です。この教材も同じ計算を使います。',stories:[
      'オレンジの枠が3×3の参照領域です。選んだ出力位置に対応する9個の入力値を使います。',
      'Kernelは9個の重みです。位置の対応する「画素値 × 重み」を1組ずつ計算します。',
      '9個の積を合計すると1個の出力値になります。負の結果もそのまま残ります。',
      'RGBなら3チャネル×9＝27個の積を足します。出力はそれでも1つの値です。'
    ]},
    {kicker:'LESSON 03 — FEATURE MAP',title:'同じ計算を、画像全体へ。',lead:'Kernelを1マスずつ動かし、出力値を並べたものが特徴マップです。',insight:'出力の1マスは「入力画像をそこだけ独立に切り出したもの」ではありません。隣接する3×3の参照領域は大きく重なります。',stories:[
      'ある位置で1つの出力値ができたら、Kernelを右へ移動します。',
      '次の位置では入力の3×3領域が少し重なります。それでも別の値を計算します。',
      '画像全体へ繰り返すと48×32個の出力値が並びます。これが1枚の特徴マップです。',
      '再生ボタンで全位置を計算する過程を見てください。最後に完成した分布を観察できます。'
    ]},
    {kicker:'LESSON 04 — OUTPUT CHANNELS',title:'Kernelが違えば、見える特徴も違う。',lead:'複数のKernelから、複数の特徴チャネルを作ります。',insight:'3つの出力チャネルは、この教材で選んだ3つの固定Kernelの計算結果です。実際のCNNでは重みを学習で獲得し、チャネル数も設計により変わります。',stories:[
      '同じ入力画像に「ぼかし」「縦方向の境界」「横方向の境界」の3つのKernelを適用します。',
      'それぞれが48×32の異なる特徴マップを作ります。つまり出力は48×32×3です。',
      '各チャネルを選ぶと、画像のどの変化に強く反応するか見比べられます。',
      '次のCNN層は、この3枚を「まとめて」入力として受け取ります。'
    ]},
    {kicker:'LESSON 05 — NEXT LAYER',title:'特徴を重ねて、さらに新しい特徴へ。',lead:'次の層は、前の層の複数チャネルを使って計算します。',insight:'「Stride 2」は計算位置を2マスずつ進めるという意味です。画像を4分割して別々に送ることではありません。',stories:[
      '左は入力、中央は3チャネルの特徴マップ、右はさらに畳み込んだ次の層です。',
      '次の層では各出力位置で、入力の全3チャネルから得た積を合計します。',
      'Strideを2にすると計算位置が間引かれ、48×32から24×16へ空間解像度が下がります。',
      '格子は特徴値を配置する座標です。格子の幅が、参照する画像領域の幅と一致するとは限りません。'
    ]},
    {kicker:'LESSON 06 — WRITE IT YOURSELF',title:'自分の手で、畳み込みを実装しよう。',lead:'次は見た計算を、そのままNumPyで再現します。',insight:'コード演習ではscipyやtorch.nn.Conv2dを使わず、入力・Kernel・padding・stride・チャネル数から出力を計算します。',stories:[
      'まず入力の高さ・幅、Kernelのサイズ、padding、strideから出力サイズを計算します。',
      '次に2重ループで出力位置を走査し、3×3の積と合計を自分で実装します。',
      '3チャネル入力へ拡張し、最後に2つの出力Kernelで異なる特徴マップを作ります。',
      '自動テストとブラウザの選択位置の値を一致させられたら、この章は修了です。'
    ]}
  ];
  const $=s=>document.querySelector(s);
  const state = {step:0, beat:0, image:'zselfie', x:22,y:12, outX:11,outY:6, grid:true,
    filter:'blur',channelMode:'gray',term:0,scanIndex:0,scanPlaying:false, flowStride:2,flowChannel:0,
    guidePlaying:false,guideTimer:null,guideSpeed:1,scanTimer:null,customUrl:null, mathReveal:false};
  const data = {source:null,raw:null,r:null,g:null,b:null,gray:null,stage1:{},stage2:[],stage2W:24,stage2H:16};
  const off=document.createElement('canvas');off.width=W;off.height=H;
  const grayWeights=[.299,.587,.114];
  const fi=v=>Number.isFinite(v)?v:0;
  const fmt=(v,n=3)=>fi(v).toFixed(n);
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const plain=(n)=>String(n).padStart(2,'0');
  const R=v=>Math.round(v*255);
  const shape=(a,b,c)=>`${a} × ${b} × ${c}`;
  const outSize=(n,k,p,s)=>Math.floor((n+2*p-k)/s)+1;
  const at=(a,x,y,w,h)=>x<0||y<0||x>=w||y>=h?0:a[y*w+x];
  const scaleKernel=(kernel,scale)=>kernel.map(v=>v*scale);
  function conv(input,w,h,kernel,stride=1,pad=1){
    const wo=outSize(w,3,pad,stride),ho=outSize(h,3,pad,stride);
    const output=new Float64Array(wo*ho);
    for(let oy=0;oy<ho;oy++)for(let ox=0;ox<wo;ox++){
      let value=0;
      for(let ky=0;ky<3;ky++)for(let kx=0;kx<3;kx++)value+=at(input,ox*stride+kx-pad,oy*stride+ky-pad,w,h)*kernel[ky*3+kx];
      output[oy*wo+ox]=value;
    }
    return {output,w:wo,h:ho};
  }
  function computeLayers(){
    if(!data.gray)return;
    data.stage1={};
    for(const key of Object.keys(FILTERS))data.stage1[key]=conv(data.gray,W,H,FILTERS[key].mat).output;
    computeLayer2();
  }
  function computeLayer2(){
    const s=state.flowStride;
    if(!data.gray)return;
    data.stage2W=outSize(W,3,1,s);data.stage2H=outSize(H,3,1,s);
    const channels=['blur','sx','sy'].map(k=>data.stage1[k]);
    // Each output filter spans all three input feature channels: W[out,in,3,3].
    const bank=[
      [scaleKernel(FILTERS.blur.mat,.70),scaleKernel(FILTERS.blur.mat,.08),scaleKernel(FILTERS.blur.mat,.08)],
      [scaleKernel(FILTERS.sx.mat,.12),scaleKernel(FILTERS.sy.mat,.11),scaleKernel(FILTERS.blur.mat,.22)]
    ];
    data.stage2=bank.map((weights)=>{
      const contributions=channels.map((c,i)=>conv(c,W,H,weights[i],s,1).output);
      return Float64Array.from(contributions[0],(_,j)=>Math.max(0,contributions.reduce((acc,c)=>acc+c[j],0)));
    });
  }
  function loadImage(url,key){
    const im=new Image();
    im.onload=()=>{
      data.source=im;
      const ctx=off.getContext('2d',{willReadFrequently:true});ctx.clearRect(0,0,W,H);ctx.drawImage(im,0,0,W,H);
      const px=ctx.getImageData(0,0,W,H).data;
      data.raw=px;
      data.r=new Float64Array(W*H);data.g=new Float64Array(W*H);data.b=new Float64Array(W*H);data.gray=new Float64Array(W*H);
      for(let j=0;j<W*H;j++){
        data.r[j]=px[j*4]/255;data.g[j]=px[j*4+1]/255;data.b[j]=px[j*4+2]/255;
        data.gray[j]=grayWeights[0]*data.r[j]+grayWeights[1]*data.g[j]+grayWeights[2]*data.b[j];
      }
      computeLayers();state.image=key;
      document.querySelectorAll('[data-image]').forEach(b=>b.classList.toggle('active',b.dataset.image===key));
      render();
    };
    im.onerror=()=>{alert('画像を読み込めませんでした。別の画像を選択してください。');};
    im.src=url;
  }
  function setStep(n,keepGuide=false){
    state.step=clamp(n,0,5);state.beat=0;
    if(!keepGuide)stopGuide();
    stopScan();
    if(state.step===2&&state.scanIndex===0)state.scanIndex=1;
    render();
    if(!keepGuide){document.querySelector('.workspace').scrollIntoView({behavior:'smooth',block:'start'});}
  }
  function setBeat(beat){state.beat=clamp(beat,0,lessons[state.step].stories.length-1);renderStory();}
  function renderStory(){
    const lesson=lessons[state.step];
    $('#storyText').textContent=lesson.stories[state.beat];
    $('#storyDots').innerHTML=lesson.stories.map((_,i)=>`<i class="${i===state.beat?'active':''}"></i>`).join('');
  }
  function render(){
    if(!data.gray)return;
    const l=lessons[state.step];
    $('#lessonKicker').textContent=l.kicker;
    $('#lessonTitle').textContent=l.title;
    $('#lessonLead').textContent=l.lead;
    $('#insightText').textContent=l.insight;
    $('#stepCount').textContent=`${plain(state.step+1)} / 06`;
    $('#progressBar').style.width=`${(state.step+1)/6*100}%`;
    document.querySelectorAll('[data-step]').forEach(el=>{el.classList.toggle('active',Number(el.dataset.step)===state.step);el.setAttribute('aria-current',Number(el.dataset.step)===state.step?'step':'false');});
    $('#backBtn').disabled=state.step===0;
    $('#nextBtn').disabled=state.step===5;
    $('#nextBtn').textContent=state.step===5?'ここまでで修了':'次の章へ →';
    $('#navHint').textContent=['画像の好きな場所をクリック','Kernelを1項ずつ計算','再生してマップを作る','3つの特徴を比べる','Strideを切り替えてみる','付属の演習コードで実装'][state.step];
    $('#cursorLabel').textContent=`選択位置 (${state.x}, ${state.y})`;
    $('#overlayLegend').textContent=state.step===0?'選択中の画素':state.step===4?'前層からの参照範囲':'オレンジ＝参照範囲 / 緑＝選択位置';
    renderStory();renderPhoto();renderCard();renderMap();renderExtra();
  }
  function renderPhoto(){
    const cvs=$('#photoCanvas'),ctx=cvs.getContext('2d'),sw=cvs.width,sh=cvs.height;
    ctx.clearRect(0,0,sw,sh);ctx.drawImage(data.source,0,0,sw,sh);
    const a=sw/W,b=sh/H;
    if(state.grid){ctx.strokeStyle='#ffffff69';ctx.lineWidth=.6;ctx.beginPath();for(let x=0;x<=W;x++){ctx.moveTo(x*a,0);ctx.lineTo(x*a,sh)}for(let y=0;y<=H;y++){ctx.moveTo(0,y*b);ctx.lineTo(sw,y*b)}ctx.stroke();}
    const range=state.step===0?0:state.step===4?2:1;
    const cx=state.step===4?state.outX*state.flowStride:state.x;
    const cy=state.step===4?state.outY*state.flowStride:state.y;
    ctx.fillStyle='#F0A93C31';ctx.fillRect((cx-range)*a,(cy-range)*b,(range*2+1)*a,(range*2+1)*b);
    ctx.strokeStyle='#f3ab3a';ctx.lineWidth=3.5;ctx.strokeRect((cx-range)*a,(cy-range)*b,(range*2+1)*a,(range*2+1)*b);
    ctx.fillStyle='#0fae9f5c';ctx.fillRect(cx*a,cy*b,a,b);
    ctx.strokeStyle='#27e2c7';ctx.lineWidth=2.5;ctx.strokeRect(cx*a,cy*b,a,b);
    if(state.step===1||state.step===2){const index=(state.term%9);const sx=cx+(index%3)-1,sy=cy+Math.floor(index/3)-1;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(sx*a+2,sy*b+2,a-4,b-4);}
  }
  function featuresToRgb(v,maxAbs){
    let t=clamp(Math.abs(v)/Math.max(1e-6,maxAbs),0,1);
    t=Math.pow(t,.62);
    const bg=[232,238,235], target=v<0?[54,108,174]:[218,110,69];
    return [Math.round(bg[0]+(target[0]-bg[0])*t),Math.round(bg[1]+(target[1]-bg[1])*t),Math.round(bg[2]+(target[2]-bg[2])*t)];
  }
  function pixelsForMap(arr,w,h,mask=arr.length){
    const m=Math.max(.01,...Array.from(arr,Math.abs));
    const buf=new Uint8ClampedArray(w*h*4);
    for(let i=0;i<arr.length;i++){
      const rgb=i>=mask?[247,249,247]:featuresToRgb(arr[i],m);
      buf.set([...rgb,255],i*4);
    }
    return new ImageData(buf,w,h);
  }
  function mapCanvasFor(arr,w,h,mask=arr.length,sourceRgb=false){
    const cvs=document.createElement('canvas');cvs.width=w;cvs.height=h;
    const ctx=cvs.getContext('2d');
    if(sourceRgb){ctx.drawImage(off,0,0,w,h);}else{ctx.putImageData(pixelsForMap(arr,w,h,mask),0,0);}
    return cvs;
  }
  function renderMap(){
    const active=state.step>=2&&state.step<=4;
    $('#mapSection').hidden=!active;
    if(!active)return;
    let arr,w=W,h=H,mask=W*H;
    if(state.step===2){arr=data.stage1[state.filter];mask=state.scanIndex;}
    if(state.step===3){arr=data.stage1[state.filter];}
    if(state.step===4){arr=data.stage2[state.flowChannel];w=data.stage2W;h=data.stage2H;}
    $('#mapTitle').textContent=state.step===4?`次の層の出力 · channel ${state.flowChannel}`:`出力特徴マップ · ${FILTERS[state.filter].short}`;
    $('#mapShape').textContent=shape(w,h,1);
    const canvas=$('#mapCanvas'),ctx=canvas.getContext('2d'),width=canvas.width,height=canvas.height;
    ctx.clearRect(0,0,width,height);
    ctx.imageSmoothingEnabled=false;
    const img=mapCanvasFor(arr,w,h,mask);
    ctx.drawImage(img,0,0,width,height);
    const x=state.step===4?state.outX:state.x,y=state.step===4?state.outY:state.y;
    const a=width/w,b=height/h;
    ctx.strokeStyle='#f1aa42';ctx.lineWidth=3;ctx.strokeRect(x*a+1,y*b+1,a-2,b-2);
    if(state.step===2&&mask<0){}
    const value=arr[clamp(y,0,h-1)*w+clamp(x,0,w-1)];
    $('#mapReadout').textContent=`(${x}, ${y}) : ${fmt(value)}`;
  }
  function rgbAt(x,y){const i=y*W+x;return [data.r[i],data.g[i],data.b[i]];}
  function cellVal(x,y,channel){const inp=channel===0?data.r:channel===1?data.g:channel===2?data.b:data.gray;return at(inp,x,y,W,H);}
  function matrixFor(inputType='gray',kernel=false){
    const channel=inputType==='gray'?3:Number(inputType);
    const kernelInfo=FILTERS[state.filter];
    const active=state.term%9;
    let html='<div class="matrix-grid">';
    for(let t=0;t<9;t++){
      const x=state.x+t%3-1,y=state.y+Math.floor(t/3)-1,padded=x<0||y<0||x>=W||y>=H;
      const v=kernel?kernelInfo.mat[t]:cellVal(x,y,channel);
      const display=kernel?(state.filter==='blur'?'1/9':String(v)):(padded?'0':String(Math.round(v*255)));
      html+=`<div class="matrix-cell ${kernel?'kernel-cell':''} ${t===active?'active':''} ${!kernel&&padded?'pad':''}">${display}${!kernel&&padded?'<small>PAD</small>':''}</div>`;
    }
    return html+'</div>';
  }
  function termList(){
    const out=[];
    const mode=state.channelMode==='rgb'?3:1;
    for(let c=0;c<mode;c++)for(let t=0;t<9;t++){
      const x=state.x+t%3-1,y=state.y+Math.floor(t/3)-1;
      const channel=mode===1?3:c;
      const inValue=cellVal(x,y,channel);
      const w=FILTERS[state.filter].mat[t]*(mode===1?1:CH[c]);
      out.push({channel,t,x,y,inValue,w,product:inValue*w});
    }
    return out;
  }
  function convCard(){
    const list=termList(),max=list.length,term=list[Math.min(state.term,max-1)],sum=list.slice(0,state.term).reduce((a,t)=>a+t.product,0);
    const full=list.reduce((a,t)=>a+t.product,0);
    const currentChannel=state.channelMode==='rgb'?term.channel:3;
    return `<span class="eyeline">1つの出力値を計算</span><h3>「入力 × 重み」を順番に足す</h3>
      <p>画像をクリックして計算位置を選択。まずは9項だけを確認し、次にRGBの27項へ進んでください。</p>
      <div class="control-row"><label for="filterSel">Kernel</label><select id="filterSel" class="select" data-role="filter">${Object.keys(FILTERS).map(k=>`<option value="${k}" ${k===state.filter?'selected':''}>${FILTERS[k].name}</option>`).join('')}</select></div>
      <div class="control-row"><button class="tool-btn ${state.channelMode==='gray'?'active':''}" data-action="gray">明るさ1チャネル（9項）</button><button class="tool-btn ${state.channelMode==='rgb'?'active':''}" data-action="rgb">RGB 3チャネル（27項）</button></div>
      ${state.channelMode==='rgb'?`<p class="card-subtitle">今の計算チャネル：${['R（赤）','G（緑）','B（青）'][currentChannel]}　× ${fmt(CH[currentChannel])}（明るさへの寄与）</p>`:''}
      <div class="matrix-wrap"><div class="matrix-col"><div class="mini-heading">入力：${state.channelMode==='rgb'?['R','G','B'][currentChannel]:'明るさ'}（0〜255で表示）</div>${matrixFor(currentChannel===3?'gray':String(currentChannel))}</div><div class="matrix-col"><div class="mini-heading">3×3 Kernel（重み）</div>${matrixFor('gray',true)}</div></div>
      <div class="math-status"><div><small>ここまでの積の合計（${state.term} / ${max}項）</small><div class="math-number">${fmt(sum,5)}</div></div><div><small>全${max}項の最終結果</small><div class="math-number">${fmt(full,5)}</div></div></div>
      <div class="formula">${fmt(term.inValue,4)} × ${fmt(term.w,4)} = ${fmt(term.product,5)}</div>
      <div class="control-row"><button class="tool-btn" data-action="term-prev" ${state.term<=0?'disabled':''}>← 戻す</button><button class="tool-btn emphasis" data-action="term-next" ${state.term>=max?'disabled':''}>次の積を足す →</button><button class="tool-btn" data-action="term-reset">やり直す</button></div>
      <div class="clear-note">数値表示は読みやすく丸めています。内部の計算は0〜1の浮動小数点値を使い、途中で丸めません。端の外側は0埋め（padding=1）です。</div>`;
  }
  function renderCard(){
    let html='';
    switch(state.step){
      case 0:{
        const rgb=rgbAt(state.x,state.y),lum=data.gray[state.y*W+state.x];
        html=`<span class="eyeline">画像 = 高さ × 幅 × チャネル</span><h3>この1マスには、3つの数字がある</h3>
        <p>左の写真をクリックすると、対応するサンプル位置を調べられます。今は画像を<strong>48×32</strong>へ縮小して計算しています。</p>
        <div class="readout-row"><div class="readout-tile"><b style="color:#c94e49">${R(rgb[0])}</b><small>R・赤</small></div><div class="readout-tile"><b style="color:#289d6e">${R(rgb[1])}</b><small>G・緑</small></div><div class="readout-tile"><b style="color:#4575bd">${R(rgb[2])}</b><small>B・青</small></div><div class="readout-tile"><b>${R(lum)}</b><small>明るさ</small></div></div>
        <div class="mini-heading">明るさの計算（この位置）</div><div class="formula">0.299 × ${R(rgb[0])} + 0.587 × ${R(rgb[1])} + 0.114 × ${R(rgb[2])} ≈ ${R(lum)}</div>
        <div class="tag-row"><span class="mini-tag">入力：RGB / 3チャネル</span><span class="mini-tag">学習サンプル：48×32</span><span class="mini-tag">値の範囲：0〜255</span></div>
        <p class="clear-note">画像を縮小すると細かい情報は失われます。ここではCNNの計算を追いやすくするため、意図的に小さくしています。</p>`;break;
      }
      case 1:html=convCard();break;
      case 2:{
        const arr=data.stage1[state.filter],v=arr[state.y*W+state.x];
        html=`<span class="eyeline">同じKernelを繰り返し使う</span><h3>1回の計算 → 特徴マップ1枚</h3>
        <p>各位置で前の章と同じ演算を行います。右下の再生ボタンで、特徴マップが埋まっていく過程を見てください。</p>
        <div class="control-row"><label for="filterSel">Kernel</label><select id="filterSel" class="select" data-role="filter">${Object.keys(FILTERS).map(k=>`<option value="${k}" ${k===state.filter?'selected':''}>${FILTERS[k].name}</option>`).join('')}</select></div>
        <div class="metric-strip"><div class="metric"><b>48×32</b><small>入力の空間サイズ</small></div><div class="metric"><b>3×3</b><small>Kernel</small></div><div class="metric"><b>${fmt(v)}</b><small>選択位置の出力</small></div></div>
        <div class="formula">出力高さ = ⌊(32 + 2×1 − 3) / 1⌋ + 1 = 32<br>出力幅 = ⌊(48 + 2×1 − 3) / 1⌋ + 1 = 48</div>
        <div class="range-row"><label for="scanRange">計算済み</label><input id="scanRange" type="range" min="0" max="${W*H}" step="1" value="${state.scanIndex}" data-role="scan"><output>${state.scanIndex} / ${W*H}</output></div>
        <div class="control-row"><button class="tool-btn emphasis" data-action="scan-play">${state.scanPlaying?'⏸ 一時停止':'▶ 画像全体を計算'}</button><button class="tool-btn" data-action="scan-one">1マス進める</button><button class="tool-btn" data-action="scan-reset">最初から</button></div>
        <div class="clear-note">色は表示しやすいよう各マップの最大絶対値で正規化しています。計算に使う数値は正規化前の値です。青＝負、オレンジ＝正。</div>`;break;
      }
      case 3:{
        html=`<span class="eyeline">特徴量にはいろいろな種類がある</span><h3>同じ画像に異なるKernelを使う</h3>
        <p>1枚の入力から3種類の特徴マップを作ります。下のカードを押して、それぞれがどこに反応するか見比べてください。</p>
        <div class="formula">入力 48 × 32 × 1<br>↓ 3個の異なるKernel<br>出力 48 × 32 × 3</div>
        <div class="mini-heading">選択中の特徴チャネル</div><div class="num-badge">${FILTERS[state.filter].name}</div>
        <div class="clear-note">Kernelごとに得られる特徴の種類が変わります。学習済みCNNではこれらを人が指定せず、誤差逆伝播によって重みを更新します。</div>`;break;
      }
      case 4:{
        const n=state.flowChannel,xx=state.outX,yy=state.outY,v=data.stage2[n][yy*data.stage2W+xx];
        html=`<span class="eyeline">チャネルを合わせて次層へ</span><h3>Strideを変えると何が起きる？</h3>
        <p>前層の3つの特徴マップから、次の層の2つの出力マップを計算します。右の出力1マスは、前層の全3チャネルの3×3領域を参照します。</p>
        <div class="control-row"><label>Stride</label><button class="tool-btn ${state.flowStride===1?'active':''}" data-action="stride1">1（隣へ1マス）</button><button class="tool-btn ${state.flowStride===2?'active':''}" data-action="stride2">2（隣へ2マス）</button></div>
        <div class="control-row"><label>出力チャネル</label><button class="tool-btn ${state.flowChannel===0?'active':''}" data-action="flow0">#0</button><button class="tool-btn ${state.flowChannel===1?'active':''}" data-action="flow1">#1</button></div>
        <div class="metric-strip"><div class="metric"><b>48×32×3</b><small>前層</small></div><div class="metric"><b>${data.stage2W}×${data.stage2H}×2</b><small>次層</small></div><div class="metric"><b>${fmt(v)}</b><small>選択位置の出力</small></div></div>
        <div class="formula">出力幅 = ⌊(48 + 2×1 − 3) / ${state.flowStride}⌋ + 1 = ${data.stage2W}<br>出力高さ = ⌊(32 + 2×1 − 3) / ${state.flowStride}⌋ + 1 = ${data.stage2H}</div>
        <div class="clear-note">右側のマップは3つの前層チャネルから計算し、ReLUを適用した実値です。2つの出力チャネルでは別々の重みを使っています。</div>`;break;
      }
      case 5:{
        html=`<span class="eyeline">次はあなたが実装する</span><h3>Pythonを書かずに通過しない章</h3>
        <p>コード演習の前に、簡単な出力サイズの問題を解いてください。入力48×32、Kernel3×3、padding=1、stride=2のとき、出力は何×何になるでしょう？</p>
        <div class="control-row"><label for="outWGuess">幅</label><input id="outWGuess" class="select" type="number" min="1" max="100" style="width:65px" placeholder="?" value=""><label for="outHGuess">高さ</label><input id="outHGuess" class="select" type="number" min="1" max="100" style="width:65px" placeholder="?" value=""><button class="tool-btn emphasis" data-action="check-size">答えを確認</button></div>
        <div id="guessFeedback" class="callout" role="status">⌊(入力 + 2×padding − Kernel) / stride⌋ + 1 を使います。</div>
        <div class="control-row"><button class="tool-btn emphasis" data-action="download-samples">↓ 48×32 RGBサンプルをCSV保存</button></div><p>このCSVにはブラウザが実際に計算へ使用したRGB値が入っています。Pythonの検証に使えます。</p><div class="mini-heading">NumPyでの中心処理</div><div class="lesson-code"><pre>for oy in range(out_h):
    for ox in range(out_w):
        patch = padded[oy*s:oy*s+3,
                       ox*s:ox*s+3]
        out[oy, ox] = (patch * kernel).sum()</pre></div>
        <p class="clear-note">学生用ファイルには完成コードを入れていません。まず自分で埋め、テストを実行してください。解答コードは別フォルダに分離しています。</p>`;break;
      }
    }
    $('#lessonContent').innerHTML=html;
  }
  function renderExtra(){
    $('#extraSection').hidden=![0,3,4].includes(state.step);
    if(state.step===0){
      $('#extraContent').innerHTML=`<div class="panel-heading"><div><span class="panel-index">B</span><strong>縮小後の48×32サンプルを拡大表示</strong></div><span class="panel-meta">本当に計算へ入る画像です</span></div><div class="map-canvas-outer"><canvas id="sampleCanvas" width="672" height="448" aria-label="48×32に縮小された入力画像"></canvas></div><p class="math-detail">元画像のピクセルをそのまま48×32に切り分けたのではなく、縮小時に補間（リサンプリング）しています。各マスにRGBの値があります。</p>`;
      const c=$('#sampleCanvas'),ctx=c.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(off,0,0,c.width,c.height);
      return;
    }
    if(state.step===3){
      $('#extraContent').innerHTML=`<div class="panel-heading"><div><span class="panel-index">C</span><strong>3種類の特徴マップを見比べる</strong></div><span class="panel-meta">すべて同じ入力から計算</span></div><div class="channel-gallery">${['blur','sx','sy'].map((k,i)=>`<button type="button" data-feature="${k}" class="channel-card ${state.filter===k?'active':''}"><canvas id="feature-${k}" width="192" height="128"></canvas><b>${['ぼかし','左右の境界','上下の境界'][i]}</b><small>48×32×1 · ${FILTERS[k].short}</small></button>`).join('')}</div><p class="channel-expl">同じ画像でもKernelが違えば出力が変わります。これは「3枚に分割した」のではなく「3種類の計算を適用した」結果です。</p>`;
      for(const k of ['blur','sx','sy']){
        const c=$(`#feature-${k}`),ctx=c.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(mapCanvasFor(data.stage1[k],W,H),0,0,c.width,c.height);
      }
      return;
    }
    if(state.step===4){
      $('#extraContent').innerHTML=`<div class="panel-heading"><div><span class="panel-index">C</span><strong>特徴の流れを立体表示</strong></div><span class="panel-meta">各板の画素は実計算した特徴値</span></div><div class="viz3d-wrap"><canvas id="sceneCanvas" width="930" height="325" aria-label="入力画像から3つの特徴チャネルを通り2つの次層の特徴へ伝播する図"></canvas></div><div class="three-labels"><div>入力 RGB<small>48×32×3（色で表示）</small></div><div>1層目<small>48×32×3（特徴）</small></div><div>2層目<small>${data.stage2W}×${data.stage2H}×2（特徴）</small></div></div><p class="math-detail">この3D図はテンソルの見せ方です。RGBを明るさ1チャネルに変換してから3種類の固定Kernelを適用しています（明るさ変換の中間表示は省略）。板の奥行きを物理空間の座標と誤解しないでください。2層目は各出力で全3特徴チャネルの重み付き和を計算しています。</p>`;
      drawScene();return;
    }
  }
  function drawScene(){
    if(state.step!==4)return;
    const cvs=$('#sceneCanvas');if(!cvs)return;
    const c=cvs.getContext('2d'),cw=cvs.width,ch=cvs.height;
    c.clearRect(0,0,cw,ch);
    c.fillStyle='#f3f9f6';c.fillRect(0,0,cw,ch);
    c.fillStyle='#7e9890';c.font='12px system-ui';c.textAlign='center';
    function plane(img,x,y,sc=1,outline='#9eb8ad'){
      const pw=img.width,ph=img.height;
      c.save();
      const a=2.7*sc,b=.63*sc,d=1.75*sc,xx=-1.35*sc;
      c.setTransform(a,b,xx,d,x,y);c.imageSmoothingEnabled=false;c.drawImage(img,0,0);
      c.restore();
      c.strokeStyle=outline;c.lineWidth=1.3;c.beginPath();c.moveTo(x,y);c.lineTo(x+a*pw,y+b*pw);c.lineTo(x+a*pw+xx*ph,y+b*pw+d*ph);c.lineTo(x+xx*ph,y+d*ph);c.closePath();c.stroke();
    }
    const first=mapCanvasFor(null,W,H,0,true);
    plane(first,132,100,1,'#3b837a');
    const stage1=['blur','sx','sy'];
    for(let i=2;i>=0;i--)plane(mapCanvasFor(data.stage1[stage1[i]],W,H),413+i*9,88-i*15,1,'#78a59a');
    for(let i=1;i>=0;i--)plane(mapCanvasFor(data.stage2[i],data.stage2W,data.stage2H),726+i*10,120-i*16,1.55,'#76a49a');
    // Dynamic transfers (visual animation only). All map pixels above are real computed results.
    const t=(Date.now()%3800)/3800;
    function arrow(x1,y1,x2,y2,color){
      c.strokeStyle=color;c.lineWidth=2.5;c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.stroke();
      c.fillStyle=color;c.beginPath();c.moveTo(x2,y2);c.lineTo(x2-10,y2-6);c.lineTo(x2-10,y2+6);c.fill();
      c.fillStyle='#e8a33f';c.beginPath();c.arc(x1+(x2-x1)*t,y1+(y2-y1)*t,6,0,Math.PI*2);c.fill();
    }
    arrow(270,151,370,151,'#62a99a');arrow(565,151,686,151,'#62a99a');
    c.fillStyle='#1d5550';c.font='bold 15px system-ui';c.fillText('RGB入力',145,295);c.fillText('複数のKernel',453,295);c.fillText('次の層',754,295);
  }
  function startScan(){
    if(state.scanPlaying){stopScan();renderCard();return;}
    state.scanPlaying=true;
    if(state.scanIndex>=W*H)state.scanIndex=0;
    state.scanTimer=setInterval(()=>{
      if(state.step!==2||!state.scanPlaying){stopScan();return;}
      state.scanIndex=Math.min(W*H,state.scanIndex+18);
      const completed=Math.max(0,state.scanIndex-1);
      state.x=completed%W;state.y=Math.floor(completed/W);
      renderPhoto();renderMap();renderCard();
      if(state.scanIndex>=W*H){stopScan();renderCard();}
    },50);
    renderCard();
  }
  function stopScan(){if(state.scanTimer)clearInterval(state.scanTimer);state.scanTimer=null;state.scanPlaying=false;}
  function stopGuide(){if(state.guideTimer)clearInterval(state.guideTimer);state.guideTimer=null;state.guidePlaying=false;$('#guideText').textContent='最初から解説を見る';$('#guideIcon').textContent='▶';}
  function guideBeat(){
    if(!state.guidePlaying)return;
    if(state.step===1){state.term=Math.min(termList().length,state.beat*3);renderCard();renderPhoto();}
    if(state.step===2){state.scanIndex=Math.min(W*H,Math.round((state.beat+1)/4*(W*H)));renderMap();}
    if(state.step===4){state.flowStride=state.beat%2===0?1:2;computeLayer2();render();}
    if(state.beat+1<lessons[state.step].stories.length){setBeat(state.beat+1);return;}
    if(state.step<5){setStep(state.step+1,true);return;}
    stopGuide();
  }
  function toggleGuide(){
    if(state.guidePlaying){stopGuide();return;}
    stopScan();state.guidePlaying=true;
    state.step=0;state.beat=0;state.scanIndex=0;state.term=0;render();
    $('#guideText').textContent='解説を一時停止';$('#guideIcon').textContent='⏸';
    document.querySelector('.workspace').scrollIntoView({behavior:'smooth',block:'start'});
    state.guideTimer=setInterval(guideBeat,3300/state.guideSpeed);
  }
  function handleControl(action){
    switch(action){
      case 'gray':state.channelMode='gray';state.term=0;break;
      case 'rgb':state.channelMode='rgb';state.term=0;break;
      case 'term-prev':state.term=Math.max(0,state.term-1);break;
      case 'term-next':state.term=Math.min(termList().length,state.term+1);break;
      case 'term-reset':state.term=0;break;
      case 'scan-play':startScan();return;
      case 'scan-one':stopScan();state.scanIndex=Math.min(W*H,state.scanIndex+1);break;
      case 'scan-reset':stopScan();state.scanIndex=0;break;
      case 'stride1':state.flowStride=1;computeLayer2();state.outX=clamp(state.outX,0,data.stage2W-1);state.outY=clamp(state.outY,0,data.stage2H-1);break;
      case 'stride2':state.flowStride=2;computeLayer2();state.outX=clamp(state.outX,0,data.stage2W-1);state.outY=clamp(state.outY,0,data.stage2H-1);break;
      case 'flow0':state.flowChannel=0;break;
      case 'flow1':state.flowChannel=1;break;
      case 'download-samples':{
        const rows=['x,y,R,G,B'];
        for(let y=0;y<H;y++)for(let x=0;x<W;x++){
          const i=(y*W+x)*4;
          rows.push([x,y,data.raw[i],data.raw[i+1],data.raw[i+2]].join(','));
        }
        const blob=new Blob([rows.join('\n')+'\n'],{type:'text/csv;charset=utf-8'});
        const url=URL.createObjectURL(blob),a=document.createElement('a');
        a.href=url;a.download='samples_rgb_48x32.csv';document.body.appendChild(a);a.click();a.remove();
        setTimeout(()=>URL.revokeObjectURL(url),1000);
        return;
      }
      case 'check-size':{
        const w=Number($('#outWGuess').value),h=Number($('#outHGuess').value);
        const good=w===24&&h===16;
        $('#guessFeedback').textContent=good?'正解！出力は24×16。これをNumPyで実装して確かめましょう。':'もう一度。幅は ⌊(48+2−3)/2⌋+1、高さは ⌊(32+2−3)/2⌋+1 です。';
        return;
      }
    }
    render();
  }
  function initEvents(){
    document.querySelectorAll('[data-step]').forEach(btn=>btn.addEventListener('click',()=>setStep(Number(btn.dataset.step))));
    document.querySelectorAll('[data-image]').forEach(btn=>btn.addEventListener('click',()=>loadImage(IMAGES[btn.dataset.image],btn.dataset.image)));
    $('#customImage').addEventListener('change',ev=>{
      const f=ev.target.files?.[0];if(!f)return;
      if(!['image/png','image/jpeg','image/webp'].includes(f.type)){alert('PNG・JPEG・WebPの画像を選んでください。');return;}
      if(f.size>20*1024*1024){alert('20MB以下の画像を選んでください。');return;}
      if(state.customUrl)URL.revokeObjectURL(state.customUrl);
      state.customUrl=URL.createObjectURL(f);loadImage(state.customUrl,'custom');
    });
    $('#gridToggle').addEventListener('change',ev=>{state.grid=ev.target.checked;renderPhoto();});
    $('#guideBtn').addEventListener('click',toggleGuide);
    $('#storyPrev').addEventListener('click',()=>{
      stopGuide();if(state.beat>0)setBeat(state.beat-1);else if(state.step>0){setStep(state.step-1);setBeat(lessons[state.step].stories.length-1);}
    });
    $('#storyNext').addEventListener('click',()=>{
      stopGuide();if(state.beat+1<lessons[state.step].stories.length)setBeat(state.beat+1);else if(state.step<5)setStep(state.step+1);
    });
    $('#guideSpeed').addEventListener('change',e=>{
      state.guideSpeed=Number(e.target.value);
      if(state.guidePlaying){clearInterval(state.guideTimer);state.guideTimer=setInterval(guideBeat,3300/state.guideSpeed);}
    });
    $('#backBtn').addEventListener('click',()=>setStep(state.step-1));
    $('#nextBtn').addEventListener('click',()=>setStep(state.step+1));
    $('#photoCanvas').addEventListener('click',ev=>{
      const r=ev.currentTarget.getBoundingClientRect(),x=clamp(Math.floor((ev.clientX-r.left)/r.width*W),0,W-1),y=clamp(Math.floor((ev.clientY-r.top)/r.height*H),0,H-1);
      state.x=x;state.y=y;
      if(state.step===4){state.outX=clamp(Math.round(x/state.flowStride),0,data.stage2W-1);state.outY=clamp(Math.round(y/state.flowStride),0,data.stage2H-1);}
      state.term=0;render();
    });
    $('#mapCanvas').addEventListener('click',ev=>{
      const r=ev.currentTarget.getBoundingClientRect(),w=state.step===4?data.stage2W:W,h=state.step===4?data.stage2H:H;
      const x=clamp(Math.floor((ev.clientX-r.left)/r.width*w),0,w-1),y=clamp(Math.floor((ev.clientY-r.top)/r.height*h),0,h-1);
      if(state.step===4){state.outX=x;state.outY=y;state.x=clamp(x*state.flowStride,0,W-1);state.y=clamp(y*state.flowStride,0,H-1);}
      else{state.x=x;state.y=y;if(state.step===2){stopScan();state.scanIndex=y*W+x+1;}}
      render();
    });
    $('#contentCard').addEventListener('click',ev=>{const btn=ev.target.closest('[data-action]');if(btn&&!btn.disabled)handleControl(btn.dataset.action);});
    $('#contentCard').addEventListener('change',ev=>{
      if(ev.target.dataset.role==='filter'){state.filter=ev.target.value;state.term=0;render();}
      if(ev.target.dataset.role==='scan'){stopScan();state.scanIndex=Number(ev.target.value);render();}
    });
    $('#contentCard').addEventListener('input',ev=>{if(ev.target.dataset.role==='scan'){stopScan();state.scanIndex=Number(ev.target.value);state.x=clamp((state.scanIndex-1)%W,0,W-1);state.y=clamp(Math.floor((state.scanIndex-1)/W),0,H-1);renderPhoto();renderMap();ev.target.closest('.range-row').querySelector('output').textContent=`${state.scanIndex} / ${W*H}`;}});
    $('#extraSection').addEventListener('click',ev=>{const b=ev.target.closest('[data-feature]');if(b){state.filter=b.dataset.feature;render();}});
    window.addEventListener('keydown',ev=>{
      if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;
      if(ev.key==='ArrowRight'&&ev.altKey){ev.preventDefault();setStep(state.step+1);}
      if(ev.key==='ArrowLeft'&&ev.altKey){ev.preventDefault();setStep(state.step-1);}
    });
    function animation(){if(state.step===4)drawScene();requestAnimationFrame(animation);}requestAnimationFrame(animation);
  }
  // Expose a read-only educational diagnostics API for offline tests.
  window.YOLOXLab={
    getCurrent:()=>({step:state.step,image:state.image,x:state.x,y:state.y,filter:state.filter,shape1:[W,H,3],shape2:[data.stage2W,data.stage2H,2],scanIndex:state.scanIndex}),
    step:setStep,
    pick:(x,y)=>{state.x=clamp(Math.floor(x),0,W-1);state.y=clamp(Math.floor(y),0,H-1);state.term=0;render();},
    filter:key=>{if(!FILTERS[key])throw new Error('unknown kernel');state.filter=key;render();},
    getSelectedValue:()=>data.stage1[state.filter]?.[state.y*W+state.x],
    getPixel:()=>rgbAt(state.x,state.y),
    getConvolutionTerms:()=>termList().map(x=>({channel:x.channel,position:[x.x,x.y],input:x.inValue,weight:x.w,product:x.product})),
    getReferenceConvolution:()=>data.stage1[state.filter]?.slice(),
    inputGray:()=>data.gray?.slice(),
    scanTo:n=>{state.scanIndex=clamp(n,0,W*H);render();}
  };
  initEvents();loadImage(IMAGES.zselfie,'zselfie');
  const requestedStep = Number(new URLSearchParams(window.location.search).get('step'));
  if (Number.isInteger(requestedStep) && requestedStep>=1 && requestedStep<=6) {
    state.step=requestedStep-1;
    if (state.step===2) state.scanIndex=1;
    window.addEventListener('load',()=>document.querySelector('.workspace').scrollIntoView({block:'start'}), {once:true});
  }
})();
