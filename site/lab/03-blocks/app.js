/* Unit03-A: browser-only educational visualization.
   UI, calculations and test API use the same pure math in math.js. */
(() => {
  'use strict';
  const M=window.YVA03Math;
  const $=id=>document.getElementById(id);
  const all=s=>Array.from(document.querySelectorAll(s));
  const state={stage:0,image:'photo',customURL:null,rawImage:null,pixels:null,result:null,ix:24,iy:16,ox:12,oy:8,channel:0,term:0,preset:'mixed',gamma:1,beta:0,shortcut:true,playTimer:null};
  const titles=['画像はRGBの数値配列','Conv：27回の積和で1つの値へ','BatchNorm：チャネルごとに正規化','SiLU：滑らかな非線形変換','CSP：分岐・残差・結合','NumPyで計算を再実装する'];
  const hints=['画像をクリックし、実際のR・G・Bを調べます。','選択した出力値の27項を1項ずつ足し上げます。','平均・分散を計算し、γとβを変えてみましょう。','負の値も滑らかに変化することを確認します。','1つのテンソルが2経路へ分岐し、最後に結合されます。','表示した演算を自分のNumPyコードで再現します。'];
  const fmt=(x,n=5)=>Number.isFinite(x)?x.toFixed(n):'—';
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  const v=(t,y,x,c)=>M.at(t,y,x,c);
  const selected=(t)=>({y:Math.min(t.h-1,state.stage===0?state.iy:state.oy),x:Math.min(t.w-1,state.stage===0?state.ix:state.ox)});
  function canvasInputToTensor(img){
    const can=document.createElement('canvas');can.width=48;can.height=32;
    const ctx=can.getContext('2d',{willReadFrequently:true});ctx.drawImage(img,0,0,48,32);
    const pixels=ctx.getImageData(0,0,48,32).data;
    const out=M.newTensor(32,48,3);
    for(let y=0;y<32;y++)for(let x=0;x<48;x++)for(let c=0;c<3;c++)M.put(out,y,x,c,pixels[(y*48+x)*4+c]/255);
    state.pixels=pixels;return out;
  }
  function loadPhoto(url,key,revoke=false){
    const image=new Image();
    image.onload=()=>{
      if(image.width>8000||image.height>8000){$('imageStatus').textContent='縦横8000px以内の画像を選んでください';if(revoke)URL.revokeObjectURL(url);return;}
      state.image=key;state.rawImage=image;state.input=canvasInputToTensor(image);recalculate();
      all('[data-image]').forEach(b=>b.classList.toggle('chosen',b.dataset.image===key));
      $('imageStatus').textContent=key==='custom'?'画像を読み込みました（外部送信なし）':'端末内で処理中';
      if(revoke){if(state.customURL)URL.revokeObjectURL(state.customURL);state.customURL=null;}
    };
    image.onerror=()=>{$('imageStatus').textContent='画像を読み込めませんでした';if(revoke)URL.revokeObjectURL(url);};
    image.src=url;
  }
  function recalculate(){state.result=M.run(state.input,{preset:state.preset,gamma:state.gamma,beta:state.beta,shortcut:state.shortcut});refresh();}
  function activeTensor(){const r=state.result;if(!r)return null;return [r.input,r.conv,r.bn,r.act,r.csp.fused,r.csp.fused][state.stage];}
  function drawInput(){const ctx=$('photoCanvas').getContext('2d');ctx.drawImage(state.rawImage,0,0,720,480);
    const ox=state.stage===0?state.ix:(state.ox*2), oy=state.stage===0?state.iy:(state.oy*2);
    ctx.strokeStyle='rgba(255,255,255,.3)';ctx.lineWidth=.5;ctx.beginPath();for(let x=0;x<=48;x++){ctx.moveTo(x*15,0);ctx.lineTo(x*15,480);}for(let y=0;y<=32;y++){ctx.moveTo(0,y*15);ctx.lineTo(720,y*15)}ctx.stroke();
    const size=state.stage===0?1:3;
    ctx.fillStyle='rgba(238,154,42,.2)';ctx.fillRect((ox-1)*15,(oy-1)*15,size*15,size*15);
    ctx.strokeStyle='#f6ad51';ctx.lineWidth=3;ctx.strokeRect((ox-1)*15,(oy-1)*15,size*15,size*15);
    ctx.fillStyle='#20dac2';ctx.fillRect(ox*15,oy*15,15,15);
    $('pixelLabel').textContent=`入力 (${ox}, ${oy})`;
  }
  function displayColor(value,scale,isInput){
    if(isInput){const z=clamp(Math.round(value*255),0,255);return [z,z,z,255];}
    const t=Math.pow(clamp(Math.abs(value)/Math.max(1e-9,scale),0,1),.65);
    const base=[235,244,239],positive=[19,142,101],negative=[54,113,181],target=value>=0?positive:negative;
    return [0,1,2].map(i=>Math.round(base[i]+(target[i]-base[i])*t)).concat(255);
  }
  function texture(t,c){
    const can=document.createElement('canvas');can.width=t.w;can.height=t.h;const ctx=can.getContext('2d');
    const img=ctx.createImageData(t.w,t.h);let mx=0;
    for(let y=0;y<t.h;y++)for(let x=0;x<t.w;x++)mx=Math.max(mx,Math.abs(v(t,y,x,c)));
    for(let y=0;y<t.h;y++)for(let x=0;x<t.w;x++){
      const rgb=displayColor(v(t,y,x,c),mx,state.stage===0 && t===state.result.input);
      img.data.set(rgb,(y*t.w+x)*4);
    }
    ctx.putImageData(img,0,0);return can;
  }
  function drawMap(){const t=activeTensor();const canvas=$('mapCanvas'),ctx=canvas.getContext('2d');const sel=selected(t);
    ctx.clearRect(0,0,600,400);ctx.imageSmoothingEnabled=false;ctx.drawImage(texture(t,state.channel),0,0,600,400);
    const sx=600/t.w,sy=400/t.h;
    ctx.strokeStyle='rgba(255,255,255,.25)';ctx.lineWidth=.7;ctx.beginPath();for(let i=0;i<=t.w;i++){ctx.moveTo(i*sx,0);ctx.lineTo(i*sx,400);}for(let i=0;i<=t.h;i++){ctx.moveTo(0,i*sy);ctx.lineTo(600,i*sy);}ctx.stroke();
    ctx.strokeStyle='#eca53d';ctx.lineWidth=4;ctx.strokeRect(sel.x*sx+1,sel.y*sy+1,sx-2,sy-2);
    $('tensorShape').textContent=`${t.w}×${t.h}×${t.c}`;
    $('mapValue').textContent=fmt(v(t,sel.y,sel.x,state.channel),4);
    $('mapHeading').textContent=['RGB入力','Conv出力','BN出力','SiLU出力','CSP型Block出力','CSP型Block出力'][state.stage];
    const ch=$('channel');const count=t.c;
    if(ch.options.length!==count){ch.replaceChildren();for(let i=0;i<count;i++){const opt=document.createElement('option');opt.value=i;opt.textContent=state.stage===0?['R（赤）','G（緑）','B（青）'][i]:`ch ${i}`;ch.append(opt);}}
    state.channel=clamp(state.channel,0,count-1);ch.value=state.channel;
    $('paramCount').textContent=['0','108','8','0','72','188（全体）'][state.stage];
  }
  function drawStack(){const r=state.result,cv=$('stackCanvas'),ctx=cv.getContext('2d');ctx.clearRect(0,0,900,260);ctx.fillStyle='#f3f7f3';ctx.fillRect(0,0,900,260);
    const stages=[['INPUT',r.input],['Conv',r.conv],['BN',r.bn],['SiLU',r.act],['CSP',r.csp.fused]];
    for(let k=0;k<stages.length;k++){
      const [label,t]=stages[k],x=35+k*179;
      for(let c=Math.min(2,t.c-1);c>=0;c--){
        const image=texture(t,c),px=x+c*10,py=65-c*9;
        ctx.save();ctx.translate(px,py);ctx.transform(1.9,.25,-.55,2.1,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(image,0,0,t.w,t.h);ctx.strokeStyle='#9fbfb0';ctx.lineWidth=.8;ctx.strokeRect(0,0,t.w,t.h);ctx.restore();
      }
      if(k===0||k===state.stage||k===4&&state.stage===5){ctx.fillStyle='#248f67';ctx.fillRect(x,12,125,4);}
      ctx.font='bold 14px system-ui';ctx.textAlign='center';ctx.fillStyle='#234d3e';ctx.fillText(label,x+57,28);
      ctx.font='11px system-ui';ctx.fillStyle='#62756a';ctx.fillText(`${t.w}×${t.h}×${t.c}`,x+58,214);
      if(k<4){ctx.font='22px system-ui';ctx.fillStyle='#7f9d8f';ctx.fillText('→',x+158,124);}
    }
    ctx.font='11px system-ui';ctx.textAlign='center';ctx.fillStyle='#667e70';ctx.fillText('各板は実計算されたチャネル。構造理解のため斜めに投影しています。',450,247);
  }
  function formula(s){return `<div class="formula">${s}</div>`;}
  function inputPanel(){const rgb=[0,1,2].map(c=>Math.round(255*v(state.result.input,state.iy,state.ix,c)));
    return `<h4>まず1ピクセルの意味を確認</h4><p>画像を <strong>48×32×3</strong> に縮小しています。RGBは3つの独立したチャネルで、1位置に数値が3個あります。</p>${formula(`(${state.ix}, ${state.iy}) の RGB = (${rgb.join(', ')}) / 255`)}<p>次は3×3の参照範囲を各入力チャネルに重ね、27個の積を合計します。3チャネルの「画像を3分割する」という意味ではありません。</p>`;
  }
  function convPanel(){const r=state.result,terms=M.convTerms(r.input,r.weights,state.oy,state.ox,state.channel);
    const total=terms.reduce((a,t)=>a+t.product,0),partial=terms.slice(0,state.term).reduce((a,t)=>a+t.product,0);
    const term=terms[Math.min(state.term,terms.length-1)];
    return `<h4>なぜ出力が24×16×4になる？</h4>${formula(`⌊(48 + 2×1 − 3) / 2⌋ + 1 = 24<br>⌊(32 + 2×1 − 3) / 2⌋ + 1 = 16`)}
    <p>各出力チャネルでは <strong>入力3チャネル × Kernel 3×3 = 27項</strong> を加算します。出力チャネル数4はKernelバンクが4組あるためです。</p>
    <table class="math-table"><tr><th>表示</th><th>値</th></tr><tr><td>選択した出力位置</td><td>(${state.ox}, ${state.oy}), ch${state.channel}</td></tr><tr><td>ここまでの和（${state.term}/27）</td><td>${fmt(partial,6)}</td></tr><tr><td>全27項の和</td><td>${fmt(total,6)}</td></tr><tr><td>Convマップ実値</td><td>${fmt(v(r.conv,state.oy,state.ox,state.channel),6)}</td></tr></table>
    ${formula(`項 ${Math.min(state.term+1,27)}：RGB ${term.channel} の入力 ${fmt(term.input,4)} × 重み ${fmt(term.weight,4)} = ${fmt(term.product,6)}`)}
    <p>実際のConvはKernelを反転せず、入力と重みを同じ位置で掛ける相互相関演算です。</p>`;
  }
  function bnPanel(){const r=state.result,s=r.stats[state.channel],x=v(r.conv,state.oy,state.ox,state.channel),y=v(r.bn,state.oy,state.ox,state.channel);
    return `<h4>何を正規化するのか</h4><p>ここでは <strong>1画像の24×16=384位置</strong> から各チャネルの平均μと分散σ²を計算します。実際の学習では通常、ミニバッチ全体の N×H×W を対象に統計を取ります。</p>
    ${formula(`y = γ × (x − μ) / √(σ² + ε) + β`)}
    <table class="math-table"><tr><th>値</th><th>選択ch ${state.channel}</th></tr><tr><td>Conv出力 x</td><td>${fmt(x,6)}</td></tr><tr><td>平均 μ</td><td>${fmt(s.mean,6)}</td></tr><tr><td>分散 σ²</td><td>${fmt(s.var,6)}</td></tr><tr><td>γ / β</td><td>${fmt(state.gamma,2)} / ${fmt(state.beta,2)}</td></tr><tr><td>BN出力 y</td><td>${fmt(y,6)}</td></tr></table>
    <p class="step-box">ここは学習時のBatchNormを1枚の画像で模した計算です。推論時は通常、学習中に蓄積したrunning mean/varianceを利用します。ここにはrunning統計を実装していません。</p>`;
  }
  function siluPanel(){const x=v(state.result.bn,state.oy,state.ox,state.channel),y=v(state.result.act,state.oy,state.ox,state.channel);
    return `<h4>活性化関数が必要な理由</h4><p>ConvとBNだけなら、途中の変換には線形性・アフィン性が残ります。SiLUを挟むことで非線形の特徴変換ができます。</p>${formula(`SiLU(x) = x × sigmoid(x)<br>sigmoid(x) = 1 / (1 + exp(−x))`)}
    <table class="math-table"><tr><th>選択位置の入力</th><th>出力</th></tr><tr><td>BN：${fmt(x,5)}</td><td>SiLU：${fmt(y,5)}</td></tr><tr><td>sigmoid(x)</td><td>${fmt(M.sigmoid(x),5)}</td></tr><tr><td>ReLU比較</td><td>${fmt(Math.max(0,x),5)}</td></tr></table>
    <p>負の値をすべてゼロにするReLUとは異なり、SiLUは負の入力でも滑らかな応答を持ちます。</p>`;
  }
  function cspPanel(){const c=state.result.csp,x=state.ox,y=state.oy,vec=t=>[...Array(t.c)].map((_,ch)=>fmt(v(t,y,x,ch),4)).join(', ');
    return `<h4>分岐しても情報はどうつながる？</h4>
    <p>4チャネルを2本の<strong>1×1射影</strong>へ渡します。片方はボトルネックへ、もう片方は近道として保持します。最後にチャネル方向でConcatし、1×1で混合します。</p>
    ${formula(`4ch → [Main 2ch + Bypass 2ch] → Concat 4ch → Fuse 4ch`)}
    <table class="math-table"><tr><th>経路（選択位置）</th><th>特徴値</th></tr>
    <tr><td>Main（1×1＋SiLU）</td><td>${vec(c.branch0)}</td></tr>
    <tr><td>Bottleneck出力</td><td>${vec(c.inner2)}</td></tr>
    <tr><td>Main＋Residual</td><td>${vec(c.main)}</td></tr>
    <tr><td>Bypass</td><td>${vec(c.branch1)}</td></tr>
    <tr><td>Concat 4ch</td><td>${vec(c.joined)}</td></tr>
    <tr><td>Fuse 4ch</td><td>${vec(c.fused)}</td></tr></table>
    <p class="step-box">これはCSPの考え方を追う簡略実装です。公式YOLOXのCSPLayerは各枝にConv-BN-SiLU Blockを配置するため、構造と重み互換性は同一ではありません。</p>`;
  }
  function exercisePanel(){return `<h4>目で見た計算をコードへ</h4><p>これまでの表示はすべて独立した固定重みから得た計算結果です。完成したコードを写す前に、出力形状を自分で導出してください。</p>
    <div class="step-box">問題：入力48×32×3、Convの出力4ch、Kernel 3×3、stride2、padding1。BN・SiLUの後、CSP型Blockで空間サイズを維持したら、最終形状は？</div>
    <div class="control-row"><label>幅</label><input id="answerW" type="number" min="1" max="99" placeholder="?" style="max-width:100px;border:1px solid #d9e4dc;border-radius:8px;padding:7px"></div><div class="control-row"><label>高さ</label><input id="answerH" type="number" min="1" max="99" placeholder="?" style="max-width:100px;border:1px solid #d9e4dc;border-radius:8px;padding:7px"></div>
    <div class="control-row"><label>チャネル</label><input id="answerC" type="number" min="1" max="99" placeholder="?" style="max-width:100px;border:1px solid #d9e4dc;border-radius:8px;padding:7px"></div><div class="control-buttons"><button type="button" id="checkAnswer">判定する</button></div><p id="answerFeedback" role="status"></p>
    <pre class="code-preview"># まずはNumPyでConvを実装
for oy in range(out_h):
    for ox in range(out_w):
        patch = padded[oy*stride:oy*stride+3,
                       ox*stride:ox*stride+3, :]
        # ここに27項の積和を書く</pre>
    <p><a href="../../downloads/unit03a_numpy_starter.zip" download>学生用NumPy演習ZIP</a>に未完成関数とテストがあります。PyTorchのNCHW形式へ変換する発展演習も用意しています。</p>`;}
  function renderLesson(){const t=state.stage;
    $('lessonKicker').textContent=`LESSON ${String(t+1).padStart(2,'0')}`;
    $('lessonTitle').textContent=titles[t];$('lessonLead').textContent=hints[t];$('stageNum').textContent=`${String(t+1).padStart(2,'0')} / 06`;
    $('explanation').innerHTML=[inputPanel,convPanel,bnPanel,siluPanel,cspPanel,exercisePanel][t]();
    const c=$('controls');c.replaceChildren();
    if(t===1){
      const row=document.createElement('div');row.className='control-row';row.innerHTML=`<label for="termSlider">積を足す順番 <b>${state.term} / 27</b></label><input id="termSlider" type="range" min="0" max="27" step="1" value="${state.term}">`;
      c.append(row);$('termSlider').addEventListener('input',e=>{state.term=+e.target.value;renderLesson()});
      const buttons=document.createElement('div');buttons.className='control-buttons';buttons.innerHTML='<button type="button" id="termPrev">−1項</button><button type="button" id="termNext">＋1項</button><button type="button" id="termAll">すべて計算</button>';c.append(buttons);
      $('termPrev').onclick=()=>{state.term=Math.max(0,state.term-1);renderLesson()};$('termNext').onclick=()=>{state.term=Math.min(27,state.term+1);renderLesson()};$('termAll').onclick=()=>{state.term=27;renderLesson()};
    }
    if(t===2){for(const [key,min,max,step,lab] of [['gamma',.2,2,.05,'γ（拡大率）'],['beta',-1,1,.05,'β（移動量）']]){
      const row=document.createElement('div');row.className='control-row';row.innerHTML=`<label for="bn_${key}">${lab} <b>${fmt(state[key],2)}</b></label><input id="bn_${key}" type="range" min="${min}" max="${max}" step="${step}" value="${state[key]}">`;c.append(row);
      $(`bn_${key}`).oninput=e=>{state[key]=+e.target.value;recalculate()};}
    }
    if(t===4){const row=document.createElement('div');row.className='control-buttons';row.innerHTML=`<button type="button" id="shortcutToggle">残差接続：${state.shortcut?'ON':'OFF'}</button>`;c.append(row);$('shortcutToggle').onclick=()=>{state.shortcut=!state.shortcut;recalculate()};}
    if(t===5){$('checkAnswer').onclick=()=>{const a=[Number($('answerW').value),Number($('answerH').value),Number($('answerC').value)];$('answerFeedback').textContent=a[0]===24&&a[1]===16&&a[2]===4?'正解。24×16×4です。今度はPythonで同じ形状と計算を再現してください。':'出力サイズの式と、Concat後に最後の1×1 Convで4chへ戻る点を確認してください。';};}
    $('prevStage').disabled=t===0;$('nextStage').disabled=t===5;
  }
  function refresh(){if(!state.result)return;all('[data-stage]').forEach(b=>{b.classList.toggle('selected',+b.dataset.stage===state.stage);b.setAttribute('aria-current',+b.dataset.stage===state.stage?'step':'false');});
    all('[data-pipeline]').forEach(b=>b.classList.toggle('selected',+b.dataset.pipeline===Math.min(state.stage,4)));
    $('progressText').textContent=`${String(state.stage+1).padStart(2,'0')} / 06`;$('progressFill').style.width=`${(state.stage+1)/6*100}%`;
    drawInput();drawMap();drawStack();renderLesson();}
  function setStage(n,scroll=false){state.stage=clamp(n,0,5);state.channel=0;state.term=0;refresh();history.replaceState(null,'',`#step-${state.stage+1}`);if(scroll)$('workspace').scrollIntoView({behavior:'smooth',block:'start'});}
  function init(){
    all('[data-stage]').forEach(b=>b.addEventListener('click',()=>{stopPlay();setStage(+b.dataset.stage)}));
    $('prevStage').onclick=()=>{stopPlay();setStage(state.stage-1)};
    $('nextStage').onclick=()=>{stopPlay();setStage(state.stage+1)};
    $('channel').onchange=e=>{state.channel=+e.target.value;drawMap();renderLesson();};
    $('photoCanvas').onclick=e=>{const r=e.currentTarget.getBoundingClientRect();state.ix=clamp(Math.floor((e.clientX-r.left)/r.width*48),0,47);state.iy=clamp(Math.floor((e.clientY-r.top)/r.height*32),0,31);state.ox=Math.floor(state.ix/2);state.oy=Math.floor(state.iy/2);refresh();};
    $('mapCanvas').onclick=e=>{const r=e.currentTarget.getBoundingClientRect(),t=activeTensor();let x=clamp(Math.floor((e.clientX-r.left)/r.width*t.w),0,t.w-1),y=clamp(Math.floor((e.clientY-r.top)/r.height*t.h),0,t.h-1);if(state.stage===0){state.ix=x;state.iy=y;state.ox=Math.floor(x/2);state.oy=Math.floor(y/2);}else{state.ox=x;state.oy=y;state.ix=clamp(x*2,0,47);state.iy=clamp(y*2,0,31);}refresh();};
    all('[data-image]').forEach(b=>b.onclick=()=>{stopPlay();loadPhoto(b.dataset.image==='photo'?'../../assets/003Z-selfie.png':'../01-cnn/assets/person.jpg',b.dataset.image)});
    $('fileInput').onchange=e=>{const f=e.target.files?.[0];e.target.value='';if(!f)return;if(!['image/png','image/jpeg','image/webp'].includes(f.type)||f.size>12*1024*1024){$('imageStatus').textContent='PNG/JPEG/WebP、12MB以下を選んでください';return;}loadPhoto(URL.createObjectURL(f),'custom',true)};
    $('startBtn').onclick=()=>{if(state.playTimer){stopPlay();return;}setStage(0,true);$('startBtn').textContent='Ⅱ 自動解説を停止';state.playTimer=setInterval(()=>{if(state.stage>=5){stopPlay();return;}setStage(state.stage+1);},4200);};
    const fragment=location.hash.match(/^#step-([1-6])$/);state.stage=fragment?+fragment[1]-1:0;
    loadPhoto('../../assets/003Z-selfie.png','photo');
    window.YVA03={getState:()=>({stage:state.stage,image:state.image,channel:state.channel,ix:state.ix,iy:state.iy,ox:state.ox,oy:state.oy,gamma:state.gamma,beta:state.beta,shortcut:state.shortcut}),getDiagnostics:()=>{if(!state.result)return null;let t=activeTensor(),sel=selected(t);return {shape:[t.h,t.w,t.c],value:v(t,sel.y,sel.x,state.channel),convTerms:M.convTerms(state.result.input,state.result.weights,state.oy,state.ox,Math.min(state.channel,3))}},chooseStage:n=>setStage(n),selectPixel:(x,y)=>{state.ix=clamp(x|0,0,47);state.iy=clamp(y|0,0,31);state.ox=Math.floor(state.ix/2);state.oy=Math.floor(state.iy/2);refresh();}};
  }
  function stopPlay(){if(state.playTimer){clearInterval(state.playTimer);state.playTimer=null;$('startBtn').textContent='▶ 6ステップを体験する';}}
  init();
})();
