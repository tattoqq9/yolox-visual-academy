'use strict';
/* Static renderer of genuine CPU PyTorch update records; no browser-side training. */
(() => {
  const $=id=>document.getElementById(id);
  const state={data:null,model:'one',snapshot:0,image:0,score:.25,gt:true,pred:true};
  const fmt=(x,d=3)=>Number.isFinite(x)?Number(x).toFixed(d):'—';
  const num=x=>Number(x)||0;
  const size=128;
  const imgs=new Map();
  for(let i=0;i<6;i++){const im=new Image();im.src=`val-${i}.png`;im.onload=()=>{imgs.set(i,im);render()};}
  function models(){return state.data?.experiments[state.model]}
  function snap(){return models()?.snapshots[state.snapshot]}
  function final(){return models()?.snapshots.at(-1)}
  function getPredictions(){
    if(!models())return null;
    if(state.image===0)return snap()?.preview?.[0]?.predictions||[];
    if(state.snapshot!==models().snapshots.length-1)return null;
    return models().final_predictions?.[state.image]?.predictions||[];
  }
  function drawPred(){
    const c=$('predictionCanvas'),ctx=c.getContext('2d');const W=c.width,H=c.height,s=W/size;
    ctx.clearRect(0,0,W,H);ctx.fillStyle='#dce6de';ctx.fillRect(0,0,W,H);
    const im=imgs.get(state.image);if(im)ctx.drawImage(im,0,0,W,H);
    const gt=state.data?.validation_examples?.[state.image]?.gt;
    if(gt&&state.gt){ctx.strokeStyle='#fbe06c';ctx.lineWidth=3;ctx.setLineDash([]);ctx.strokeRect((gt[0]-gt[2]/2)*s,(gt[1]-gt[3]/2)*s,gt[2]*s,gt[3]*s);}
    const all=getPredictions();const shown=(all||[]).filter(x=>x.score>=state.score);
    if(state.pred){ctx.strokeStyle='#159768';ctx.lineWidth=2.3;for(const p of shown){const [x1,y1,x2,y2]=p.box;ctx.strokeRect(x1*s,y1*s,(x2-x1)*s,(y2-y1)*s)}}
    $('predInfo').textContent=all===null ? 'この学習時点の予測は画像01のみ保存しました。他の画像は600更新時点で確認できます。' :
      `保存済み上位${all.length}候補のうち ${shown.length}件を表示。`+(state.image===0?' 実際の学習モデルから取得した予測です。':' 最終モデルの出力です。')+
      ' 閾値変更は表示だけを更新します。AP50を再計算しません。';
  }
  function plot(){
    const c=$('curveCanvas'),cx=c.getContext('2d'),w=c.width,h=c.height;
    cx.clearRect(0,0,w,h);cx.fillStyle='#f8fbf8';cx.fillRect(0,0,w,h);
    const p={l:47,r:15,t:20,b:38},pw=w-p.l-p.r,ph=h-p.t-p.b;
    cx.strokeStyle='#dce5df';cx.lineWidth=1;cx.font='12px system-ui';cx.fillStyle='#647e6e';
    for(let i=0;i<=4;i++){const y=p.t+ph*i/4;cx.beginPath();cx.moveTo(p.l,y);cx.lineTo(w-p.r,y);cx.stroke();cx.fillText((1-i/4).toFixed(2),8,y+4)}
    const entries=models()?.snapshots||[];if(!entries.length)return;
    const max=entries.at(-1).step||1;
    function trace(col,get){cx.strokeStyle=col;cx.lineWidth=3;cx.beginPath();entries.forEach((e,i)=>{const x=p.l+e.step/max*pw,y=p.t+(1-Math.min(1,Math.max(0,get(e))))*ph;if(i)cx.lineTo(x,y);else cx.moveTo(x,y)});cx.stroke()}
    trace('#1a855a',x=>x.training.ap50);trace('#2b79be',x=>x.validation.ap50);
    const e=snap();if(e){const x=p.l+e.step/max*pw;cx.strokeStyle='#d89b28';cx.setLineDash([5,5]);cx.beginPath();cx.moveTo(x,p.t);cx.lineTo(x,p.t+ph);cx.stroke();cx.setLineDash([])}
    cx.fillStyle='#63766a';cx.font='12px system-ui';cx.fillText('0',p.l-2,h-12);cx.fillText(String(max),w-p.r-22,h-12);cx.fillText('更新回数',w/2-25,h-10)
  }
  function render(){
    if(!state.data)return;
    const e=snap();const f=final();if(!e||!f)return;
    $('stepValue').textContent=`${e.step} / ${f.step}`;
    $('metricTrain').textContent=fmt(e.training.ap50);
    $('metricVal').textContent=fmt(e.validation.ap50);
    $('metricCounts').textContent=`${e.validation.tp} / ${e.validation.fp} / ${e.validation.fn}`;
    $('metricLoss').textContent=e.loss?fmt(e.loss.total):'未学習';
    $('shaInfo').textContent=`Weights SHA-256（先頭16桁）：${e.weight_sha256.slice(0,16)} · seed=${state.data.provenance.seed}`;
    $('thresholdOut').textContent=state.score.toFixed(2);
    const early=e.step===0;
    $('insight').textContent=early?'更新前の乱数初期化モデルです。BBox精度とObjectnessが低い状態を観察できます。'
      : `更新${e.step}回の実測：訓練AP50 ${fmt(e.training.ap50)}、未学習画像AP50 ${fmt(e.validation.ap50)}。`+
        (state.model==='one'?' 1枚だけ学習したモデルは評価モードでの精度が安定しないことがあります。BatchNorm統計の影響も検討してください。':' 8枚学習でも6枚の小規模検証から汎化性能を断定できません。');
    $('compareRows').innerHTML=['one','eight'].map(key=>{const s=state.data.experiments[key].snapshots.at(-1);return `<tr class="${key===state.model?'active-row':''}"><td>${key==='one'?'1枚':'8枚'}学習</td><td>${fmt(s.training.ap50)}</td><td>${fmt(s.validation.ap50)}</td><td>${s.validation.tp} / ${s.validation.fp} / ${s.validation.fn}</td></tr>`}).join('');
    drawPred();plot();
  }
  async function start(){
    try{
      const response=await fetch('trace05.json',{cache:'no-store'});
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const data=await response.json();
      if(data.schema!=='yva.unit05.public.v1'||!data.experiments?.one||!data.experiments?.eight)throw new Error('trace schema mismatch');
      state.data=data;
      $('stepSlider').max=data.experiments.one.snapshots.length-1;
      $('stepMilestones').textContent=data.experiments.one.snapshots.map(x=>x.step).join(' / ');
      document.querySelectorAll('input[name=model]').forEach(input=>input.addEventListener('change',()=>{if(input.checked){state.model=input.value;state.snapshot=Math.min(state.snapshot,models().snapshots.length-1);render()}}));
      $('stepSlider').addEventListener('input',e=>{state.snapshot=Number(e.target.value);render()});
      $('imageChoice').addEventListener('change',e=>{state.image=Number(e.target.value);render()});
      $('scoreSlider').addEventListener('input',e=>{state.score=Number(e.target.value);render()});
      $('showGT').addEventListener('change',e=>{state.gt=e.target.checked;render()});
      $('showPred').addEventListener('change',e=>{state.pred=e.target.checked;render()});
      render();
    }catch(err){$('predInfo').textContent=`データを読み込めませんでした：${String(err)}。HTTPサーバー経由で開いてください。`;console.error(err)}
  }
  start();
})();
