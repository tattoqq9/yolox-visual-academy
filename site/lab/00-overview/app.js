'use strict';
(() => {
  const $ = (s) => document.querySelector(s);
  const all = (s) => [...document.querySelectorAll(s)];
  const stageData = [
    {label:'STEP 01 / INPUT',name:'AIが受け取るのは、色の数値。',body:'写真は画素（Pixel）の集まりです。各画素には赤（R）・緑（G）・青（B）の数値があり、モデルにはこの数値配列が入力されます。',term:'RGBとPixel（画素）',definition:'RGBは1画素の色の3成分。例えば416×416×3なら416×416個の位置に3値ずつあります。',hint:'ここでは「頭部」などの意味をまだ予測していません。',link:'../01-cnn/index.html?step=1',linkText:'数値とRGBを詳しく：Unit 01 →',status:'INPUT / RGB',shape:'元画像 960×640',caption:'この写真から、モデルに入力する数値を作ります。'},
    {label:'STEP 02 / CONVOLUTION',name:'小さな窓で、1つの数値を計算する。',body:'3×3の参照領域に重みを掛け、合計する計算を画像の各位置へ繰り返します。これが畳み込み（Conv）の基本です。',term:'Kernel（カーネル）',definition:'計算に使う小さな重みの集まり。3×3なら9個の場所を参照します。カラー入力ではチャネルごとに重みがあります。',hint:'この枠は演算の位置を示した模式図です。実際の重みや出力ではありません。',link:'../01-cnn/index.html?step=2',linkText:'9項の計算を試す：Unit 01 →',status:'3×3 KERNEL',shape:'概念：3×3の参照位置',caption:'黄色い3×3領域は、計算するときに参照する画素の例です。'},
    {label:'STEP 03 / BACKBONE',name:'いろいろな特徴を段階的に取り出す。',body:'畳み込みなどの処理を何層も重ね、低次の模様から、より複雑な情報を表現する数値を作ります。出力の配列が「特徴マップ」です。',term:'Feature Map（特徴マップ）',definition:'CNNの中間計算値の配列。たくさんのチャネルがあり、それぞれに異なる反応が現れます。',hint:'板を重ねた図はチャネルの模式図です。頭の輪郭などの意味が自動で一枚に現れるという意味ではありません。',link:'../03-nano/index.html',linkText:'本物のネットワーク構造：Unit 03-B →',status:'BACKBONE',shape:'例：stride 8/16/32',caption:'画像から複数解像度の特徴が生成されます（模式図）。'},
    {label:'STEP 04 / PAFPN',name:'大きさの違う特徴を、組み合わせる。',body:'小さい領域を細かく見る特徴と、広い範囲を捉える特徴を、上下方向の経路で融合します。P3・P4・P5は異なる空間解像度の出力です。',term:'PAFPN / P3・P4・P5',definition:'Backboneからの情報を融合し、異なる解像度に分けてHeadへ送る仕組み。P3/P4/P5は後段の出力です。',hint:'例：416×416入力ならP3=52×52、P4=26×26、P5=13×13。これは格子数で、頭部の個数ではありません。',link:'../03-fpn-head/index.html',linkText:'特徴融合を詳しく：Unit 03-C →',status:'FEATURE FUSION',shape:'P3:52² / P4:26² / P5:13²',caption:'特徴の流れは上方向と下方向の両方にあります。'},
    {label:'STEP 05 / DECOUPLED HEAD',name:'位置・物体らしさ・種類を別々に予測。',body:'Headは各候補位置についてBBoxの4値、Objectnessの1値、クラスの値を出力します。これらは最終的な検出結果そのものではなく、後処理前の予測です。',term:'BBox / Objectness / Classification',definition:'BBox＝位置・幅・高さ。Objectness＝物体らしさ。Classification＝どのクラスかの予測。クラスの数はモデル設定で変わります。',hint:'公開済みの80クラス構成なら1候補あたり4+1+80=85個のraw値です。1クラス頭部検出器なら4+1+1=6値です。',link:'../03-fpn-head/index.html',linkText:'Headの実テンソル：Unit 03-C →',status:'DECOUPLED HEAD',shape:'4 BBox + 1 Obj + C classes',caption:'Headが出力する数値を計算し、後段でBBoxへ復元します。'},
    {label:'STEP 06 / POSTPROCESS',name:'多くの候補から、最終的なBBoxを選ぶ。',body:'候補をスコアで絞り、重複するBBoxをNMSで整理します。残った候補が最終的な検出結果になります。',term:'NMS / Bounding Box',definition:'NMSは重複候補を整理する後処理。BBoxは対象を囲む四角形。複数の頭があれば複数のBBoxが出ることもあります。',hint:'黄色い四角形は、人が例として描いたものです。この未学習モデルの実際の検出結果・精度ではありません。',link:'../04-detection/index.html',linkText:'BBoxとNMSを詳しく：Unit 04 →',status:'BBOX RESULT',shape:'例示用のBBox（非推論）',caption:'黄色い枠は教師用の概念例で、モデルが検出したわけではありません。'}
  ];
  const state = {stage: 0, playing: false, timer: null, trace: null, traceStage: 'dark2'};
  const select = $('#traceStage');
  function setStage(index, {keepPlaying=false} = {}) {
    if (!Number.isInteger(index) || index < 0 || index >= stageData.length) return;
    if (!keepPlaying) pause();
    state.stage = index;
    const data = stageData[index];
    all('[data-stage]').forEach((button) => {
      const active = Number(button.dataset.stage) === index;
      button.classList.toggle('is-active', active);
      if (active) button.setAttribute('aria-current', 'step'); else button.removeAttribute('aria-current');
    });
    $('#stepCount').textContent = `${index + 1} / 6`;
    $('#progressFill').style.width = `${(index + 1) / 6 * 100}%`;
    $('#stageKicker').textContent = data.label;
    $('#stageTitle').textContent = data.name;
    $('#stageBody').textContent = data.body;
    $('#termName').textContent = data.term;
    $('#termDescription').textContent = data.definition;
    $('#stageExample').textContent = data.hint;
    $('#stageLink').textContent = data.linkText;
    $('#stageLink').href = data.link;
    $('#visualStatus').textContent = data.status;
    $('#visualShape').textContent = data.shape;
    $('#imageCaption').textContent = data.caption;
    $('#stageGrid').classList.toggle('show', index === 1);
    $('#kernelOverlay').classList.toggle('show', index === 1);
    $('#stageImage').querySelector('img').classList.toggle('is-muted', [2,3,4].includes(index));
    $('#illustrativeBox').hidden = index !== 5;
    $('#tensorIllustration').hidden = ![2,3].includes(index);
    $('#predictionBranches').hidden = index !== 4;
    $('#tensorCaption').textContent = index === 3
      ? 'P3・P4・P5の特徴を融合します。各板は特徴マップの模式図です。'
      : '1枚の写真から複数チャネルの特徴が生まれます（模式図）。';
    $('#prevFlow').disabled = index === 0;
    $('#nextFlow').disabled = index === stageData.length - 1;
  }
  function pause() {
    state.playing = false;
    clearInterval(state.timer);
    state.timer = null;
    $('#playIcon').textContent = '▶';
    $('#playLabel').textContent = '最初から再生';
    $('#playFlow').setAttribute('aria-pressed', 'false');
  }
  function play() {
    if (state.playing) { pause(); return; }
    setStage(0);
    state.playing = true;
    $('#playIcon').textContent = 'Ⅱ';
    $('#playLabel').textContent = '一時停止';
    $('#playFlow').setAttribute('aria-pressed', 'true');
    state.timer = setInterval(() => {
      if (state.stage === stageData.length - 1) { pause(); return; }
      setStage(state.stage + 1, {keepPlaying:true});
    }, 2800);
  }
  function colorFromValue(value, maxAbs) {
    const t = Math.min(1, Math.pow(Math.abs(value) / Math.max(maxAbs, 1e-12), .63));
    const base = [231,239,233], target = value >= 0 ? [211,106,60] : [61,110,173];
    return base.map((v, i) => Math.round(v + (target[i] - v) * t));
  }
  function paintTrace() {
    if (!state.trace) return;
    const m = state.trace.maps.find((x) => x.id === state.traceStage);
    if (!m) return;
    const w = m.sample.width, h = m.sample.height;
    const layer = document.createElement('canvas');layer.width = w;layer.height = h;
    const ctx = layer.getContext('2d');
    const pixels = ctx.createImageData(w,h);
    const ma = Math.max(...m.sample.values.map(Math.abs), 1e-12);
    for (let i = 0; i < w*h; i++) {
      const c = colorFromValue(m.sample.values[i], ma);
      pixels.data.set([...c, 255], i * 4);
    }
    ctx.putImageData(pixels,0,0);
    const canvas = $('#traceCanvas'), dst = canvas.getContext('2d');
    dst.clearRect(0, 0, canvas.width, canvas.height);
    dst.imageSmoothingEnabled = false;
    dst.drawImage(layer, 0, 0, canvas.width, canvas.height);
    $('#traceShape').textContent = `実テンソル [${m.shape.join('×')}] / 表示 ${w}×${h} 抽出 / ch ${m.channel}`;
    $('#traceValue').textContent = `min ${Math.min(...m.sample.values).toExponential(2)}, max ${Math.max(...m.sample.values).toExponential(2)}`;
    $('#tracePlaceholder').hidden = true;
  }
  async function loadTrace() {
    const button = $('#loadTrace');button.disabled = true;button.textContent = '実数値を読み込み中…';
    try {
      const response = await fetch('actual-untrained-trace.json', {cache:'no-store'});
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const parsed = await response.json();
      if (parsed.schema !== 'yva.unit00.untrained-preview.v1' || parsed.trained !== false || parsed.maps.length !== 3) throw new Error('データ形式が違います');
      for (const m of parsed.maps) if (m.sample.width*m.sample.height !== m.sample.values.length || !m.sample.values.every(Number.isFinite)) throw new Error('テンソルの形が不正です');
      state.trace = parsed;
      select.value = state.traceStage;
      paintTrace();
      button.textContent = '実PyTorch値を表示中';
    } catch (error) {
      $('#tracePlaceholder').textContent = `データを読み込めませんでした：${error.message}。ローカルではHTTPサーバーで開いてください。`;
      button.disabled = false;button.textContent = '再読み込み';
    }
  }
  all('[data-stage]').forEach(btn => btn.addEventListener('click', () => setStage(Number(btn.dataset.stage))));
  $('#playFlow').addEventListener('click', play);
  $('#prevFlow').addEventListener('click', () => setStage(state.stage - 1));
  $('#nextFlow').addEventListener('click', () => setStage(state.stage + 1));
  $('#loadTrace').addEventListener('click', loadTrace);
  select.addEventListener('change', (e) => {state.traceStage = e.target.value;paintTrace();});
  window.addEventListener('keydown', (e) => {
    if (['INPUT','TEXTAREA','SELECT','BUTTON','SUMMARY'].includes(document.activeElement?.tagName)) return;
    if (e.altKey && e.key === 'ArrowRight') {e.preventDefault();setStage(Math.min(5,state.stage + 1));}
    if (e.altKey && e.key === 'ArrowLeft') {e.preventDefault();setStage(Math.max(0,state.stage - 1));}
  });
  window.YVAUnit00 = Object.freeze({step: (s) => setStage(s), getState: () => ({stage:state.stage, playing:state.playing, trained:state.trace?.trained ?? null, traceId:state.traceStage})});
  setStage(0);
})();
