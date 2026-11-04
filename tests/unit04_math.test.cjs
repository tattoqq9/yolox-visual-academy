/* Run with node --test tests/unit04_math.test.cjs; no external packages. */
const {test}=require('node:test');const assert=require('node:assert/strict');
const M=require('../site/lab/04-detection/math.js');
const close=(a,b,eps=1e-9)=>assert.ok(Math.abs(a-b)<eps,`${a} ≠ ${b}`);
test('sigmoid stable extremes and bce',()=>{close(M.sigmoid(0),.5);close(M.sigmoid(-1000),0);close(M.sigmoid(1000),1);close(M.bce(0,1),Math.log(2));close(M.bce(-1000,1),1000);});
test('YOLOX bbox decode integer grid, not +0.5',()=>{const b=M.decode([.25,-.5,Math.log(2),Math.log(3)],3,5,8);assert.deepEqual([b.cx,b.cy],[26,36]);close(b.w,16);close(b.h,24);});
test('IoU 4 px 1 px displacement',()=>close(M.iou({cx:0,cy:0,w:4,h:4},{cx:1,cy:0,w:4,h:4}),.6));
test('IoU disjoint, identical, degenerate',()=>{close(M.iou([0,0,4,4],[0,0,4,4]),1);close(M.iou([0,0,4,4],[5,5,9,9]),0);close(M.iou([0,0,0,0],[0,0,0,0]),0)});
test('NMS is class aware unless toggled',()=>{const c=[{id:'a',box:[0,0,10,10],score:.9,cls:0},{id:'b',box:[1,0,11,10],score:.8,cls:1},{id:'c',box:[2,0,12,10],score:.7,cls:0}];assert.equal(M.nms(c,.5,.2,true).keep.length,2);assert.equal(M.nms(c,.5,.2,false).keep.length,1)});
test('grid geometry center with STRICT 1.5 stride',()=>{const gt={cx:4,cy:4};assert.equal(M.geometry(4,4,gt,8),true);assert.equal(M.geometry(4,4,{cx:16,cy:4},8),false);});
test('classification cost is finite two-class sqrt obj*cls BCE',()=>{const got=M.classifyCost(-4.595,[-4.595,-4.595],0);assert.ok(got>0);assert.ok(Number.isFinite(got))});
test('Dynamic K exactly top 10 floor max 1',()=>{const c=Array.from({length:12},(_,i)=>({id:i,gx:8,gy:8,stride:8,box:{cx:8,cy:8,w:20,h:20},objLogit:1,clsLogits:[1,0]}));const g=[{cx:8,cy:8,w:20,h:20,cls:0}];const r=M.simota(c,g);assert.equal(r.choices[0].k,10);assert.equal(r.byGT[0].length,10)});
test('two-GT conflict resolves min cost',()=>{const boxes=[{id:'s',gx:8,gy:8,stride:16,box:{cx:8,cy:8,w:20,h:20},objLogit:2,clsLogits:[1,1]}];const g=[{cx:8,cy:8,w:20,h:20,cls:0},{cx:8,cy:8,w:20,h:20,cls:1}];const r=M.simota(boxes,g);assert.equal(r.conflicts.length,1);assert.equal([...r.owner].length,1);assert.equal(r.byGT[0].length+r.byGT[1].length,1)});
test('BCE analytic gradient matches finite diff',()=>{for(const z of [-5,-1,0,2,5])for(const t of [0,.4,1])close(M.finiteDiff(x=>M.bce(x,t),z),M.sigmoid(z)-t,5e-8)});
test('mini loss terms follow 5*IOU + obj + cls',()=>{const l=M.lossTerms({cx:1,cy:0,w:4,h:4},{cx:0,cy:0,w:4,h:4},{obj:0,cls:0,negObj:-3});close(l.iou,.6);close(l.liou,.64);close(l.total,5*l.liou+l.posObj+l.negObj+l.cls)});
