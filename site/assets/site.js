// Conceptual visualization only: actual numeric convolution is in the lab.
const pb=document.getElementById('pixelBoard');const mm=document.getElementById('miniMap');
if(pb&&mm){for(let i=0;i<36;i++){let b=document.createElement('b');let x=i%6,y=Math.floor(i/6);let r=110+Math.round(24*x+8*y),g=80+Math.round(18*y+7*x),bl=85+Math.round(12*x+5*y);b.style.background=`rgb(${Math.min(255,r)},${Math.min(255,g)},${Math.min(255,bl)})`;pb.appendChild(b);}
for(let i=0;i<16;i++){let b=document.createElement('i');let x=i%4,y=Math.floor(i/4);b.style.background=`rgba(${x%2?44:185},${85+y*25},${y%2?95:135},.65)`;mm.appendChild(b);}}
