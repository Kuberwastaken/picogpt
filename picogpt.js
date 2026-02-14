#!/usr/bin/env node
// PicoGPT.js — GPT trained from scratch in JavaScript
// Port of picogpt.py by @kuberwastaken
// Custom autograd · multi-head attention · AdamW · training & inference

import { readFileSync, writeFileSync, existsSync } from 'fs';

// ── Seeded PRNG (xoshiro128**) ──────────────────────────────────────────────
const _s = new Uint32Array(4);
const seed = v => { let s = v|0; for (let i = 0; i < 4; i++) _s[i] = (s = Math.imul(s,1664525)+1013904223) >>> 0; };
const rand = () => { const r = Math.imul(_s[1],5), t = _s[1]<<9; _s[2]^=_s[0]; _s[3]^=_s[1]; _s[1]^=_s[2]; _s[0]^=_s[3]; _s[2]^=t; _s[3]=(_s[3]<<11)|(_s[3]>>>21); return (r>>>0)/4294967296; };
const gauss = (m,s) => m + s * Math.sqrt(-2*Math.log(rand())) * Math.cos(2*Math.PI*rand());
const shuffle = a => { for (let i = a.length-1; i > 0; i--) { const j = rand()*(i+1)|0; [a[i],a[j]]=[a[j],a[i]]; } };
const choices = (p,w) => { let t=0; for (const x of w) t+=x; let r=rand()*t; for (let i=0;i<p.length;i++) { r-=w[i]; if(r<=0) return p[i]; } return p.at(-1); };

seed(42);

// ── Data ─────────────────────────────────────────────────────────────────────
const INPUT = 'input.txt';
if (!existsSync(INPUT)) {
  console.log('Downloading names.txt …');
  const r = await fetch('https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt');
  writeFileSync(INPUT, await r.text());
}
const docs = readFileSync(INPUT,'utf-8').trim().split('\n').map(s=>s.trim()).filter(Boolean);
shuffle(docs);
console.log(`num docs: ${docs.length}`);

const chars = [...new Set(docs.join(''))].sort();
const BOS = chars.length, V = chars.length + 1;
console.log(`vocab size: ${V}`);

// ── Autograd ─────────────────────────────────────────────────────────────────
let _gen = 0;
class Value {
  constructor(d, ch=[], lg=[]) { this.data=d; this.grad=0; this._ch=ch; this._lg=lg; this._gen=0; }
  add(o) { o = o instanceof Value ? o : new Value(o); return new Value(this.data+o.data,[this,o],[1,1]); }
  mul(o) { o = o instanceof Value ? o : new Value(o); return new Value(this.data*o.data,[this,o],[o.data,this.data]); }
  pow(n) { return new Value(this.data**n,[this],[n*this.data**(n-1)]); }
  log()  { return new Value(Math.log(this.data),[this],[1/this.data]); }
  exp()  { const e=Math.exp(this.data); return new Value(e,[this],[e]); }
  relu() { return new Value(Math.max(0,this.data),[this],[+(this.data>0)]); }
  neg()  { return this.mul(-1); }
  sub(o) { return this.add(o instanceof Value ? o.neg() : -o); }
  div(o) { return this.mul(o instanceof Value ? o.pow(-1) : 1/o); }
  backward() {
    const g=++_gen, ord=[];
    (function tp(v){ if(v._gen===g) return; v._gen=g; for(const c of v._ch) tp(c); ord.push(v); })(this);
    this.grad=1;
    for (let i=ord.length-1; i>=0; i--) { const v=ord[i]; for (let j=0;j<v._ch.length;j++) v._ch[j].grad += v._lg[j]*v.grad; }
  }
}

// ── Model ────────────────────────────────────────────────────────────────────
const D=16, M=4, J=1, Y=8, H=D/M;
const mat = (r,c,std=.02) => Array.from({length:r}, ()=>Array.from({length:c}, ()=>new Value(gauss(0,std))));
const W = { wte:mat(V,D), wpe:mat(Y,D), lm_head:mat(V,D) };
for (let i=0; i<J; i++) {
  W[`l${i}.wq`]=mat(D,D); W[`l${i}.wk`]=mat(D,D); W[`l${i}.wv`]=mat(D,D);
  W[`l${i}.wo`]=mat(D,D,0); W[`l${i}.f1`]=mat(4*D,D); W[`l${i}.f2`]=mat(D,4*D,0);
}
const params = Object.values(W).flat(Infinity);
console.log(`num params: ${params.length}`);

// ── Ops ──────────────────────────────────────────────────────────────────────
const sum   = a => a.reduce((x,y) => x.add(y));
const linear = (x,w) => w.map(r => sum(r.map((v,i) => v.mul(x[i]))));
const softmax = a => { const m=Math.max(...a.map(v=>v.data)); const e=a.map(v=>v.sub(m).exp()); const s=sum(e); return e.map(v=>v.div(s)); };
const rmsnorm = a => { const s=sum(a.map(v=>v.mul(v))).mul(1/a.length).add(1e-5).pow(-.5); return a.map(v=>v.mul(s)); };

// ── Forward ──────────────────────────────────────────────────────────────────
function fwd(tok, pos, keys, vals) {
  let x = rmsnorm(W.wte[tok].map((v,i) => v.add(W.wpe[pos][i])));
  for (let l=0; l<J; l++) {
    let res=x; x=rmsnorm(x);
    const q=linear(x,W[`l${l}.wq`]), k=linear(x,W[`l${l}.wk`]), v0=linear(x,W[`l${l}.wv`]);
    keys[l].push(k); vals[l].push(v0);
    const heads=[];
    for (let h=0; h<M; h++) {
      const o=h*H, qh=q.slice(o,o+H);
      const kh=keys[l].map(k=>k.slice(o,o+H)), vh=vals[l].map(v=>v.slice(o,o+H));
      const attn=softmax(kh.map(k=>sum(qh.map((q,i)=>q.mul(k[i]))).mul(1/H**.5)));
      for (let i=0;i<H;i++) heads.push(sum(attn.map((a,j)=>a.mul(vh[j][i]))));
    }
    x=linear(heads,W[`l${l}.wo`]).map((v,i)=>v.add(res[i]));
    res=x; x=rmsnorm(x);
    x=linear(x,W[`l${l}.f1`]).map(v=>v.relu().pow(2));
    x=linear(x,W[`l${l}.f2`]).map((v,i)=>v.add(res[i]));
  }
  return linear(x, W.lm_head);
}

// ── Training ─────────────────────────────────────────────────────────────────
const lr=.01, b1=.9, b2=.95, eps=1e-8, steps=500;
const mb=new Float64Array(params.length), vb=new Float64Array(params.length);

for (let step=0; step<steps; step++) {
  const doc=docs[step%docs.length];
  const toks=[BOS,...Array.from(doc,c=>chars.indexOf(c)),BOS];
  const ctx=Math.min(Y,toks.length-1);
  const keys=Array.from({length:J},()=>[]), vals=Array.from({length:J},()=>[]);
  const losses=[];
  for (let t=0;t<ctx;t++) { const logits=softmax(fwd(toks[t],t,keys,vals)); losses.push(logits[toks[t+1]].log().neg()); }
  const loss=sum(losses).mul(1/ctx);
  loss.backward();
  const lr_t=lr*.5*(1+Math.cos(Math.PI*step/steps));
  for (let i=0;i<params.length;i++) {
    const p=params[i];
    mb[i]=b1*mb[i]+(1-b1)*p.grad; vb[i]=b2*vb[i]+(1-b2)*p.grad**2;
    p.data -= lr_t*(mb[i]/(1-b1**(step+1))) / ((vb[i]/(1-b2**(step+1)))**.5+eps);
    p.grad=0;
  }
  console.log(`step ${String(step+1).padStart(4)} / ${String(steps).padStart(4)} | loss ${loss.data.toFixed(4)}`);
}

// ── Inference ────────────────────────────────────────────────────────────────
const temp=.5, ids=Array.from({length:V},(_,i)=>i);
console.log('\n--- inference ---');
for (let s=0; s<20; s++) {
  const keys=Array.from({length:J},()=>[]), vals=Array.from({length:J},()=>[]);
  let tok=BOS; const name=[];
  for (let t=0;t<Y;t++) {
    tok=choices(ids, softmax(fwd(tok,t,keys,vals).map(v=>v.div(temp))).map(v=>v.data));
    if (tok===BOS) break;
    name.push(chars[tok]);
  }
  console.log(`sample ${String(s+1).padStart(2)}: ${name.join('')}`);
}
