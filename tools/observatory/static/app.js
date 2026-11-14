const API = '';
let state = { overview:null, leaderboard:[], l7:[], curvesRuns:[], replayList:[], bundle:null, episodes:[], currentEp:0, frameIdx:0, playing:false, speed:1, layers:{}, diffSel:[], sortCol:null, sortDir:1, hover:{} };

const $ = (s)=>document.querySelector(s);
const $$ = (s)=>Array.from(document.querySelectorAll(s));

function tab(name){
  $$('.tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===name));
  $$('.tabpane').forEach(p=>p.classList.toggle('active',p.id==='tab-'+name));
  requestAnimationFrame(()=>{
    if(name==='curves'){
      Object.keys(CHART_SPECS).forEach(id=>{const c=document.getElementById(id); if(c) renderChart(c,id);});
    } else if(name==='replay'){
      drawReplay();
      drawStrips();
    }
  });
}

document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>tab(t.dataset.tab)));

const LEVELS=['ALL','L0','L1','L2','L3','L4','L5','L_real','L_chaos','L6'];
const TRACKS=['ALL','student','teacher','baseline','gazebo_l6','l7','eval_json','archive'];
let levelFilter='ALL', trackFilter='ALL', paretoOnly=false, search='';

function buildFilters(){
  const lf=$('#level-filters'); lf.innerHTML='';
  LEVELS.forEach(l=>{const b=document.createElement('button');b.textContent=l;b.className=l===levelFilter?'active':'';b.onclick=()=>{levelFilter=l;buildFilters();renderLeaderboard();};lf.appendChild(b);});
  const tf=$('#track-filters'); tf.innerHTML='';
  TRACKS.forEach(t=>{const b=document.createElement('button');b.textContent=t;b.className=t===trackFilter?'active':'';b.onclick=()=>{trackFilter=t;buildFilters();renderLeaderboard();};tf.appendChild(b);});
}
$('#pareto-only').addEventListener('change',e=>{paretoOnly=e.target.checked;renderLeaderboard();});
$('#search-box').addEventListener('input',e=>{search=e.target.value.toLowerCase();renderLeaderboard();});

async function fetchJSON(url){try{const r=await fetch(url);if(!r.ok)throw new Error(r.status);return await r.json();}catch(e){return null;}}

async function loadOverview(){
  const d=await fetchJSON('/api/overview'); state.overview=d;
  if(d){
    const c=d.counts||{};
    $('#pill-eval').textContent=`Eval runs: ${c.evals??'—'} (archive ${c.archive??'—'}, gz ${c.gazebo_l6??'—'})`;
    $('#pill-train').textContent='Training logs: '+((c.logs??0)+(c.metrics??0));
    $('#pill-l7').textContent='L7 field runs: '+(c.l7_runs??'—');
    const pb=d.pi_bench;
    const ms=pb?(pb.step_ms_mean??pb.mean_ms??pb.latency_ms_single_core?.stateful_step_plus_shield?.p50??null):null;
    $('#pill-pi').textContent='Pi4B: '+(ms!=null?Number(ms).toFixed(3)+' ms/step':'n/a');
  }
}

async function loadLeaderboard(){
  const d=await fetchJSON('/api/leaderboard');
  state.leaderboard=Array.isArray(d)?d:(d?.entries||[]);
  state.l7=d?.l7_runs||[];
  renderLeaderboard();
  renderL7Table();
}

function sortRows(rows){
  if(!state.sortCol) return rows;
  const col=state.sortCol, dir=state.sortDir;
  const val=r=>{
    switch(col){
      case 'success': return (r.success??0)/Math.max(1,r.episodes??1);
      case 'time_median': return r.time_median??Infinity;
      case 'pos_p95': return r.pos_p95??Infinity;
      case 'yaw_p95_deg': return r.yaw_p95_deg??Infinity;
      case 'collisions': return r.collisions??0;
      case 'timeouts': return r.timeouts??0;
      default: return (r[col]??'').toString().toLowerCase();
    }
  };
  return [...rows].sort((a,b)=>{const va=val(a),vb=val(b);return (va<vb?-1:va>vb?1:0)*dir;});
}

function renderLeaderboard(){
  const tbody=$('#leaderboard-table tbody'); tbody.innerHTML='';
  let rows=state.leaderboard.filter(r=>{
    if(levelFilter!=='ALL' && r.level!==levelFilter) return false;
    if(trackFilter!=='ALL' && r.track!==trackFilter) return false;
    if(paretoOnly && !r.pareto) return false;
    if(search && !(r.run_id||'').toLowerCase().includes(search)) return false;
    return true;
  });
  rows=sortRows(rows);
  rows.forEach(r=>{
    const tr=document.createElement('tr');
    const hasRate=r.success_rate!=null;
    const lo=r.wilson_lo, hi=r.wilson_hi;
    const succCell=hasRate?`${r.success??0}/${r.episodes??0}`:'—';
    const ciCell=(lo!=null&&hi!=null)?`<span class="ci-bar"><span class="ci-fill" style="left:${lo*100}%;width:${Math.max(0,(hi-lo))*100}%"></span></span> [${lo.toFixed(2)},${hi.toFixed(2)}]`:'—';
    const posMm=r.pos_p95!=null?(r.pos_p95*1000).toFixed(1):'—';
    const yaw=r.yaw_p95_deg!=null?Number(r.yaw_p95_deg).toFixed(2):'—';
    const med=r.time_median!=null?Number(r.time_median).toFixed(2):'—';
    tr.innerHTML=`
      <td>${r.run_id||''}</td><td>${r.track||''}</td><td>${r.level||''}</td>
      <td>${succCell}</td>
      <td>${ciCell}</td>
      <td>${r.collisions??0}</td><td>${r.timeouts??0}</td><td>${med}</td><td>${posMm}</td><td>${yaw}</td>
      <td>${(r.badges||[]).map(b=>`<span class="badge ${String(b).toLowerCase()}">${b}</span>`).join('')}</td>
      <td>${r.parent_id||'—'}</td>
      <td><button class="diff-btn" data-id="${r.run_id}">Diff</button></td>`;
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll('.diff-btn').forEach(btn=>btn.addEventListener('click',()=>toggleDiff(btn.dataset.id)));
  // cards
  const cards=$('#level-cards'); cards.innerHTML='';
  ['L0','L1','L2','L3','L_real','L_chaos','L6','L7'].forEach(l=>{
    const subset=state.leaderboard.filter(r=>r.level===l);
    const best=subset.reduce((a,b)=>((b.success_rate??0)>(a?.success_rate??-1)?b:a), null);
    const c=document.createElement('div'); c.className='card';
    c.innerHTML=`<h4>${l}</h4><div class="v">${best?((best.success_rate??0)*100).toFixed(0)+'% succ':'—'}</div><div style="font-size:11px;color:var(--muted)">${best?best.run_id:''}</div>`;
    cards.appendChild(c);
  });
  renderDiffPanel();
}

$$('#leaderboard-table th[data-col]').forEach(th=>th.addEventListener('click',()=>{
  const col=th.dataset.col;
  if(state.sortCol===col){state.sortDir=-state.sortDir;}else{state.sortCol=col;state.sortDir=1;}
  renderLeaderboard();
}));

function toggleDiff(id){
  const i=state.diffSel.indexOf(id);
  if(i>=0) state.diffSel.splice(i,1);
  else { state.diffSel.push(id); if(state.diffSel.length>2) state.diffSel.shift(); }
  renderDiffPanel();
}

function diffMetric(r,k){return r?.[k];}

function renderDiffPanel(){
  const panel=$('#diff-panel'); if(!panel) return;
  if(state.diffSel.length<2){
    panel.hidden=true;
    panel.innerHTML='';
    return;
  }
  const [a,b]=state.diffSel.map(id=>state.leaderboard.find(r=>r.run_id===id));
  if(!a||!b){panel.hidden=true;return;}
  const rows=[
    ['Level', a.level, b.level],
    ['Track', a.track, b.track],
    ['Success rate', a.success_rate!=null?(a.success_rate*100).toFixed(1)+'%':'—', b.success_rate!=null?(b.success_rate*100).toFixed(1)+'%':'—'],
    ['Wilson 95% CI', a.wilson_lo!=null&&a.wilson_hi!=null?`[${a.wilson_lo.toFixed(2)}, ${a.wilson_hi.toFixed(2)}]`:'—', b.wilson_lo!=null&&b.wilson_hi!=null?`[${b.wilson_lo.toFixed(2)}, ${b.wilson_hi.toFixed(2)}]`:'—'],
    ['Episodes', a.episodes, b.episodes],
    ['Collisions', a.collisions, b.collisions],
    ['Timeouts', a.timeouts, b.timeouts],
    ['Med time (s)', a.time_median!=null?Number(a.time_median).toFixed(2):'—', b.time_median!=null?Number(b.time_median).toFixed(2):'—'],
    ['Pos p95 (mm)', a.pos_p95!=null?(a.pos_p95*1000).toFixed(1):'—', b.pos_p95!=null?(b.pos_p95*1000).toFixed(1):'—'],
    ['Yaw p95 (deg)', a.yaw_p95_deg!=null?Number(a.yaw_p95_deg).toFixed(2):'—', b.yaw_p95_deg!=null?Number(b.yaw_p95_deg).toFixed(2):'—'],
    ['Jerk mean', a.jerk_mean!=null?Number(a.jerk_mean).toFixed(3):'—', b.jerk_mean!=null?Number(b.jerk_mean).toFixed(3):'—'],
    ['Pareto score', a.pareto_score!=null?Number(a.pareto_score).toFixed(3):'—', b.pareto_score!=null?Number(b.pareto_score).toFixed(3):'—'],
  ];
  panel.hidden=false;
  panel.innerHTML=`<div class="diff-head"><strong>Diff:</strong> ${a.run_id} <span class="muted">vs</span> ${b.run_id}<button class="diff-close" id="diff-close">×</button></div>
    <table class="diff-table"><thead><tr><th>Metric</th><th>${a.run_id}</th><th>${b.run_id}</th></tr></thead>
    <tbody>${rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]??'—'}</td><td>${r[2]??'—'}</td></tr>`).join('')}</tbody></table>`;
  panel.querySelector('#diff-close').addEventListener('click',()=>{state.diffSel=[];renderDiffPanel();});
}

function renderL7Table(){
  const tbody=$('#l7-table tbody'); tbody.innerHTML='';
  (state.l7||[]).forEach(r=>{
    const tr=document.createElement('tr');
    const secs=r.seconds!=null?Number(r.seconds).toFixed(1):'—';
    tr.innerHTML=`<td>${r.run_id||r.id||''}</td><td>${r.mode||'—'}</td><td>${r.outcome||'—'}</td><td>${secs}</td><td>${r.hz!=null?Number(r.hz).toFixed(1):'—'}</td><td>${r.vision_ms_p50!=null?Number(r.vision_ms_p50).toFixed(1):'—'}</td><td>${r.loop_ms_p50!=null?Number(r.loop_ms_p50).toFixed(2):'—'}</td><td>${r.loop_ms_p95!=null?Number(r.loop_ms_p95).toFixed(2):'—'}</td><td><button class="open-replay" data-id="${r.run_id}">Open in Replay</button></td>`;
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll('.open-replay').forEach(btn=>btn.addEventListener('click',()=>{
    tab('replay');
    loadBundle('l7:'+btn.dataset.id);
  }));
}

// Curves
async function loadCurvesRuns(){
  const d=await fetchJSON('/api/curves/runs'); state.curvesRuns=Array.isArray(d)?d:(d?.runs||[]);
  const box=$('#curve-runs'); box.innerHTML='';
  const seen=new Set();
  state.curvesRuns.forEach(r=>{
    const id=r.run_id||r.id;
    const check=seen.size<3 && !seen.has(id) && (r.format!=='log' || state.curvesRuns.length<=3);
    if(check) seen.add(id);
    const lab=document.createElement('label');
    lab.innerHTML=`<input type="checkbox" value="${id}" ${check?'checked':''}> ${id} ${r.active?'<span class="badge">active</span>':''} ${r.format==='log'?'(legacy .log)':'(metrics.jsonl)'}`;
    lab.querySelector('input').addEventListener('change',()=>loadSelectedCurves());
    box.appendChild(lab);
  });
  loadSelectedCurves();
}

function selectedCurveRuns(){
  return Array.from(new Set($$('#curve-runs input[type=checkbox]:checked').map(i=>i.value)));
}

async function loadSelectedCurves(){
  const ids=selectedCurveRuns();
  if(!ids.length){drawCharts({});return;}
  const d=await fetchJSON('/api/curves?runs='+encodeURIComponent(ids.join(',')));
  drawCharts(d?.series||{});
}

const CHART_COLORS=['#4aa3ff','#3dd68c','#f5b041','#a569bd','#ff5c6c','#58d6d6'];
const CHART_SPECS={
  chart1:[{label:'succ_rate',keys:['succ_rate','succ']},{label:'coll_rate',keys:['coll_rate','coll']},{label:'timeout_rate',keys:['timeout_rate','timeout']},{label:'false_park',keys:['false_park']},{label:'dr_scale',keys:['dr_scale']}],
  chart2:[{label:'pol_loss',keys:['pol_loss','pol']},{label:'val_loss',keys:['val_loss','val']}],
  chart3:[{label:'aux_loss',keys:['aux_loss','aux']},{label:'dyn_loss',keys:['dyn_loss','dyn']},{label:'std',keys:['std']}],
  chart4:[{label:'std',keys:['std']},{label:'sps',keys:['sps']}],
};

function seriesValue(p,keys){for(const k of keys){const v=p?.[k];if(v!=null&&Number.isFinite(v))return v;}return null;}

function drawCharts(series){
  Object.keys(CHART_SPECS).forEach(id=>{
    const c=document.getElementById(id); if(!c) return;
    const spec=CHART_SPECS[id];
    const runs=Object.keys(series||{});
    // build flat line list
    const lines=[];
    runs.forEach((rid,ri)=>{
      const pts=series[rid]?.points||[];
      spec.forEach((s,si)=>{
        const ptsXY=[];
        pts.forEach(p=>{const v=seriesValue(p,s.keys);const it=p?.iter;if(v!=null&&it!=null)ptsXY.push({x:it,y:v});});
        if(ptsXY.length)lines.push({name:`${rid} · ${s.label}`,color:CHART_COLORS[(ri*spec.length+si)%CHART_COLORS.length],pts:ptsXY});
      });
    });
    state.hover[id]={lines,hoverIdx:null};
    renderChart(c,id);
  });
}

function renderChart(c,id){
  const ctx=c.getContext('2d');
  c.width=c.clientWidth; c.height=c.clientHeight;
  const W=c.width,H=c.height,padL=44,padR=8,padT=18,padB=20;
  ctx.fillStyle='#0a0d12'; ctx.fillRect(0,0,W,H);
  const {lines,hoverIdx}=state.hover[id]||{lines:[],hoverIdx:null};
  if(!lines.length){ctx.fillStyle='#7a8699';ctx.fillText('No data — select runs above',padL,H/2);return;}
  let yMin=Infinity,yMax=-Infinity,xMin=Infinity,xMax=-Infinity;
  lines.forEach(l=>l.pts.forEach(p=>{if(p.y<yMin)yMin=p.y;if(p.y>yMax)yMax=p.y;if(p.x<xMin)xMin=p.x;if(p.x>xMax)xMax=p.x;}));
  if(!Number.isFinite(yMin)){yMin=0;yMax=1;}
  if(yMax-yMin<1e-9){yMax=yMin+1;}
  const yPad=(yMax-yMin)*0.08; yMin-=yPad; yMax+=yPad;
  if(xMax-xMin<1e-9)xMax=xMin+1;
  const X=x=>padL+(x-xMin)/(xMax-xMin)*(W-padL-padR);
  const Y=y=>H-padB-(y-yMin)/(yMax-yMin)*(H-padT-padB);
  // grid + y labels
  ctx.font='10px monospace';
  for(let i=0;i<=4;i++){
    const yv=yMin+(yMax-yMin)*i/4, y=Y(yv);
    ctx.strokeStyle='#1c2530'; ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(W-padR,y); ctx.stroke();
    ctx.fillStyle='#7a8699'; ctx.fillText(yv.toFixed(3),2,y+3);
  }
  // x labels
  ctx.fillStyle='#7a8699';
  ctx.fillText(String(Math.round(xMin)),padL,H-6);
  ctx.fillText(String(Math.round(xMax)),W-padR-24,H-6);
  // lines
  lines.forEach(l=>{
    ctx.strokeStyle=l.color; ctx.lineWidth=1.5; ctx.beginPath();
    l.pts.forEach((p,i)=>{const x=X(p.x),y=Y(p.y); if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});
    ctx.stroke();
  });
  // legend
  let lx=padL+4;
  lines.forEach(l=>{
    ctx.fillStyle=l.color; ctx.fillRect(lx,4,8,8);
    ctx.fillStyle='#d7dde8'; const w=ctx.measureText(l.name).width; ctx.fillText(l.name,lx+11,12); lx+=w+22;
    if(lx>W-80){lx=padL+4;}
  });
  // hover tooltip
  if(hoverIdx!=null){
    const ref=lines[0].pts[hoverIdx];
    if(ref){
      const x=X(ref.x);
      ctx.strokeStyle='#7a8699'; ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(x,padT); ctx.lineTo(x,H-padB); ctx.stroke(); ctx.setLineDash([]);
      const linesTxt=[`iter ${Math.round(ref.x)}`].concat(lines.map(l=>{const p=l.pts[hoverIdx];return p?`${l.name}: ${p.y.toFixed(4)}`:null;}).filter(Boolean));
      const bw=Math.max(...linesTxt.map(t=>ctx.measureText(t).width))+14;
      const bh=linesTxt.length*14+8;
      let bx=x+8; if(bx+bw>W)bx=x-bw-8;
      ctx.fillStyle='rgba(18,24,33,0.95)'; ctx.fillRect(bx,padT,bw,bh);
      ctx.strokeStyle='#263040'; ctx.strokeRect(bx,padT,bw,bh);
      ctx.fillStyle='#d7dde8'; linesTxt.forEach((t,i)=>ctx.fillText(t,bx+7,padT+13+i*14));
    }
  }
}

function attachChartHover(){
  Object.keys(CHART_SPECS).forEach(id=>{
    const c=document.getElementById(id); if(!c) return;
    c.addEventListener('mousemove',e=>{
      const h=state.hover[id]; if(!h||!h.lines.length) return;
      const rect=c.getBoundingClientRect();
      const x=e.clientX-rect.left;
      const padL=44,padR=8;
      let xMin=Infinity,xMax=-Infinity;
      h.lines.forEach(l=>l.pts.forEach(p=>{if(p.x<xMin)xMin=p.x;if(p.x>xMax)xMax=p.x;}));
      const frac=(x-padL)/Math.max(1,(c.width-padL-padR));
      const target=xMin+frac*(xMax-xMin);
      let best=0,bestD=Infinity;
      h.lines[0].pts.forEach((p,i)=>{const d=Math.abs(p.x-target);if(d<bestD){bestD=d;best=i;}});
      h.hoverIdx=best;
      renderChart(c,id);
    });
    c.addEventListener('mouseleave',()=>{const h=state.hover[id]; if(h){h.hoverIdx=null;renderChart(c,id);}});
  });
}

// Replay
async function loadReplayList(){
  const d=await fetchJSON('/api/replay/list'); state.replayList=Array.isArray(d)?d:(d?.bundles||[]);
  const sel=$('#bundle-select'); sel.innerHTML='';
  state.replayList.forEach(b=>{const o=document.createElement('option');o.value=b.bundle_id;o.textContent=b.bundle_id;sel.appendChild(o);});
  sel.onchange=()=>loadBundle(sel.value);
  if(state.replayList.length){
    const pref=state.replayList.find(b=>b.bundle_id==='student_L2_cons_adr_s0_L1')
      || state.replayList.find(b=>!String(b.bundle_id).startsWith('obs_'))
      || state.replayList[0];
    loadBundle(pref.bundle_id);
  }
}

async function loadRolloutOptions(){
  const d=await fetchJSON('/api/rollout/options'); if(!d) return;
  const form=$('#rollout-form');
  const pol=form.querySelector('select[name=policy]');
  const lvl=form.querySelector('select[name=level]');
  if(pol){
    pol.innerHTML='';
    (d.policies||[]).forEach(p=>{const o=document.createElement('option');o.value=p;o.textContent=p;pol.appendChild(o);});
  }
  if(lvl){
    lvl.innerHTML='';
    (d.levels||[]).forEach(l=>{const o=document.createElement('option');o.value=l;o.textContent=l;lvl.appendChild(o);});
    if((d.levels||[]).includes('L1')) lvl.value='L1';
  }
}

async function loadBundle(id){
  const d=await fetchJSON('/api/replay/bundle/'+encodeURIComponent(id));
  state.bundle=d;
  state.episodes=d?.episodes||[];
  if(!state.episodes.length && Array.isArray(d?.poses)){
    state.episodes=[{episode_id:0,outcome:d?.summary?.outcome||'unknown',duration:0,steps:d.poses.length,frames:d.poses,end_frame_url:d?.end_frame_url||null}];
  }
  state.currentEp=0; state.frameIdx=0; state.playing=false; $('#btn-play').textContent='Play';
  const sel=$('#bundle-select'); if(sel && [...sel.options].some(o=>o.value===id)) sel.value=id;
  renderEpisodeList();
  loadEpisode();
}

function renderEpisodeList(){
  const filters=['All','Success','Collision','Timeout','False-Park'];
  const box=$('#ep-filters'); box.innerHTML='';
  filters.forEach(f=>{const b=document.createElement('button');b.textContent=f;b.onclick=()=>{box.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');filterEpisodes(f);};if(f==='All')b.classList.add('active');box.appendChild(b);});
  renderEpList(state.episodes);
}
function filterEpisodes(f){
  let eps=state.episodes;
  if(f!=='All') eps=eps.filter(e=> (e.outcome||'').toLowerCase().includes(f.toLowerCase().replace('-park','park')));
  renderEpList(eps);
}
function renderEpList(eps){
  const ul=$('#episode-list'); ul.innerHTML='';
  eps.forEach((e,i)=>{
    const li=document.createElement('li');
    li.textContent=`#${e.episode_id??i} ${e.outcome||''} dur=${e.duration??'—'} err=${e.final_error!=null?e.final_error+'mm':'—'}`;
    li.onclick=()=>{state.currentEp=state.episodes.indexOf(e);state.frameIdx=0;state.playing=false;$('#btn-play').textContent='Play';loadEpisode();};
    ul.appendChild(li);
  });
}

function poseOf(f){
  const p=f?.pose;
  if(!p) return null;
  if(Array.isArray(p)) return {x:p[0]??0,y:p[1]??0,yaw:p[2]??0};
  return {x:p.x??0,y:p.y??0,yaw:p.yaw??0};
}
function tagOf(f){
  const p=f?.tag; if(!p) return null;
  if(Array.isArray(p)) return {x:p[0]??0,y:p[1]??0,yaw:p[2]??0};
  return {x:p.x??0,y:p.y??0,yaw:p.yaw??0};
}
function auxOf(f){
  const p=f?.aux_pose; if(!p) return poseOf(f);
  if(Array.isArray(p)) return {x:p[0]??0,y:p[1]??0,yaw:p[2]??0};
  return {x:p.x??0,y:p.y??0,yaw:p.yaw??0};
}

function loadEpisode(){
  const ep=state.episodes[state.currentEp];
  const scrubber=$('#scrubber');
  if(!ep){
    $('#time-readout').textContent='t=0.00s step 0';
    scrubber.max=0; scrubber.value=0;
    drawReplay(); drawStrips();
    $('#endframe-card').hidden=true;
    return;
  }
  const frames=ep.frames||[];
  state.frameIdx=Math.max(0,Math.min(state.frameIdx,Math.max(0,frames.length-1)));
  const f=frames[state.frameIdx]||{};
  $('#time-readout').textContent=`t=${(f.t??0).toFixed(2)}s step ${state.frameIdx}/${Math.max(0,frames.length-1)}`;
  scrubber.max=Math.max(0,frames.length-1);
  scrubber.value=state.frameIdx;
  drawReplay();
  drawStrips();
  const card=$('#endframe-card');
  const ef=ep.end_frame_url;
  if(ef){card.hidden=false;card.innerHTML=`<img src="${ef}" alt="end frame"><div style="font-size:11px;color:var(--muted)">${ef}</div>`;}else{card.hidden=true;}
}

const LAYERS=['True Pose','Noisy/Delayed Tag','Aux/Est Pose','Camera FOV Cone','Blind Ring','Swept-Clear Shield','Path Trail','ArUco Tags'];
function buildLayers(){
  const box=$('#layers'); box.innerHTML='';
  LAYERS.forEach(l=>{state.layers[l]=true;const lab=document.createElement('label');lab.innerHTML=`<input type="checkbox" checked> ${l}`;lab.querySelector('input').onchange=e=>{state.layers[l]=e.target.checked;drawReplay();};box.appendChild(lab);});
}

function drawCapsule(ctx,x1,y1,x2,y2,rPx){
  ctx.beginPath();
  ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
  ctx.lineCap='round'; ctx.lineWidth=rPx*2; ctx.stroke();
  ctx.lineCap='butt';
}

function drawReplay(){
  const c=$('#replay-canvas'); const ctx=c.getContext('2d');
  c.width=c.clientWidth; c.height=c.clientHeight;
  ctx.fillStyle='#0a0d12'; ctx.fillRect(0,0,c.width,c.height);
  const geo=state.bundle?.geometry||{};
  const pxPerMeter=Math.min(c.width,c.height)/2.6;
  const cx=c.width/2, cy=c.height/2;
  const W2S=(x,y)=>[cx+x*pxPerMeter, cy-y*pxPerMeter];
  // grid every 0.25 m
  const step=0.25;
  const xHalf=c.width/2/pxPerMeter, yHalf=c.height/2/pxPerMeter;
  ctx.font='9px monospace';
  for(let gx=Math.floor(-xHalf/step)*step; gx<=xHalf+1e-9; gx+=step){
    const [sx]=W2S(gx,0);
    const major=Math.abs(gx%1)<1e-9;
    ctx.strokeStyle=major?'#2c3848':'#1a2230'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(sx,0); ctx.lineTo(sx,c.height); ctx.stroke();
    if(major){ctx.fillStyle='#7a8699'; ctx.fillText(gx.toFixed(1)+'m', sx+2, c.height-4);}
  }
  for(let gy=Math.floor(-yHalf/step)*step; gy<=yHalf+1e-9; gy+=step){
    const [,sy]=W2S(0,gy);
    const major=Math.abs(gy%1)<1e-9;
    ctx.strokeStyle=major?'#2c3848':'#1a2230';
    ctx.beginPath(); ctx.moveTo(0,sy); ctx.lineTo(c.width,sy); ctx.stroke();
    if(major){ctx.fillStyle='#7a8699'; ctx.fillText(gy.toFixed(1)+'m', 2, sy-2);}
  }
  // axes through origin
  ctx.strokeStyle='#33404f'; ctx.beginPath(); ctx.moveTo(0,cy); ctx.lineTo(c.width,cy); ctx.moveTo(cx,0); ctx.lineTo(cx,c.height); ctx.stroke();

  const L=state.layers;
  const ep=state.episodes[state.currentEp];
  const frames=ep?.frames||[];
  const f=frames[state.frameIdx]||{};
  const pose=poseOf(f);

  // Blind Ring (r = 0.40 m, dashed)
  if(L['Blind Ring']){
    const [bx,by]=W2S(0,0);
    ctx.strokeStyle='#f5b041'; ctx.setLineDash([6,5]); ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(bx,by,0.40*pxPerMeter,0,Math.PI*2); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle='#f5b041'; ctx.fillText('blind ring 0.40m', bx+0.40*pxPerMeter*0.72, by-0.40*pxPerMeter*0.72);
  }

  // Goal Box (±0.03 × ±0.05 m) + heading arrow
  {
    const [gx,gy]=W2S(0,0);
    const gw=0.06*pxPerMeter, gh=0.10*pxPerMeter;
    ctx.strokeStyle='#3dd68c'; ctx.lineWidth=2; ctx.strokeRect(gx-gw/2, gy-gh/2, gw, gh);
    // heading arrow (goal faces -y)
    ctx.beginPath(); ctx.moveTo(gx,gy); ctx.lineTo(gx,gy+0.10*pxPerMeter); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gx,gy+0.10*pxPerMeter); ctx.lineTo(gx-4,gy+0.10*pxPerMeter-8); ctx.moveTo(gx,gy+0.10*pxPerMeter); ctx.lineTo(gx+4,gy+0.10*pxPerMeter-8); ctx.stroke();
    ctx.fillStyle='#3dd68c'; ctx.fillText('goal', gx+6, gy-6);
  }

  // ArUco Tags
  if(L['ArUco Tags']){
    const tags=geo.tags||[{id:14,x:-0.028,y:0},{id:6,x:0.06,y:-0.05},{id:3,x:-0.105,y:-0.049}];
    tags.forEach(t=>{
      const [tx,ty]=W2S(t.x,t.y);
      const s=0.05*pxPerMeter;
      ctx.fillStyle='#22d3ee'; ctx.fillRect(tx-s/2,ty-s/2,s,s);
      ctx.strokeStyle='#0e7490'; ctx.strokeRect(tx-s/2,ty-s/2,s,s);
      ctx.fillStyle='#041318'; ctx.font='bold 9px monospace'; ctx.fillText(String(t.id), tx-s/2+2, ty+3); ctx.font='9px monospace';
    });
  }

  // True-Scale Drone Skids: capsules at x=±0.157, y∈[-0.1524,+0.1524], r=0.03
  {
    const skidX=geo.skid_x??0.157, halfLen=(geo.gear_len??0.3048)/2, r=geo.gear_r??0.03;
    [-skidX, skidX].forEach(sx=>{
      const [x1,y1]=W2S(sx,-halfLen), [x2,y2]=W2S(sx,halfLen);
      ctx.strokeStyle='#4aa3ff'; ctx.globalAlpha=0.9;
      drawCapsule(ctx,x1,y1,x2,y2,r*pxPerMeter);
      ctx.globalAlpha=1;
    });
    // Swept-Clear Shield: inflated capsule r=0.045, dashed amber
    if(L['Swept-Clear Shield']){
      const rInf=(geo.gear_r??0.03)+(geo.inflation??0.015);
      [-skidX, skidX].forEach(sx=>{
        const [x1,y1]=W2S(sx,-halfLen), [x2,y2]=W2S(sx,halfLen);
        ctx.strokeStyle='#f5b041'; ctx.setLineDash([5,4]); ctx.lineWidth=1.5;
        drawCapsule(ctx,x1,y1,x2,y2,rInf*pxPerMeter);
        ctx.setLineDash([]);
      });
    }
  }

  // Path Trail
  if(L['Path Trail'] && frames.length){
    ctx.strokeStyle='#58d6d6'; ctx.lineWidth=1.5; ctx.globalAlpha=0.9; ctx.beginPath();
    let started=false;
    for(let i=0;i<=state.frameIdx && i<frames.length;i++){
      const p=poseOf(frames[i]); if(!p) continue;
      const [sx,sy]=W2S(p.x,p.y);
      if(!started){ctx.moveTo(sx,sy);started=true;}else ctx.lineTo(sx,sy);
    }
    ctx.stroke(); ctx.globalAlpha=1;
  }

  // Camera FOV Cone (±33.5°)
  if(L['Camera FOV Cone'] && pose){
    const half=((geo.fov_half_deg??33.5)*Math.PI/180);
    const range=1.4*pxPerMeter;
    const [ux,uy]=W2S(pose.x,pose.y);
    // heading: yaw CCW positive in policy frame; screen y is flipped.
    // world -y (car facing at yaw=0) is screen-down, i.e. canvas angle +PI/2.
    const ang=Math.PI/2 - pose.yaw;
    ctx.fillStyle='rgba(74,163,255,0.14)';
    ctx.beginPath(); ctx.moveTo(ux,uy);
    ctx.arc(ux,uy,range,ang-half,ang+half);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle='rgba(74,163,255,0.4)'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(ux,uy); ctx.arc(ux,uy,range,ang-half,ang+half); ctx.closePath(); ctx.stroke();
  }

  // Noisy/Delayed Tag pose ghost (dashed amber)
  const tag=tagOf(f);
  if(L['Noisy/Delayed Tag'] && tag){
    const [tx,ty]=W2S(tag.x,tag.y);
    ctx.save(); ctx.translate(tx,ty); ctx.rotate(-tag.yaw);
    ctx.strokeStyle='#f5b041'; ctx.setLineDash([4,4]); ctx.lineWidth=1.5;
    ctx.strokeRect(-0.08*pxPerMeter,-0.11*pxPerMeter,0.16*pxPerMeter,0.22*pxPerMeter);
    ctx.setLineDash([]); ctx.restore();
    ctx.fillStyle='#f5b041'; ctx.fillText('tag', tx+6, ty+12);
  }

  // Aux/Est Pose ghost (dotted purple)
  const aux=auxOf(f);
  if(L['Aux/Est Pose'] && aux){
    const [ax,ay]=W2S(aux.x,aux.y);
    ctx.save(); ctx.translate(ax,ay); ctx.rotate(-aux.yaw);
    ctx.strokeStyle='#a569bd'; ctx.setLineDash([2,3]); ctx.lineWidth=1.5;
    ctx.strokeRect(-0.08*pxPerMeter,-0.11*pxPerMeter,0.16*pxPerMeter,0.22*pxPerMeter);
    ctx.setLineDash([]); ctx.restore();
    ctx.fillStyle='#a569bd'; ctx.fillText('aux', ax+6, ay-6);
  }

  // True/Est UGV Body (0.16 × 0.22 m) + wheels + heading arrow
  if(L['True Pose'] && pose){
    const [ux,uy]=W2S(pose.x,pose.y);
    const blocked=!!f.shield_blocked;
    const body=blocked?'#ff5c6c':'#3dd68c';
    ctx.save(); ctx.translate(ux,uy); ctx.rotate(-pose.yaw);
    const bw=0.16*pxPerMeter, bl=0.22*pxPerMeter;
    // wheels
    ctx.fillStyle='#0a0d12';
    const ww=0.03*pxPerMeter, wl=0.06*pxPerMeter;
    [[-bw/2-ww/2,-bl/2+2],[bw/2-ww/2,-bl/2+2],[-bw/2-ww/2,bl/2-wl-2],[bw/2-ww/2,bl/2-wl-2]].forEach(([wx,wy])=>ctx.fillRect(wx,wy,ww,wl));
    // body
    ctx.fillStyle=body; ctx.globalAlpha=0.25; ctx.fillRect(-bw/2,-bl/2,bw,bl); ctx.globalAlpha=1;
    ctx.strokeStyle=body; ctx.lineWidth=2; ctx.strokeRect(-bw/2,-bl/2,bw,bl);
    // heading arrow (car faces -y at yaw=0; local +y is screen-down)
    const tipY=bl/2+6;
    ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(0,tipY); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,tipY); ctx.lineTo(-5,tipY-8); ctx.moveTo(0,tipY); ctx.lineTo(5,tipY-8); ctx.stroke();
    ctx.restore();
    ctx.fillStyle=body; ctx.fillText(blocked?'BLOCKED':'ugv', ux+8, uy);
  }

  if(!pose){ctx.fillStyle='#7a8699'; ctx.fillText('No pose in this frame', cx-50, cy-10);}
  ctx.fillStyle='#d7dde8'; ctx.fillText(`t=${(f.t??0).toFixed(2)}  frame ${state.frameIdx}/${Math.max(0,frames.length-1)}`, 10, 16);
}

function drawStrips(){
  const ep=state.episodes[state.currentEp];
  const frames=ep?.frames||[];
  const n=frames.length;
  const playFrac=n>0?state.frameIdx/n:0;
  ['strip1','strip2','strip3'].forEach(id=>{
    const c=document.getElementById(id); if(!c) return;
    const ctx=c.getContext('2d');
    c.width=c.clientWidth; c.height=c.clientHeight;
    ctx.fillStyle='#0a0d12'; ctx.fillRect(0,0,c.width,c.height);
    if(!n){
      ctx.fillStyle='#7a8699'; ctx.fillText('No frames',10,c.height/2);
      return;
    }
    const W=c.width,H=c.height,padT=10,padB=4;
    const X=i=>(i/Math.max(1,n))*W;
    const BW=Math.max(1,W/n);
    if(id==='strip1'){
      // detection bars: valid green / blackout red / outlier amber / none gray
      let maxAge=1;
      frames.forEach(f=>{const a=Number(f.tag_age)||0; if(a>maxAge)maxAge=a;});
      frames.forEach((f,i)=>{
        let col='#263040';
        if(f.outlier) col='#f5b041';
        else if(f.blackout) col='#ff5c6c';
        else if(f.valid) col='#3dd68c';
        ctx.fillStyle=col;
        ctx.fillRect(X(i),2,BW,H-14);
      });
      // tag_age line
      ctx.strokeStyle='#f5b041'; ctx.lineWidth=1; ctx.beginPath();
      frames.forEach((f,i)=>{const a=Math.min(1,(Number(f.tag_age)||0)/maxAge); const y=H-12-a*(H-22); const x=X(i)+BW/2; if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});
      ctx.stroke();
      ctx.fillStyle='#7a8699'; ctx.fillText('valid/blackout/outlier · tag_age',4,H-2);
    } else if(id==='strip2'){
      let m=1e-6;
      frames.forEach(f=>{m=Math.max(m,Math.abs(Number(f.imu_w)||0),Math.abs(Number(f.imu_yaw)||0));});
      ctx.strokeStyle='#263040'; ctx.beginPath(); ctx.moveTo(0,(H)/2); ctx.lineTo(W,(H)/2); ctx.stroke();
      const trace=(key,col)=>{
        ctx.strokeStyle=col; ctx.lineWidth=1; ctx.beginPath();
        frames.forEach((f,i)=>{const v=(Number(f[key])||0)/m; const y=H/2-v*(H/2-padT); const x=X(i)+BW/2; if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});
        ctx.stroke();
      };
      trace('imu_w','#4aa3ff'); trace('imu_yaw','#a569bd');
      ctx.fillStyle='#4aa3ff'; ctx.fillText('imu_w',4,10);
      ctx.fillStyle='#a569bd'; ctx.fillText('imu_yaw',44,10);
    } else if(id==='strip3'){
      let m=1e-6;
      frames.forEach(f=>{const a=f.act||[0,0,0]; m=Math.max(m,Math.abs(a[0]||0),Math.abs(a[1]||0),Math.abs(a[2]||0));});
      ctx.strokeStyle='#263040'; ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();
      // amber highlight where |w| > 0.8
      frames.forEach((f,i)=>{const w=(f.act||[0,0,0])[2]||0; if(Math.abs(w)>0.8){ctx.fillStyle='rgba(245,176,65,0.18)'; ctx.fillRect(X(i),0,BW,H);}});
      const trace=(idx,col)=>{
        ctx.strokeStyle=col; ctx.lineWidth=1; ctx.beginPath();
        frames.forEach((f,i)=>{const a=f.act||[0,0,0]; const v=(a[idx]||0)/m; const y=H/2-v*(H/2-padT); const x=X(i)+BW/2; if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});
        ctx.stroke();
      };
      trace(0,'#3dd68c'); trace(1,'#58d6d6'); trace(2,'#f5b041');
      // red markers where shield_blocked
      frames.forEach((f,i)=>{if(f.shield_blocked){ctx.fillStyle='#ff5c6c'; ctx.fillRect(X(i),0,BW,4);}});
      ctx.fillStyle='#3dd68c'; ctx.fillText('vx',4,10);
      ctx.fillStyle='#58d6d6'; ctx.fillText('vy',26,10);
      ctx.fillStyle='#f5b041'; ctx.fillText('w',48,10);
    }
    // playhead
    const px=playFrac*W;
    ctx.strokeStyle='#d7dde8'; ctx.lineWidth=1; ctx.setLineDash([2,2]);
    ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,H); ctx.stroke(); ctx.setLineDash([]);
  });
}

function scrubFromEvent(e){
  const c=e.currentTarget; const frames=state.episodes[state.currentEp]?.frames||[];
  if(!frames.length) return;
  const rect=c.getBoundingClientRect();
  const frac=Math.max(0,Math.min(1,(e.clientX-rect.left)/Math.max(1,rect.width)));
  state.frameIdx=Math.round(frac*(frames.length-1));
  loadEpisode();
}

// transport
$('#btn-play').onclick=()=>{state.playing=!state.playing;$('#btn-play').textContent=state.playing?'Pause':'Play';};
$('#btn-step-back').onclick=()=>{if(state.frameIdx>0)state.frameIdx--;loadEpisode();};
$('#btn-step-fwd').onclick=()=>{const ep=state.episodes[state.currentEp]; if(ep && state.frameIdx<(ep.frames||[]).length-1)state.frameIdx++;loadEpisode();};
$('#speed-sel').onchange=e=>{state.speed=parseFloat(e.target.value)||1;};
$('#scrubber').oninput=e=>{state.frameIdx=parseInt(e.target.value)||0;loadEpisode();};

// playback loop
let lastTs=null, playAcc=0;
function playbackLoop(ts){
  requestAnimationFrame(playbackLoop);
  if(!state.playing){lastTs=null;playAcc=0;return;}
  const ep=state.episodes[state.currentEp]; if(!ep){state.playing=false;return;}
  const frames=ep.frames||[]; if(!frames.length){state.playing=false;return;}
  if(lastTs==null){lastTs=ts;return;}
  const dt=(ts-lastTs)/1000; lastTs=ts;
  playAcc+=dt*state.speed;
  let advanced=false;
  while(state.frameIdx<frames.length-1){
    const curT=frames[state.frameIdx]?.t ?? state.frameIdx*0.02;
    const nxtT=frames[state.frameIdx+1]?.t ?? curT+0.02;
    const step=Math.max(1e-3,nxtT-curT);
    if(playAcc>=step){playAcc-=step;state.frameIdx++;advanced=true;}else break;
  }
  if(state.frameIdx>=frames.length-1){state.playing=false;$('#btn-play').textContent='Play';}
  if(advanced) loadEpisode();
}
requestAnimationFrame(playbackLoop);

// contract
async function loadContractRuns(){
  const d=await fetchJSON('/api/contract/runs'); const runs=Array.isArray(d)?d:(d?.runs||[]);
  const sel=$('#contract-run'); sel.innerHTML='';
  runs.forEach(r=>{const id=typeof r==='string'?r:(r.id||r.run_id); const o=document.createElement('option');o.value=id;o.textContent=id;sel.appendChild(o);});
  sel.onchange=()=>loadContract(sel.value);
  if(runs.length){const first=runs[0]; loadContract(typeof first==='string'?first:(first.id||first.run_id));}
}
async function loadContract(id){
  const d=await fetchJSON('/api/contract?run_id='+encodeURIComponent(id));
  if(!d) return;
  const banner=$('#contract-banner');
  banner.textContent=d.status||'—';
  banner.className='banner '+(String(d.status).toLowerCase().includes('pass')?'pass':'fail');
  const ss=$('#contract-sim-source');
  if(ss){
    const src=d.sim_source||'unknown';
    ss.textContent=src==='nominal_contract'
      ? 'sim reference: nominal contract (sim reference stats unavailable — nominal constants used)'
      : 'sim reference: '+src;
  }
  const checks=$('#contract-checks'); checks.innerHTML='';
  (d.checks||[]).forEach(ch=>{
    const div=document.createElement('div'); div.className='check '+(ch.pass?'pass':'fail');
    div.innerHTML=`<div class="name">${ch.name}</div><div class="val">${ch.measured} / ${ch.limit}</div>`;
    checks.appendChild(div);
  });
  const chEl=$('#contract-channels'); chEl.innerHTML='';
  (d.channels||[]).forEach(c=>{
    const row=document.createElement('div'); row.className='channel-row';
    const rm=Number(c.real_mean), sm=Number(c.sim_mean);
    const delta=Number.isFinite(rm)&&Number.isFinite(sm)?Math.abs(rm-sm):0;
    const scale=Math.max(Math.abs(rm||0),Math.abs(sm||0),1e-6);
    const pct=Math.min(100,(delta/scale)*100);
    const fmt=v=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toPrecision(4);
    row.innerHTML=`<div>${c.name}</div><div>real μ=${fmt(c.real_mean)} σ=${fmt(c.real_std)}</div><div>sim μ=${fmt(c.sim_mean)} σ=${fmt(c.sim_std)}</div><div><div class="delta-bar"><div class="delta-fill" style="width:${pct}%"></div></div></div>`;
    chEl.appendChild(row);
  });
}

// rollout
$('#rollout-form').addEventListener('submit',async e=>{
  e.preventDefault();
  const fd=new FormData(e.target);
  const body={policy:fd.get('policy'),level:fd.get('level'),episodes:parseInt(fd.get('episodes')),seed:parseInt(fd.get('seed'))};
  const r=await fetch('/api/rollout',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json(); const jobId=j.job_id||j.id;
  $('#rollout-status').textContent='job '+jobId;
  const poll=setInterval(async()=>{
    const s=await fetchJSON('/api/rollout/'+jobId);
    $('#rollout-status').textContent='job '+jobId+' : '+(s?.status||'');
    if(s?.status==='completed'||s?.status==='done'||s?.status==='failed'){clearInterval(poll); if(s?.bundle_id) loadBundle(s.bundle_id);}
  },2000);
});

// init
async function init(){
  buildFilters();
  buildLayers();
  attachChartHover();
  ['strip1','strip2','strip3'].forEach(id=>{const c=document.getElementById(id); if(c)c.addEventListener('click',scrubFromEvent);});
  await loadOverview();
  await loadLeaderboard();
  await loadCurvesRuns();
  await loadRolloutOptions();
  await loadReplayList();
  await loadContractRuns();
  $('#bundle-select').addEventListener('change',e=>loadBundle(e.target.value));
  $('#auto-refresh').addEventListener('change',e=>{
    if(e.target.checked){ state._iv=setInterval(()=>{loadOverview();loadLeaderboard();},5000);} else {clearInterval(state._iv);}
  });
  window.addEventListener('resize',()=>{drawReplay();drawStrips();});
}
init();
