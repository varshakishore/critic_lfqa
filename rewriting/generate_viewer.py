import glob, json, re, os

# Paths are relative to this script, so `python3 generate_viewer.py` works from
# the project dir regardless of cwd. Override with env vars if needed.
HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.environ.get("VIEWER_DIR", os.path.join(HERE, "test_samples"))
OUT = os.environ.get("VIEWER_OUT", os.path.join(HERE, "trace_viewer.html"))

# Friendly labels for known files. Any file not listed gets an auto-derived
# label from its filename suffix, so new runs show up without editing this.
LABELS = {
    'drtulu_answers_w_critiques_rewritten.jsonl':                       'gpt-5.4 (orig)',
    'drtulu_answers_w_critiques_rewritten_promptfix.jsonl':             'gpt-5.4 (legacy)',
    'drtulu_answers_w_critiques_rewritten_glm.jsonl':                   'GLM (legacy r2)',
    'drtulu_answers_w_critiques_rewritten_glm_run1.jsonl':              'GLM (legacy r1)',
    'drtulu_answers_w_critiques_rewritten_gpt-5.6-luna.jsonl':          'luna (legacy)',
    'drtulu_answers_w_critiques_rewritten_gpt-5.6-luna_v1.jsonl':       'luna (v1 prog)',
    'drtulu_answers_w_critiques_rewritten_zai-org-GLM-5.2-FP8_v1.jsonl':'GLM-5.2 (v1 prog)',
    'drtulu_answers_w_critiques_rewritten_gpt-5.6-luna_v2.jsonl':       'luna (v2 fixed)',
    'drtulu_answers_w_critiques_rewritten_zai-org-GLM-5.2-FP8_v2.jsonl':'GLM-5.2 (v2 fixed)',
    'drtulu_answers_w_critiques_rewritten_gpt-5.4_v2.jsonl':            'gpt-5.4 (v2 fixed)',
}


def auto_label(basename):
    """Derive a readable label from a filename's tag when not in LABELS."""
    tag = basename.replace('drtulu_answers_w_critiques_rewritten', '').lstrip('_').replace('.jsonl', '')
    return tag or 'gpt-5.4 (orig)'


records, order, models = {}, [], []
for fn in sorted(glob.glob(DIR + '/drtulu_answers_w_critiques_rewritten*.jsonl')):
    lbl = LABELS.get(os.path.basename(fn)) or auto_label(os.path.basename(fn))
    if lbl not in models:
        models.append(lbl)
    for line in open(fn):
        if not line.strip():
            continue
        r = json.loads(line)
        q = r['question']
        if q not in records:
            records[q] = {'question': q, 'critique': r.get('critique', '[]'),
                          'original_trace': r.get('original_trace', ''),
                          'original_answer': r.get('original_answer', ''),
                          'models': {}}
            order.append(q)
        records[q]['models'][lbl] = {'rewritten_trace': r.get('rewritten_trace', ''),
                                     'rewritten': r.get('rewritten', '')}

data = [records[q] for q in order]
data_json = json.dumps(data).replace("<", "\\u003c")
models_json = json.dumps(models)

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rewrite Model Comparison</title>
<style>
:root{--bg:#f7f7f8;--panel:#fff;--ink:#1a1a1a;--muted:#6b7280;--line:#e5e7eb;--think:#eef2ff;--think-b:#c7d2fe;--search:#ecfdf5;--search-b:#a7f3d0;--tool:#fffbeb;--tool-b:#fde68a;--answer:#f8fafc;--new:#16a34a;--cite:#2563eb;--del:#fee2e2;--delink:#991b1b;--ins:#dcfce7;--insink:#166534;--chip:#eef2ff;}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--panel:#171a21;--ink:#e6e6e6;--muted:#9aa4b2;--line:#2a2f3a;--think:#1b1f3a;--think-b:#3b4270;--search:#0f2a1e;--search-b:#245c41;--tool:#2a2411;--tool-b:#5c4c1a;--answer:#12161d;--new:#34d399;--cite:#7dd3fc;--del:#3a1a1d;--delink:#fca5a5;--ins:#132a1c;--insink:#86efac;--chip:#1b1f3a;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:15px;margin:0 0 8px}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;font-size:13px}
select{padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);max-width:64ch}
label{color:var(--muted);display:flex;gap:6px;align-items:center}
.pill{font-weight:600;padding:2px 8px;border-radius:6px;border:1px solid var(--line)}
.pill.a{background:rgba(37,99,235,.12)} .pill.b{background:rgba(22,163,74,.12)}
main{padding:14px 16px;max-width:1700px;margin:0 auto}
.question{font-size:15px;font-weight:600;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:12px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:18px 0 8px}
details.crits>summary{cursor:pointer;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:10px 0}
.crit{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--search-b);border-radius:8px;padding:8px 10px;margin-bottom:6px}
.kv{display:grid;grid-template-columns:150px 1fr;gap:10px;padding:1px 0;align-items:start}
.kv .k{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--muted);text-align:right}
.kv .v{white-space:pre-wrap;word-break:break-word;font-size:12.5px}
.kv .v.span{font-family:ui-monospace,Menlo,monospace;font-size:11.5px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media (max-width:960px){.cols{grid-template-columns:1fr}}
.col{min-width:0}
.colhead{position:sticky;top:70px;background:var(--bg);padding:6px 0;z-index:2}
.colhead .name{font-weight:700;font-size:13px}
.colhead .stats{font-size:11px;color:var(--muted);font-family:ui-monospace,Menlo,monospace}
.colhead .bad{color:var(--delink);font-weight:700}
.block{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin-bottom:7px;overflow-x:auto}
.block .lbl{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-bottom:4px;display:flex;gap:8px;align-items:center}
.block.think{background:var(--think);border-color:var(--think-b)}
.block.call{background:var(--search);border-color:var(--search-b)}
.block.tool{background:var(--tool);border-color:var(--tool-b)}
.block.answerblk{background:var(--answer)}
.block .body{white-space:pre-wrap;font-size:12.5px}
.block.call .body{font-family:ui-monospace,Menlo,monospace}
.isnew{outline:2px solid var(--new);outline-offset:1px}
.newtag{color:var(--new);border:1px solid var(--new);border-radius:999px;font-size:10px;padding:0 6px}
.reftag{color:#7c3aed;border:1px solid #7c3aed;border-radius:999px;font-size:10px;padding:0 6px}
.baretag{color:#b45309;border:1px solid #f59e0b;border-radius:999px;font-size:10px;padding:0 6px}
.block.think.bare{border-style:dashed;border-width:2px}
details.snips>summary{cursor:pointer;font-size:12px;color:var(--muted)}
.snip{border-top:1px dashed var(--line);padding:5px 0;margin-top:5px}
.snip .sid{font-family:ui-monospace,monospace;font-size:11px;color:var(--cite)}
.snip .st{font-weight:600;font-size:12px}.snip .sx{font-size:12px;white-space:pre-wrap}
.answer-body{white-space:pre-wrap;font-size:13px;line-height:1.55}
.cited{background:rgba(37,99,235,.07);border-bottom:1px solid var(--cite)}
.citeid{color:var(--cite);font-size:10px;margin-left:1px}
del{background:var(--del);color:var(--delink);text-decoration:line-through;border-radius:3px}
ins{background:var(--ins);color:var(--insink);text-decoration:none;border-radius:3px}
.empty{color:var(--muted);font-style:italic}
.viewtoggle{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden;margin-left:6px}
.viewtoggle button{border:0;background:var(--panel);color:var(--muted);padding:3px 10px;cursor:pointer;font:inherit;font-size:12px}
.viewtoggle button.on{background:var(--cite);color:#fff}
</style></head>
<body>
<header>
  <h1>Rewrite model comparison</h1>
  <div class="controls">
    <label>Record <select id="recsel"></select></label>
    <label><span class="pill a">A</span> <select id="mA"></select></label>
    <label><span class="pill b">B</span> <select id="mB"></select></label>
    <label><input type="checkbox" id="hlnew" checked> flag inserted rounds</label>
  </div>
</header>
<main>
  <div class="question" id="question"></div>
  <details class="crits"><summary>Critiques (shared)</summary><div id="critiques"></div></details>

  <h2>Trace</h2>
  <div class="cols"><div class="col"><div class="colhead"><div class="name" id="tAname"></div><div class="stats" id="tAstats"></div></div><div id="tA"></div></div>
                    <div class="col"><div class="colhead"><div class="name" id="tBname"></div><div class="stats" id="tBstats"></div></div><div id="tB"></div></div></div>

  <h2>Answer <span id="ansmode" class="viewtoggle"><button data-m="side" class="on">Side by side</button><button data-m="diff">Word diff A→B</button></span></h2>
  <div class="cols" id="ansSide"><div class="col"><div class="colhead"><div class="name" id="aAname"></div><div class="stats" id="aAstats"></div></div><div id="aA"></div></div>
                    <div class="col"><div class="colhead"><div class="name" id="aBname"></div><div class="stats" id="aBstats"></div></div><div id="aB"></div></div></div>
  <div id="ansDiff" style="display:none"><div class="block answerblk"><div class="answer-body" id="aDiff"></div></div></div>
</main>
<script type="application/json" id="data">__DATA__</script>
<script type="application/json" id="models">__MODELS__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const MODELS=JSON.parse(document.getElementById('models').textContent);
const $=id=>document.getElementById(id);
const esc=s=>(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const stripCE=s=>(s||'').replace(/<\/?can_edit>/g,'');
const stripAns=s=>(s||'').replace(/<answer>[\s\S]*?<\/answer>/g,'').trim();
const norm=s=>(s||'').replace(/\s+/g,' ').trim();
function parseAttrs(s){const o={};const re=/(\w+)="([^"]*)"/g;let m;while((m=re.exec(s)))o[m[1]]=m[2];return o;}
// DR Tulu convention: only the first reasoning block is wrapped <think>…</think>;
// reasoning after each <tool_output> is emitted BARE and closed by a lone </think>.
// A leftover chunk that contains a </think> is such an implicit think block —
// render it as a think box (strip the tags) instead of raw "other" text.
function pushChunk(b,x){if(!x.trim())return;
  if(/<\/think>/.test(x)){b.push({type:'think',body:x.replace(/<\/?think>/g,'').trim(),implicit:true});}
  else b.push({type:'other',body:x});}
function parseBlocks(t){t=stripCE(t);const re=/<(think|call_tool|tool_output|answer)\b([^>]*)>([\s\S]*?)<\/\1>/g;const b=[];let last=0,m;
  while((m=re.exec(t))){if(m.index>last)pushChunk(b,t.slice(last,m.index));b.push({type:m[1],attrs:m[2],body:m[3]});last=re.lastIndex;}
  if(last<t.length)pushChunk(b,t.slice(last));return b;}
function parseSnippets(b){const re=/<snippet id=([^>]+)>([\s\S]*?)<\/snippet>/g;const o=[];let m;
  while((m=re.exec(b))){const inner=m[2];const tm=inner.match(/Title:\s*([\s\S]*?)(?:\n|$)/);const sm=inner.match(/Snippet:\s*([\s\S]*)/);
  o.push({id:m[1].trim(),title:tm?tm[1].trim():'',text:sm?sm[1].trim():norm(inner)});}return o;}
function renderAns(body){body=stripCE(body);let h='',last=0;const re=/<cite id="([^"]*)">([\s\S]*?)<\/cite>/g;let m;
  while((m=re.exec(body))){h+=esc(body.slice(last,m.index));h+='<span class="cited">'+esc(m[2])+'<sup class="citeid">['+esc(m[1])+']</sup></span>';last=re.lastIndex;}
  h+=esc(body.slice(last));return h;}
function blockHTML(b,isNew){
  const nt=isNew?'<span class="newtag">NEW</span>':'';const cls=isNew?' isnew':'';
  if(b.type==='think'){const bare=b.implicit?'<span class="baretag" title="reasoning block with no opening &lt;think&gt; tag (DR Tulu bare/post-retrieval reasoning)">no opening &lt;think&gt;</span>':'';
    return `<div class="block think${cls}${b.implicit?' bare':''}"><div class="lbl">💭 think ${bare} ${nt}</div><div class="body">${esc(stripCE(b.body).trim())}</div></div>`;}
  if(b.type==='call_tool'){const a=parseAttrs(b.attrs||'');const ins=a.limit==='5';const meta=[a.year?('year '+a.year):'',a.fieldsOfStudy?('· '+a.fieldsOfStudy):''].filter(Boolean).join(' ');
    return `<div class="block call${cls}"><div class="lbl">🔍 search ${meta?'<span class="newtag" style="color:var(--muted);border-color:var(--line)">'+esc(meta)+'</span>':''} ${ins?'<span class="reftag">inserted</span>':''} ${nt}</div><div class="body">${esc(stripCE(b.body).trim())}</div></div>`;}
  if(b.type==='tool_output'){const sn=parseSnippets(b.body);const items=sn.map(s=>`<div class="snip"><span class="sid">${esc(s.id)}</span> <span class="st">${esc(s.title)}</span><div class="sx">${esc(s.text)}</div></div>`).join('');
    return `<div class="block tool${cls}"><div class="lbl">📄 tool_output ${nt}</div><details class="snips"><summary>${sn.length} snippet${sn.length!==1?'s':''}</summary>${items||'<span class=empty>none</span>'}</details></div>`;}
  if(b.type==='answer')return `<div class="block answerblk"><div class="lbl">answer</div><div class="answer-body">${renderAns(b.body)}</div></div>`;
  return `<div class="block${cls}"><div class="body">${esc(norm(b.body))}</div></div>`;}
// DR Tulu-aware tag-balance: after </tool_output> reasoning resumes *bare*
// (implicit), so a lone </think> there is legitimate, not a double-close.
function structProblems(t){const re=/<think>|<\/think>|<call_tool\b[^>]*>|<\/call_tool>|<tool_output>|<\/tool_output>|<answer>/g;
  let reasoning=null,oc=false,oo=false,n=0,m;   // reasoning: null | 'explicit' | 'implicit'
  while((m=re.exec(t))){const g=m[0];
    if(g==='<think>'){if(reasoning==='explicit')n++;if(oc||oo)n++;reasoning='explicit';}
    else if(g==='</think>'){if(reasoning===null)n++;reasoning=null;}
    else if(g[1]!=='/'&&g.startsWith('<call_tool')){if(reasoning==='explicit')n++;if(oo)n++;reasoning=null;oc=true;}
    else if(g==='</call_tool>'){if(!oc)n++;oc=false;}
    else if(g==='<tool_output>'){if(reasoning==='explicit')n++;if(oc)n++;reasoning=null;oo=true;}
    else if(g==='</tool_output>'){if(!oo)n++;oo=false;reasoning='implicit';}
    else if(g==='<answer>'){reasoning=null;}}
  return n+(reasoning==='explicit'?1:0)+(oc?1:0)+(oo?1:0);}
const KNOWN=new Set(['<think','</think','<call_tool','</call_tool','<tool_output','</tool_output','<snippet','</snippet','<answer','</answer','<cite','</cite']);
function foreignTags(t){const s=new Set();(t.match(/<\/?[a-zA-Z_]+/g)||[]).forEach(x=>{if(!KNOWN.has(x))s.add(x);});return [...s];}
function wordDiff(a,b){const A=stripCE(a).replace(/<\/?cite[^>]*>/g,'').split(/(\s+)/),B=stripCE(b).replace(/<\/?cite[^>]*>/g,'').split(/(\s+)/);
  const n=A.length,m=B.length,dp=Array.from({length:n+1},()=>new Uint16Array(m+1));
  for(let i=n-1;i>=0;i--)for(let j=m-1;j>=0;j--)dp[i][j]=A[i]===B[j]?dp[i+1][j+1]+1:Math.max(dp[i+1][j],dp[i][j+1]);
  let i=0,j=0,o=[];const push=(t,s)=>{const p=o[o.length-1];if(p&&p[0]===t)p[1]+=s;else o.push([t,s]);};
  while(i<n&&j<m){if(A[i]===B[j]){push('=',A[i]);i++;j++;}else if(dp[i+1][j]>=dp[i][j+1]){push('-',A[i]);i++;}else{push('+',B[j]);j++;}}
  while(i<n)push('-',A[i++]);while(j<m)push('+',B[j++]);
  return o.map(([t,s])=>t==='='?esc(s):t==='-'?'<del>'+esc(s)+'</del>':'<ins>'+esc(s)+'</ins>').join('');}
const ORDER=['tag','location','search_required','organization_related','issue','critique_span','edit_span','s2_search_queries'];
function critVal(k,v){if(v==null)return '<span class=empty>null</span>';if(typeof v==='boolean')return v?'true':'false';
  if(Array.isArray(v)){if(k==='s2_search_queries')return v.map(q=>'• '+esc(q.query||JSON.stringify(q))).join('<br>')||'[]';
    return v.map(p=>Array.isArray(p)?(esc(p[0])+'  …  '+esc(p[1])):esc(String(p))).join('<br>')||'[]';}
  return esc(String(v));}

let ansMode='side';
function traceFor(rec,label){return label==='(original)'?stripAns(rec.original_trace):(rec.models[label]?rec.models[label].rewritten_trace:'');}
function ansFor(rec,label){return label==='(original)'?rec.original_answer:(rec.models[label]?rec.models[label].rewritten:'');}

function renderTraceCol(nameEl,statsEl,bodyEl,rec,label,oldSet,hl){
  const t=traceFor(rec,label);
  nameEl.textContent=label;
  if(label!=='(original)'&&!rec.models[label]){bodyEl.innerHTML='<div class=empty>no output for this model on this record</div>';statsEl.textContent='';return;}
  const blocks=parseBlocks(t);
  const nprob=structProblems(t),ft=foreignTags(t),ncall=(t.match(/<call_tool/g)||[]).length;
  const nthink=blocks.filter(b=>b.type==='think').length,nbare=blocks.filter(b=>b.implicit).length;
  statsEl.innerHTML=`${ncall} call_tool · ${blocks.length} blocks · think:${nthink} (<span class="${nbare?'bad':''}">${nbare} bare</span>) · struct:<span class="${nprob?'bad':''}">${nprob}</span>${ft.length?' · foreign:<span class="bad">'+esc(ft.join(','))+'</span>':''}`;
  bodyEl.innerHTML=blocks.map(b=>{const key=b.type+'|'+norm(b.body);const isNew=hl&&oldSet&&b.type!=='other'&&!oldSet.has(key);return blockHTML(b,isNew);}).join('')||'<div class=empty>empty</div>';
}
function renderAnsCol(nameEl,statsEl,bodyEl,rec,label){
  const a=ansFor(rec,label);nameEl.textContent=label;
  if(label!=='(original)'&&!rec.models[label]){bodyEl.innerHTML='<div class=empty>no output</div>';statsEl.textContent='';return;}
  statsEl.textContent=`${a.split(/\s+/).length} words · ${(a.match(/<cite/g)||[]).length} cites`;
  bodyEl.innerHTML='<div class="block answerblk"><div class="answer-body">'+renderAns(a.replace(/^<answer>\s*/,'').replace(/<\/answer>\s*$/,''))+'</div></div>';
}
function render(){
  const rec=DATA[+$('recsel').value];if(!rec)return;
  $('question').textContent=rec.question;
  let crit=[];try{const c=JSON.parse(rec.critique||'[]');crit=Array.isArray(c)?c:(c.local||[]);}catch(e){}
  $('critiques').innerHTML=crit.map(c=>{const keys=ORDER.filter(k=>k in c).concat(Object.keys(c).filter(k=>!ORDER.includes(k)));
    return '<div class="crit">'+keys.map(k=>`<div class="kv"><span class="k">${esc(k)}</span><span class="v${(k==='critique_span'||k==='edit_span')?' span':''}">${critVal(k,c[k])}</span></div>`).join('')+'</div>';}).join('')||'<div class=empty>none</div>';
  const oldSet=new Set(parseBlocks(stripAns(rec.original_trace)).map(b=>b.type+'|'+norm(b.body)));
  const hl=$('hlnew').checked,A=$('mA').value,B=$('mB').value;
  renderTraceCol($('tAname'),$('tAstats'),$('tA'),rec,A,oldSet,hl);
  renderTraceCol($('tBname'),$('tBstats'),$('tB'),rec,B,oldSet,hl);
  renderAnsCol($('aAname'),$('aAstats'),$('aA'),rec,A);
  renderAnsCol($('aBname'),$('aBstats'),$('aB'),rec,B);
  $('aDiff').innerHTML=wordDiff(ansFor(rec,A),ansFor(rec,B));
}
// build selectors
$('recsel').innerHTML=DATA.map((r,i)=>`<option value="${i}">#${i+1} — ${esc(r.question.slice(0,70))}</option>`).join('');
const opts=['(original)'].concat(MODELS).map(m=>`<option>${esc(m)}</option>`).join('');
$('mA').innerHTML=opts;$('mB').innerHTML=opts;
$('mA').value='(original)';
$('mB').value=MODELS[MODELS.length-1]||'(original)';
['recsel','mA','mB','hlnew'].forEach(id=>$(id).addEventListener('change',render));
document.querySelectorAll('#ansmode button').forEach(b=>b.addEventListener('click',()=>{ansMode=b.dataset.m;
  document.querySelectorAll('#ansmode button').forEach(x=>x.classList.toggle('on',x===b));
  $('ansSide').style.display=ansMode==='side'?'':'none';$('ansDiff').style.display=ansMode==='diff'?'':'none';}));
render();
</script></body></html>
"""

open(OUT, "w").write(HTML.replace("__DATA__", data_json).replace("__MODELS__", models_json))
print("wrote", OUT, "|", len(data), "records |", len(models), "models:", models)
