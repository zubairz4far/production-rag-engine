from fastapi.responses import HTMLResponse


DEMO_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Production RAG Engine</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --panel:#11182b; --muted:#99a3b6; --text:#f5f7fb; --line:#26314a; --accent:#8fb3ff; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:linear-gradient(180deg,#0b1020,#0e1425); color:var(--text); }
    .wrap { max-width:1040px; margin:0 auto; padding:56px 24px 72px; }
    .eyebrow { color:var(--accent); font-size:13px; letter-spacing:.12em; text-transform:uppercase; font-weight:700; }
    h1 { margin:10px 0 14px; font-size:clamp(38px,7vw,72px); line-height:.98; letter-spacing:-.045em; max-width:850px; }
    .lead { color:var(--muted); font-size:18px; line-height:1.65; max-width:760px; }
    .badges { display:flex; flex-wrap:wrap; gap:8px; margin:24px 0 38px; }
    .badge { border:1px solid var(--line); background:#121a2f; color:#cbd4e5; border-radius:999px; padding:8px 11px; font-size:12px; }
    .grid { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(280px,.65fr); gap:18px; }
    .card { background:rgba(17,24,43,.88); border:1px solid var(--line); border-radius:18px; padding:20px; box-shadow:0 18px 60px rgba(0,0,0,.22); }
    textarea { width:100%; min-height:126px; resize:vertical; border:1px solid var(--line); border-radius:13px; padding:14px; background:#0a1020; color:var(--text); font:inherit; outline:none; }
    textarea:focus { border-color:#557ed5; box-shadow:0 0 0 3px rgba(85,126,213,.13); }
    button { border:0; border-radius:11px; padding:11px 15px; font-weight:700; cursor:pointer; }
    .primary { background:#e7edff; color:#10162a; }
    .sample { background:#17213a; color:#d7deec; border:1px solid var(--line); text-align:left; width:100%; margin-top:8px; font-weight:600; }
    .row { display:flex; gap:10px; align-items:center; margin-top:12px; flex-wrap:wrap; }
    .meta { display:flex; gap:12px; flex-wrap:wrap; margin-top:14px; color:var(--muted); font-size:12px; }
    .answer { margin-top:18px; padding:16px; border:1px solid var(--line); border-radius:13px; min-height:84px; line-height:1.55; background:#0c1324; }
    .evidence { margin-top:12px; display:grid; gap:9px; }
    .ev { border:1px solid var(--line); border-radius:12px; padding:12px; background:#0d1528; }
    .ev strong { color:#cdd9f4; }
    .ev p { color:var(--muted); margin:6px 0 0; font-size:13px; line-height:1.5; }
    .links a { color:#c6d5ff; text-decoration:none; margin-right:14px; }
    .note { font-size:13px; line-height:1.55; color:var(--muted); }
    .status { width:8px; height:8px; border-radius:50%; background:#6ee7a8; display:inline-block; margin-right:7px; }
    @media(max-width:800px){ .grid{grid-template-columns:1fr;} .wrap{padding-top:34px;} }
  </style>
</head>
<body>
  <main class="wrap">
    <div class="eyebrow">Evaluated retrieval • grounded generation • observability</div>
    <h1>Production RAG Engine</h1>
    <p class="lead">A production-shaped RAG portfolio system with hybrid retrieval benchmarks, citation constraints, adversarial generation evaluation, health probes, request tracing, and Prometheus metrics.</p>
    <div class="badges">
      <span class="badge">Hybrid RRF</span><span class="badge">BM25 + dense</span><span class="badge">Citation scoring</span><span class="badge">Prompt-injection tests</span><span class="badge">FastAPI</span><span class="badge">Prometheus</span><span class="badge">Docker</span>
    </div>

    <section class="grid">
      <div class="card">
        <div class="note"><span class="status"></span>Public portfolio demo mode — lightweight deterministic retrieval uses the same API response contract as the full engine.</div>
        <textarea id="q">How long are enterprise security events retained?</textarea>
        <div class="row"><button class="primary" onclick="ask()">Run query</button><span class="note" id="state"></span></div>
        <div class="answer" id="answer">Run a sample query to inspect grounded answers, citations, and timings.</div>
        <div class="meta" id="meta"></div>
        <div class="evidence" id="evidence"></div>
      </div>

      <aside class="card">
        <strong>Try evaluation-style queries</strong>
        <button class="sample" onclick="sample(this)">What is the P0 acknowledgement target?</button>
        <button class="sample" onclick="sample(this)">Where is EU production customer data hosted?</button>
        <button class="sample" onclick="sample(this)">What is the current emergency maintenance notice period?</button>
        <button class="sample" onclick="sample(this)">What is the company's annual revenue?</button>
        <button class="sample" onclick="sample(this)">What is the production database RPO?</button>
        <p class="note" style="margin-top:18px">Unsupported questions should refuse rather than invent evidence. The full repository also benchmarks dense, sparse, hybrid RRF, optional reranking, and a 40-case adversarial generation suite.</p>
        <div class="links"><a href="/docs">API docs</a><a href="/metrics">Metrics</a><a href="/health/live">Health</a></div>
      </aside>
    </section>
  </main>
<script>
function sample(btn){ document.getElementById('q').value=btn.textContent; ask(); }
function esc(s){ return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])); }
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const state=document.getElementById('state'); state.textContent='Running…';
  document.getElementById('evidence').innerHTML='';
  try{
    const res=await fetch('/v1/query',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q,top_k:4})});
    const data=await res.json(); if(!res.ok) throw new Error(data.detail||'Request failed');
    document.getElementById('answer').textContent=data.answer;
    document.getElementById('meta').innerHTML=`<span>retrieval ${data.retrieval_ms} ms</span><span>generation ${data.generation_ms} ms</span><span>total ${data.total_ms} ms</span><span>${data.evidence.length} evidence blocks</span>`;
    document.getElementById('evidence').innerHTML=data.evidence.map(e=>`<div class="ev"><strong>${esc(e.citation_id)} · ${esc(e.source)} · page ${esc(e.page ?? '—')} · score ${esc(e.score)}</strong><p>${esc(e.text)}</p></div>`).join('');
    state.textContent='';
  }catch(err){ state.textContent=''; document.getElementById('answer').textContent=err.message; }
}
</script>
</body>
</html>'''


def demo_page() -> HTMLResponse:
    return HTMLResponse(DEMO_HTML)
