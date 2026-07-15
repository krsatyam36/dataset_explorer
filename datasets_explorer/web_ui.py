"""Lightweight live-status web UI for dataset_search runs.

Pure stdlib — runs an in-process ThreadingHTTPServer that:
  - serves a single-page dashboard on  /
  - streams events via Server-Sent Events on  /events

The agent (and CLI) push events through a global Broker; every browser tab
connected to /events receives them in real time.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

logger = logging.getLogger("datasets_explorer.web_ui")


class Broker:
    """Pub/sub fan-out for SSE clients. Each subscriber gets its own queue."""

    def __init__(self) -> None:
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._history: list[dict] = []  # replayed to new subscribers

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2048)
        with self._lock:
            for ev in self._history[-500:]:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, ev: dict) -> None:
        with self._lock:
            self._history.append(ev)
            if len(self._history) > 5000:
                del self._history[:1000]
            for q in self._subs:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass


_broker = Broker()
_server_thread: Optional[threading.Thread] = None
_server: Optional[ThreadingHTTPServer] = None
_server_port: Optional[int] = None

# Model-switch coordination. The web UI writes here; the agent loop polls
# `consume_model_switch()` at the top of each iteration.
_pending_model: Optional[str] = None
_pending_lock = threading.Lock()


def publish(ev: dict) -> None:
    """Public API — agents and CLI call this to emit an event."""
    _broker.publish(ev)


def request_model_switch(name: str) -> None:
    """Called by the HTTP handler when the user picks a new model."""
    global _pending_model
    with _pending_lock:
        _pending_model = (name or "").strip() or None


def consume_model_switch() -> Optional[str]:
    """Atomically read + clear the pending model switch. Returns the new
    model name if one was requested, else None."""
    global _pending_model
    with _pending_lock:
        m, _pending_model = _pending_model, None
        return m


def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        # Suppress access-log spam to stderr; we have our own logger.
        def log_message(self, format, *args):  # noqa: A002
            return

        def do_GET(self):  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                self._serve_html()
            elif self.path == "/events":
                self._serve_sse()
            elif self.path == "/models":
                self._serve_models()
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            if self.path == "/switch-model":
                self._serve_switch_model()
            else:
                self.send_error(404)

        def _serve_models(self):
            try:
                import ollama
                from .config import OLLAMA_HOST
                client = ollama.Client(host=OLLAMA_HOST)
                resp = client.list()
                names = []
                for m in (resp.get("models") if isinstance(resp, dict) else getattr(resp, "models", [])) or []:
                    n = m.get("name") if isinstance(m, dict) else getattr(m, "model", None) or getattr(m, "name", None)
                    if n:
                        names.append(n)
            except Exception as e:
                names = []
                logger.warning(f"/models error: {e}")
            body = json.dumps({"models": sorted(set(names))}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_switch_model(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                name = (payload.get("model") or "").strip()
            except Exception:
                name = ""
            if not name:
                self.send_error(400, "missing 'model' in body")
                return
            request_model_switch(name)
            _broker.publish({"type": "model_switch_requested", "model": name})
            body = json.dumps({"ok": True, "model": name}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_html(self):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            q = _broker.subscribe()
            try:
                last_ping = time.time()
                while True:
                    try:
                        ev = q.get(timeout=1.0)
                    except queue.Empty:
                        ev = None
                    if ev is not None:
                        try:
                            self.wfile.write(b"data: ")
                            self.wfile.write(json.dumps(ev, default=str).encode("utf-8"))
                            self.wfile.write(b"\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                    now = time.time()
                    if now - last_ping > 15:
                        try:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                        last_ping = now
            finally:
                _broker.unsubscribe(q)

    return Handler


def start(port: int = 7860, host: str = "127.0.0.1") -> Optional[int]:
    """Start the web UI server in a background thread. Idempotent."""
    global _server, _server_thread, _server_port
    if _server_thread is not None and _server_thread.is_alive():
        return _server_port
    # Try a few ports if the requested one is busy.
    for candidate in [port, port + 1, port + 2, 0]:
        try:
            srv = ThreadingHTTPServer((host, candidate), _make_handler())
            break
        except OSError:
            continue
    else:
        logger.warning("Could not bind a port for the web UI")
        return None
    _server = srv
    _server_port = srv.server_address[1]

    def _run():
        try:
            srv.serve_forever(poll_interval=0.5)
        except Exception as e:
            logger.warning(f"web_ui server stopped: {e}")

    _server_thread = threading.Thread(target=_run, daemon=True, name="web_ui")
    _server_thread.start()
    return _server_port


def stop() -> None:
    global _server
    if _server is not None:
        try:
            _server.shutdown()
        except Exception:
            pass
        _server = None


# --------------------------------------------------------------------------
# Static dashboard. Single self-contained file with embedded CSS + JS.
# --------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>dataset_search · live</title>
<style>
:root{
  --bg:#0b0e14; --panel:#11151c; --panel-2:#161b24; --border:#222a36;
  --text:#e6edf3; --muted:#8b95a7; --accent:#7aa2f7; --accent-2:#bb9af7;
  --good:#9ece6a; --warn:#e0af68; --bad:#f7768e; --info:#7dcfff;
  --mono:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --sans:Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px}
header{padding:14px 22px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:18px;flex-wrap:wrap;background:var(--panel)}
header .brand{font-weight:700;letter-spacing:.5px;color:var(--accent)}
header .subject{font-family:var(--mono);color:var(--text);background:var(--panel-2);padding:5px 10px;border-radius:6px;border:1px solid var(--border)}
header .status{margin-left:auto;display:flex;align-items:center;gap:14px}
header .status .pill{padding:4px 10px;border-radius:999px;border:1px solid var(--border);background:var(--panel-2);font-family:var(--mono);font-size:12px;color:var(--muted)}
header .status .live{color:var(--good)}
#model-select{background:var(--panel-2);color:var(--text);border:1px solid var(--border);border-radius:4px;font-family:var(--mono);font-size:12px;padding:1px 4px;outline:none}
#model-select:hover{border-color:var(--accent)}
#model-pill{display:inline-flex;align-items:center;gap:6px}
@keyframes spin{to{transform:rotate(360deg)}}
#model-switching{display:inline-block;animation:spin 1s linear infinite}
header .status .live::before{content:'';display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--good);margin-right:6px;box-shadow:0 0 0 0 rgba(158,206,106,.7);animation:pulse 1.5s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(158,206,106,.7)}70%{box-shadow:0 0 0 8px rgba(158,206,106,0)}100%{box-shadow:0 0 0 0 rgba(158,206,106,0)}}

.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;padding:14px 22px;background:var(--bg)}
.kpi{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.kpi .lbl{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.6px}
.kpi .val{font-family:var(--mono);font-size:20px;margin-top:2px}
.kpi .sub{color:var(--muted);font-size:11px;margin-top:1px}

main{display:grid;grid-template-columns:1.55fr 1fr;gap:14px;padding:0 22px 22px}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;overflow:hidden;display:flex;flex-direction:column;min-height:300px}
.panel h2{margin:0;padding:10px 14px;border-bottom:1px solid var(--border);font-size:12px;letter-spacing:.6px;text-transform:uppercase;color:var(--muted);display:flex;align-items:center;gap:8px}
.panel h2 .count{margin-left:auto;font-family:var(--mono);color:var(--text);background:var(--panel-2);padding:2px 8px;border-radius:6px;font-size:11px}
.panel .body{flex:1;overflow:auto}

.now{padding:14px}
.now .what{font-family:var(--mono);font-size:16px;color:var(--text);word-break:break-all;line-height:1.45}
.now .next{margin-top:10px;color:var(--muted);font-size:12px}
.now .ic{display:inline-block;width:22px}

.stream{font-family:var(--mono);font-size:12px;line-height:1.5}
.stream .row{padding:4px 14px;border-bottom:1px solid #161b24;display:flex;gap:8px;align-items:flex-start}
.stream .row:hover{background:rgba(122,162,247,0.04)}
.stream .t{color:var(--muted);white-space:nowrap;flex-shrink:0}
.stream .ic{flex-shrink:0;width:20px;text-align:center}
.stream .body{flex:1;word-break:break-all;color:var(--text)}
.stream .body a{color:var(--accent);text-decoration:none}
.stream .body a:hover{text-decoration:underline}
.stream .row.search .ic{color:var(--accent-2)}
.stream .row.fetch .ic{color:var(--info)}
.stream .row.pdf .ic,.stream .row.readme .ic,.stream .row.arxiv .ic{color:var(--accent-2)}
.stream .row.store .body{color:var(--good)}
.stream .row.rejected .ic{color:var(--bad)}
.stream .row.rejected .body{color:var(--bad)}
.stream .row.error .ic{color:var(--warn)}
.stream .row.error .body{color:var(--warn)}
.stream .row.thinking{background:rgba(187,154,247,0.04)}
.stream .row.thinking .body{color:var(--accent-2);font-style:italic;white-space:pre-wrap}

.findings{font-family:var(--mono);font-size:12px}
.findings .row{padding:9px 14px;border-bottom:1px solid #161b24;display:grid;grid-template-columns:48px 1fr;gap:8px}
.findings .score{color:var(--good);font-weight:700}
.findings .score.mid{color:var(--warn)}
.findings .score.low{color:var(--bad)}
.findings .name{color:var(--text);margin-bottom:3px;font-family:var(--sans);font-size:13px;font-weight:600}
.findings .url{color:var(--accent);word-break:break-all}
.findings .url a{color:inherit;text-decoration:none}
.findings .url a:hover{text-decoration:underline}
.findings .meta{color:var(--muted);font-size:11px;margin-top:3px}
.findings .meta .tag{display:inline-block;background:var(--panel-2);border:1px solid var(--border);padding:1px 6px;border-radius:4px;margin-right:5px}

.lower{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:0 22px 22px}

.thinking-list{font-family:var(--mono);font-size:12px}
.thinking-list details{padding:8px 14px;border-bottom:1px solid #161b24}
.thinking-list details summary{cursor:pointer;color:var(--muted);outline:none}
.thinking-list details[open] summary{color:var(--accent-2);margin-bottom:6px}
.thinking-list details pre{margin:0;padding:8px 10px;background:var(--bg);border:1px solid var(--border);border-radius:6px;white-space:pre-wrap;color:var(--text);max-height:280px;overflow:auto}

.sites{font-family:var(--mono);font-size:12px;padding:10px 14px;display:flex;flex-wrap:wrap;gap:6px}
.sites .chip{background:var(--panel-2);border:1px solid var(--border);padding:3px 8px;border-radius:5px;color:var(--text);display:flex;align-items:center;gap:5px}
.sites .chip .n{color:var(--muted);font-size:11px}

footer{padding:8px 22px;color:var(--muted);font-size:11px;border-top:1px solid var(--border);font-family:var(--mono);text-align:right}

@media (max-width: 1100px){main,.lower{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(3,1fr)}}
</style>
</head>
<body>
<header>
  <div class="brand">dataset_search</div>
  <div class="subject" id="subject">—</div>
  <div class="status">
    <span class="pill" id="model-pill">
      model: <select id="model-select" title="Switch model on the fly — full history is preserved">
        <option value="">—</option>
      </select>
      <span id="model-switching" style="display:none;color:var(--accent-2);margin-left:6px">↻</span>
    </span>
    <span class="pill" id="depth">depth: —</span>
    <span class="pill live" id="live">live</span>
    <span class="pill" id="sound-btn" style="cursor:pointer" onclick="toggleSound()" title="Toggle notification sounds">🔔</span>
    <span class="pill" id="elapsed">T+00m00s</span>
  </div>
</header>

<section class="kpis">
  <div class="kpi"><div class="lbl">Iteration</div><div class="val" id="kpi-iter">0</div><div class="sub" id="kpi-iter-cap">/—</div></div>
  <div class="kpi"><div class="lbl">Stored</div><div class="val" id="kpi-stored">0</div><div class="sub" id="kpi-needed">need —</div></div>
  <div class="kpi"><div class="lbl">Mainstream : Alternative</div><div class="val" id="kpi-ratio">0 : 0</div><div class="sub">target 1 : 2+</div></div>
  <div class="kpi"><div class="lbl">Searches</div><div class="val" id="kpi-search">0</div><div class="sub" id="kpi-search-rejected">0 rejected</div></div>
  <div class="kpi"><div class="lbl">PDFs read</div><div class="val" id="kpi-pdf">0</div><div class="sub" id="kpi-readme">0 READMEs</div></div>
  <div class="kpi"><div class="lbl">Sites visited</div><div class="val" id="kpi-sites">0</div><div class="sub" id="kpi-fetches">0 fetches</div></div>
</section>

<main>
  <section class="panel">
    <h2>Live activity feed <span class="count" id="stream-count">0</span></h2>
    <div class="body stream" id="stream"></div>
  </section>
  <section>
    <div class="panel" style="margin-bottom:14px">
      <h2>Where the agent is now</h2>
      <div class="now">
        <div class="what" id="now-what"><span class="ic">·</span><span id="now-text">waiting for the agent…</span></div>
        <div class="next" id="now-meta"></div>
      </div>
    </div>
    <div class="panel">
      <h2>Confirmed datasets <span class="count" id="findings-count">0</span></h2>
      <div class="body findings" id="findings"></div>
    </div>
  </section>
</main>

<section class="lower">
  <div class="panel">
    <h2>Agent thinking <span class="count" id="thinking-count">0</span></h2>
    <div class="body thinking-list" id="thinking"></div>
  </div>
  <div class="panel">
    <h2>Sites visited <span class="count" id="sites-count">0</span></h2>
    <div class="body"><div class="sites" id="sites"></div></div>
  </div>
</section>

<footer>events stream over Server-Sent Events · keep this tab open during the run</footer>

<script>
const ICONS = {
  web_search: '🔎', fetch_page: '📄', fetch_page_js: '🌐',
  read_pdf: '📑', read_github_readme: '📘', arxiv_search: '📚',
  store_dataset: '💾', mark_search_complete: '✅',
};
const KIND_OF = {
  web_search: 'search', fetch_page: 'fetch', fetch_page_js: 'fetch',
  read_pdf: 'pdf', read_github_readme: 'readme', arxiv_search: 'arxiv',
  store_dataset: 'store', mark_search_complete: 'complete',
};

const $ = (id) => document.getElementById(id);
const stream = $('stream'), findings = $('findings'), thinkingPane = $('thinking');
const sitesPane = $('sites');

const state = {
  iteration: 0, stored: 0, mainstream: 0, alternative: 0,
  searches: 0, searchesRejected: 0, pdfs: 0, readmes: 0, fetches: 0,
  sites: new Map(), thinkingCount: 0, streamCount: 0, findingsCount: 0,
  startedAt: null,
};

function fmtElapsed(secs){
  if(!secs && secs!==0) return '';
  secs = Math.max(0, Math.floor(secs));
  const h = Math.floor(secs/3600), m = Math.floor((secs%3600)/60), s = secs%60;
  return h ? `T+${h}h${String(m).padStart(2,'0')}m${String(s).padStart(2,'0')}s`
           : `T+${String(m).padStart(2,'0')}m${String(s).padStart(2,'0')}s`;
}
function tick(){
  if(state.startedAt!=null){
    const e = (Date.now()/1000) - state.startedAt;
    $('elapsed').textContent = fmtElapsed(e);
  }
}
setInterval(tick, 1000);

function domainOf(u){
  try { const url = new URL(u); return url.hostname.replace(/^www\./,''); } catch(e) { return ''; }
}
function bumpSite(host){
  if(!host) return;
  state.sites.set(host, (state.sites.get(host)||0) + 1);
}
function renderSites(){
  $('kpi-sites').textContent = state.sites.size;
  const sorted = [...state.sites.entries()].sort((a,b)=>b[1]-a[1]).slice(0,80);
  sitesPane.innerHTML = sorted.map(([h,n]) =>
    `<span class="chip">${h} <span class="n">${n}</span></span>`).join('');
  $('sites-count').textContent = state.sites.size;
}

function pushRow(kind, ic, html, meta){
  const t = meta && meta.elapsed!=null ? fmtElapsed(meta.elapsed) : '';
  const it = meta && meta.iteration!=null ? `iter ${String(meta.iteration).padStart(3)}` : '';
  const row = document.createElement('div');
  row.className = 'row ' + kind;
  row.innerHTML = `<span class="t">[${t} · ${it}]</span><span class="ic">${ic}</span><span class="body">${html}</span>`;
  stream.appendChild(row);
  state.streamCount += 1;
  $('stream-count').textContent = state.streamCount;
  while(stream.childElementCount > 800) stream.removeChild(stream.firstChild);
  stream.scrollTop = stream.scrollHeight;
}

function setNow(ic, txt, meta){
  $('now-text').innerHTML = txt;
  $('now-what').firstChild.textContent = ic;
  if(meta){
    $('now-meta').textContent =
      `iter ${meta.iteration ?? '—'} · ${fmtElapsed(meta.elapsed)}`;
  }
}

function pushFinding(d){
  const score = (d.relevance_score ?? 0).toFixed(2);
  const cls = score >= 0.8 ? '' : (score >= 0.5 ? 'mid' : 'low');
  const dl = d.download_url && d.download_url !== d.url
        ? `<div class="meta"><span class="tag">download</span><a href="${d.download_url}" target="_blank">${d.download_url}</a></div>` : '';
  const tags = [];
  if(d.license_spdx) tags.push(`<span class="tag">${d.license_spdx}</span>`);
  if(d.country) tags.push(`<span class="tag">${d.country}</span>`);
  if(d.institution) tags.push(`<span class="tag">${d.institution}</span>`);
  if(d.doi) tags.push(`<span class="tag">DOI ${d.doi}</span>`);
  const meta = tags.length ? `<div class="meta">${tags.join(' ')}</div>` : '';
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `
    <div class="score ${cls}">${score}</div>
    <div>
      <div class="name">${escapeHtml(d.name||'')}</div>
      <div class="url"><a href="${d.url}" target="_blank">${escapeHtml(d.url||'')}</a></div>
      ${dl}${meta}
    </div>`;
  findings.insertBefore(row, findings.firstChild);
  state.findingsCount += 1;
  $('findings-count').textContent = state.findingsCount;
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

function pushThinking(meta, text){
  const det = document.createElement('details');
  const t = meta && meta.elapsed!=null ? fmtElapsed(meta.elapsed) : '';
  const it = meta && meta.iteration!=null ? `iter ${meta.iteration}` : '';
  det.innerHTML = `<summary>${t} · ${it} · ${(text||'').length} chars</summary><pre></pre>`;
  det.querySelector('pre').textContent = text || '';
  thinkingPane.insertBefore(det, thinkingPane.firstChild);
  state.thinkingCount += 1;
  $('thinking-count').textContent = state.thinkingCount;
  while(thinkingPane.childElementCount > 200) thinkingPane.removeChild(thinkingPane.lastChild);
}

function beep(freq, dur, vol){
  try {
    const ctx = new (window.AudioContext||window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.frequency.value = freq;
    osc.type = 'sine';
    gain.gain.value = vol || 0.08;
    osc.start(); osc.stop(ctx.currentTime + (dur||0.15));
  } catch(e){} // audio not supported
}
let _muted = localStorage.getItem('soundMuted')==='1';
let _volume = parseFloat(localStorage.getItem('soundVolume')||'0.08');
function beep(freq, dur, vol){
  try {
    const ctx = new (window.AudioContext||window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.frequency.value = freq;
    osc.type = 'sine';
    gain.gain.value = (vol!=null ? vol : _volume);
    osc.start(); osc.stop(ctx.currentTime + (dur||0.15));
  } catch(e){}
}
function playSound(type){
  if(_muted) return;
  if(type==='done'){ beep(660,0.12); setTimeout(()=>beep(880,0.2),140); }
  else if(type==='store'){ beep(520,0.08); }
  else if(type==='warn'){ beep(300,0.1); }
}
function toggleSound(){
  _muted = !_muted;
  localStorage.setItem('soundMuted', _muted ? '1' : '0');
  const btn = $('sound-btn');
  if(btn) btn.textContent = _muted ? '🔇' : '🔔';
  if(!_muted){ beep(660,0.08); setTimeout(()=>beep(880,0.1),100); }
}

function applyEvent(ev){
  if(ev.type === 'run_start'){
    state.startedAt = (Date.now()/1000) - (ev.elapsed||0);
    if(ev.subject) $('subject').textContent = ev.subject;
    if(ev.model) {
      state.activeModel = ev.model;
      const sel = $('model-select');
      if(![...sel.options].some(o => o.value === ev.model)){
        const opt = document.createElement('option');
        opt.value = ev.model; opt.textContent = ev.model;
        sel.appendChild(opt);
      }
      sel.value = ev.model;
    }
    if(ev.depth) $('depth').textContent = `depth: ${ev.depth}`;
    if(ev.max_iters) $('kpi-iter-cap').textContent = `/ ${ev.max_iters}`;
    if(ev.min_needed) $('kpi-needed').textContent = `need ${ev.min_needed}`;
    return;
  }
  if(ev.type === 'model_switch_requested'){
    $('model-switching').style.display = 'inline-block';
    pushRow('thinking', '↻', `model switch requested → ${escapeHtml(ev.model||'')} (will take effect at next iteration)`, ev);
    return;
  }
  if(ev.type === 'model_switched'){
    $('model-switching').style.display = 'none';
    state.activeModel = ev.to;
    const sel = $('model-select');
    if(![...sel.options].some(o => o.value === ev.to)){
      const opt = document.createElement('option');
      opt.value = ev.to; opt.textContent = ev.to;
      sel.appendChild(opt);
    }
    sel.value = ev.to;
    pushRow('store', '↻', `model switched: ${escapeHtml(ev.from||'')} → <b>${escapeHtml(ev.to||'')}</b> (history preserved)`, ev);
    return;
  }
  if(ev.type === 'iter_start'){
    state.iteration = ev.iteration || 0;
    $('kpi-iter').textContent = state.iteration;
    return;
  }
  if(ev.type === 'thinking'){
    pushThinking(ev, ev.text);
    pushRow('thinking', '💭', escapeHtml((ev.text||'').slice(0,140)) + ((ev.text||'').length>140?'…':''), ev);
    return;
  }
  if(ev.type === 'tool_call'){
    const args = ev.args || {};
    const name = ev.name;
    const ic = ICONS[name] || '·';
    const kind = KIND_OF[name] || 'misc';
    let body = '';
    if(name==='web_search' || name==='arxiv_search'){
      body = escapeHtml(args.query || '');
      state.searches += 1; $('kpi-search').textContent = state.searches;
    } else if(name==='fetch_page' || name==='fetch_page_js'){
      const u = args.url || '';
      body = `<a href="${u}" target="_blank">${escapeHtml(u)}</a>`;
      bumpSite(domainOf(u));
      state.fetches += 1; $('kpi-fetches').textContent = `${state.fetches} fetches`;
      renderSites();
    } else if(name==='read_pdf'){
      const u = args.url || '';
      body = `<a href="${u}" target="_blank">${escapeHtml(u)}</a>`;
      bumpSite(domainOf(u)); renderSites();
      state.pdfs += 1; $('kpi-pdf').textContent = state.pdfs;
    } else if(name==='read_github_readme'){
      const u = args.repo_url || args.url || '';
      body = `<a href="${u}" target="_blank">${escapeHtml(u)}</a>`;
      bumpSite('github.com'); renderSites();
      state.readmes += 1; $('kpi-readme').textContent = `${state.readmes} READMEs`;
    } else if(name==='store_dataset'){
      const score = args.relevance_score!=null ? `[${Number(args.relevance_score).toFixed(2)}] ` : '';
      body = `${score}${escapeHtml(args.name||'')}`;
    } else {
      body = escapeHtml(JSON.stringify(args).slice(0,160));
    }
    pushRow(kind, ic, body, ev);
    setNow(ic, body, ev);
    return;
  }
  if(ev.type === 'tool_result'){ playSound('warn');
    const ic = ev.status === 'rejected' ? '⛔' : '⚠️';
    const cls = ev.status === 'rejected' ? 'rejected' : 'error';
    const msg = (ev.message||'').slice(0,260);
    pushRow(cls, ic, `${escapeHtml(ev.name||'')} ${ev.status}: ${escapeHtml(msg)}`, ev);
    if(ev.name==='web_search' && ev.status==='rejected'){
      state.searchesRejected += 1;
      $('kpi-search-rejected').textContent = `${state.searchesRejected} rejected`;
    }
    return;
  }
  if(ev.type === 'dataset_stored'){
    state.stored += 1;
    $('kpi-stored').textContent = state.stored;
    playSound('store');
    if(ev.is_mainstream) state.mainstream += 1; else state.alternative += 1;
    $('kpi-ratio').textContent = `${state.mainstream} : ${state.alternative}`;
    pushFinding(ev.dataset || {});
    pushRow('store', '⭐', `[${(ev.dataset?.relevance_score ?? 0).toFixed(2)}] ${escapeHtml(ev.dataset?.name||'')}`, ev);
    return;
  }
  if(ev.type === 'run_complete'){
    $('live').textContent = 'done';
    $('live').classList.remove('live');
    playSound('done');
    pushRow('store', '✅', `Run complete · ${ev.status||''} · ${ev.stored||0} stored`, ev);
    return;
  }
}

async function loadModels(){
  try {
    const r = await fetch('/models');
    const j = await r.json();
    const sel = $('model-select');
    const current = state.activeModel || sel.value;
    const have = new Set([...sel.options].map(o => o.value));
    for(const name of (j.models || [])){
      if(have.has(name)) continue;
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      sel.appendChild(opt);
    }
    if(current) sel.value = current;
  } catch(e){ console.warn('loadModels', e); }
}
async function switchModel(name){
  if(!name) return;
  $('model-switching').style.display = 'inline-block';
  try {
    const r = await fetch('/switch-model', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: name}),
    });
    if(!r.ok){
      $('model-switching').style.display = 'none';
      console.warn('switch failed', await r.text());
    }
  } catch(e){
    $('model-switching').style.display = 'none';
    console.warn('switch error', e);
  }
}
$('model-select').addEventListener('change', (e) => {
  const name = e.target.value;
  if(name && name !== state.activeModel) switchModel(name);
});
loadModels();
setInterval(loadModels, 30000);  // refresh in case the user pulls a new model

(function initSound(){
  const btn = $('sound-btn');
  if(btn && _muted) btn.textContent = '🔇';
})();

function connect(){
  const es = new EventSource('/events');
  es.onmessage = (m) => {
    try { applyEvent(JSON.parse(m.data)); } catch(e){ console.warn(e); }
  };
  es.onerror = () => { /* browser auto-reconnects */ };
}
connect();
</script>
</body>
</html>
"""
