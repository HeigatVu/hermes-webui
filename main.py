"""
FastAPI BFF (Backend-For-Frontend) wrapper around hermes-webui server.py.

Starts server.py as a subprocess on 127.0.0.1:8787, reverse-proxies all requests
through a health-gated httpx client, and exposes chat endpoints for Claude/agy.

Architecture: subprocess (NOT threading) — avoids GIL contention between the
sync BaseHTTPRequestHandler and Uvicorn's async event loop.

WARNING: This module uses module-level globals (_proc, _healthy, _client).
Do NOT configure uvicorn with workers > 1 — each worker would spawn its own
hermes-webui subprocess.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

import httpx

# ── Configuration ────────────────────────────────────────────────────────────
HERMES_SUB_HOST = os.environ.get("HERMES_SUB_HOST", "127.0.0.1")
HERMES_SUB_PORT = int(os.environ.get("HERMES_SUB_PORT", "8787"))
HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
SIDECAR_HOST = os.environ.get("SIDECAR_HOST", "0.0.0.0")
SIDECAR_PORT = int(os.environ.get("SIDECAR_PORT", "8501"))
MAX_PROMPT_LENGTH = int(os.environ.get("SIDECAR_MAX_PROMPT_LENGTH", "16384"))  # 16KB

SERVER_SCRIPT = Path(__file__).resolve().parent / "server.py"
SUBSERVER_LOG = Path("/tmp/hermes-subserver.log")

logger = logging.getLogger("bff")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

# ── Safe environment for chat subprocesses ───────────────────────────────────
# Never pass the full os.environ to chat CLI subprocesses — it may contain
# API keys, tokens, and other secrets that could leak into CLI output.
_CHAT_SAFE_ENV_KEYS = {
    "PATH", "HOME", "USER", "LOGNAME", "SHELL",
    "LANG", "LC_ALL", "LC_CTYPE",
    "TERM", "COLORTERM", "NO_COLOR", "FORCE_COLOR",
    "PYTHONUNBUFFERED",
    "HERMES_HOME",
    "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    "TMPDIR", "TEMP", "TMP",
}

_CHAT_SAFE_ENV = {
    k: v for k, v in os.environ.items()
    if k in _CHAT_SAFE_ENV_KEYS or k.startswith(("UV_", "PIP_", "PYTHON"))
}
_CHAT_SAFE_ENV.setdefault("PYTHONUNBUFFERED", "1")
_CHAT_SAFE_ENV.setdefault("TERM", "xterm-256color")
_CHAT_SAFE_ENV.setdefault("COLORTERM", "truecolor")

# ── Workdir validation ───────────────────────────────────────────────────────
_ALLOWED_WORKDIR_ROOTS = [
    Path(HERMES_HOME).resolve(),
    Path.home().resolve(),
    Path("/workspace").resolve(),
    Path("/aiOS-ui").resolve(),
    Path("/tmp").resolve(),
    Path("/outputs").resolve(),
]


def _validate_workdir(workdir: str) -> Path:
    """Validate workdir is within an allowed root. Returns resolved Path."""
    p = Path(workdir).expanduser().resolve()
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"workdir does not exist: {workdir}")
    for root in _ALLOWED_WORKDIR_ROOTS:
        try:
            p.relative_to(root)
            return p
        except ValueError:
            continue
    raise HTTPException(status_code=403, detail=f"workdir not in allowed paths: {workdir}")


# ── SSE escaping ─────────────────────────────────────────────────────────────
# SSE field names: 'data', 'event', 'id', 'retry'. Lines starting with ':' are comments.
# We must prevent CLI output from injecting control fields.
_SSE_CONTROL_RE = re.compile(r"^(data|event|id|retry):\s*|^:", re.MULTILINE)


def _escape_sse_line(line: str) -> str:
    """Prefix-protect lines that look like SSE control fields."""
    if _SSE_CONTROL_RE.match(line):
        return f":{line}"  # SSE comment — hidden from client EventSource
    return line


# ── Subprocess management ────────────────────────────────────────────────────
_proc: subprocess.Popen | None = None
_healthy: bool = False
_client: httpx.AsyncClient | None = None
# Track chat subprocesses for cleanup on shutdown
_active_chat_procs: set[asyncio.subprocess.Process] = set()


def _subprocess_start() -> subprocess.Popen:
    """Launch server.py in an isolated process group (os.setsid).
    Redirects stdout+stderr to a log file to avoid pipe-buffer deadlock."""
    logger.info(f"Starting hermes-webui subprocess on {HERMES_SUB_HOST}:{HERMES_SUB_PORT}")
    env = {
        **os.environ,
        "HERMES_HOME": HERMES_HOME,
        "HERMES_WEBUI_HOST": HERMES_SUB_HOST,
        "HERMES_WEBUI_PORT": str(HERMES_SUB_PORT),
    }
    log_fh = open(SUBSERVER_LOG, "ab", buffering=0)
    return subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT)],
        cwd=str(SERVER_SCRIPT.parent),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def _subprocess_stop(proc: subprocess.Popen | None, timeout: float = 10.0) -> None:
    """SIGTERM the entire process group, then SIGKILL if needed."""
    if proc is None or proc.poll() is not None:
        return
    pid = proc.pid
    if pid is None:
        return
    logger.info(f"Stopping subprocess group PGID={pid}")
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"Subprocess did not exit after {timeout}s, sending SIGKILL")
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.error(f"Subprocess unkillable after SIGKILL (pid={pid})")


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """TCP connect check for health probing."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def _wait_for_subserver(timeout: float = 60.0) -> bool:
    """Poll TCP + HTTP health until the subserver responds or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_port_open(HERMES_SUB_HOST, HERMES_SUB_PORT, timeout=0.5):
            try:
                async with httpx.AsyncClient() as check_client:
                    resp = await check_client.get(
                        f"http://{HERMES_SUB_HOST}:{HERMES_SUB_PORT}/health",
                        timeout=2.0,
                    )
                    if resp.status_code == 200:
                        logger.info(f"Subserver healthy on {HERMES_SUB_HOST}:{HERMES_SUB_PORT}")
                        return True
            except Exception:
                logger.debug("Health check failed, retrying...", exc_info=True)
        await asyncio.sleep(0.5)
    logger.error(f"Subserver did not become healthy within {timeout}s")
    return False


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _proc, _healthy, _client
    _proc = _subprocess_start()
    _healthy = await _wait_for_subserver(timeout=60.0)
    if _healthy:
        _client = httpx.AsyncClient(
            base_url=f"http://{HERMES_SUB_HOST}:{HERMES_SUB_PORT}",
            timeout=httpx.Timeout(300.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
        )
    else:
        logger.warning("BFF started but subserver is NOT healthy — proxy routes will return 503")
    yield
    # Shutdown: clean up chat subprocesses, then the hermes subserver
    for chat_proc in list(_active_chat_procs):
        if chat_proc.returncode is None:
            try:
                os.killpg(os.getpgid(chat_proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
    _active_chat_procs.clear()
    if _client:
        await _client.aclose()
    _subprocess_stop(_proc)
    _healthy = False


app = FastAPI(title="aiOS", lifespan=lifespan)

# CORS — allow browser access from LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health endpoint ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    subserver_info = {
        "healthy": _healthy,
        "host": HERMES_SUB_HOST,
        "port": HERMES_SUB_PORT,
        "pid": _proc.pid if _proc else None,
    }
    return {"status": "ok", "subserver": subserver_info}


# ── Reverse proxy constants ───────────────────────────────────────────────────
PROXIED_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]


# ── aiOS Launcher (root) ─────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>aiOS</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: ui-monospace, 'Cascadia Code', 'Fira Code', monospace;
    background: #0d1117;
    color: #c9d1d9;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  /* Animated starfield background */
  .stars {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 0;
  }
  .star {
    position: absolute;
    background: #c9d1d9;
    border-radius: 50%;
    animation: twinkle var(--dur) ease-in-out infinite;
    animation-delay: var(--delay);
    opacity: 0;
  }
  @keyframes twinkle {
    0%, 100% { opacity: 0.1; }
    50% { opacity: 0.8; }
  }
  /* Scan line effect */
  .scanlines {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 1;
    background: repeating-linear-gradient(
      0deg,
      rgba(0,0,0,0.06) 0px,
      rgba(0,0,0,0.06) 2px,
      transparent 2px,
      transparent 4px
    );
  }
  /* Main content */
  .main {
    position: relative; z-index: 2;
    display: flex; flex-direction: column;
    align-items: center; gap: 2rem;
  }
  /* Title */
  .title-wrap { text-align: center; }
  .title-icon {
    font-size: 80px;
    line-height: 1;
    animation: float 4s ease-in-out infinite;
    text-shadow: 0 0 40px rgba(88,166,255,0.4), 0 0 80px rgba(88,166,255,0.2);
  }
  @keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
  }
  .title-text {
    font-size: 72px;
    font-weight: 900;
    letter-spacing: 8px;
    background: linear-gradient(135deg, #58a6ff 0%, #f0883e 50%, #3fb950 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    filter: drop-shadow(0 0 30px rgba(88,166,255,0.3));
  }
  .subtitle {
    font-size: 14px;
    color: #484f58;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 8px;
  }
  /* Launcher buttons */
  .launcher {
    display: flex; gap: 24px;
    flex-wrap: wrap; justify-content: center;
  }
  .launch-card {
    display: flex; flex-direction: column;
    align-items: center; gap: 12px;
    padding: 28px 36px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    cursor: pointer;
    text-decoration: none;
    color: #c9d1d9;
    transition: all 0.25s ease;
    min-width: 200px;
    position: relative;
    overflow: hidden;
  }
  .launch-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent);
    opacity: 0; transition: opacity 0.25s;
  }
  .launch-card:hover::before { opacity: 1; }
  .launch-card:hover {
    border-color: var(--accent);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
  }
  .launch-card .icon { font-size: 36px; }
  .launch-card .label { font-size: 16px; font-weight: 600; letter-spacing: 1px; }
  .launch-card .desc { font-size: 11px; color: #8b949e; text-align: center; }
  /* Card accent colors */
  .card-spa    { --accent: #58a6ff; }
  .card-memory { --accent: #3fb950; }
  .card-files  { --accent: #f0883e; }
  /* Footer */
  .footer {
    position: relative; z-index: 2;
    margin-top: 3rem;
    font-size: 11px; color: #30363d;
    display: flex; gap: 16px;
  }
  .footer .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #3fb950; }
  .footer .dot.dead { background: #f85149; }
</style>
</head>
<body>
<div class="stars" id="stars"></div>
<div class="scanlines"></div>

<div class="main">
  <div class="title-wrap">
    <div class="title-icon">&#9874;</div>
    <div class="title-text">aiOS</div>
    <div class="subtitle">Agent Intelligence Operating System</div>
  </div>

  <div class="launcher">
    <a class="launch-card card-spa" href="/hermes" target="_blank">
      <span class="icon">&#9874;</span>
      <span class="label">Hermes</span>
      <span class="desc">Multi-agent chat workspace</span>
    </a>
    <a class="launch-card card-memory" id="lnk-agentmemory" target="_blank">
      <span class="icon">&#9883;</span>
      <span class="label">AgentMemory</span>
      <span class="desc">Persistent memory &amp; actions</span>
    </a>
    <a class="launch-card card-files" id="lnk-files" target="_blank">
      <span class="icon">&#128193;</span>
      <span class="label">File Browser</span>
      <span class="desc">Browse, chmod, watch &amp; edit</span>
    </a>
  </div>

  <div class="footer">
    <span class="dot" id="status-dot"></span>
    <span id="status-text">checking...</span>
  </div>
</div>

<script>
// ── Starfield ──
(function(){
  var c = document.getElementById('stars');
  for (var i = 0; i < 80; i++) {
    var s = document.createElement('div');
    s.className = 'star';
    var size = Math.random() * 2 + 1;
    s.style.cssText =
      'left:' + (Math.random()*100) + '%;' +
      'top:' + (Math.random()*100) + '%;' +
      'width:' + size + 'px;' +
      'height:' + size + 'px;' +
      '--dur:' + (Math.random()*3+2) + 's;' +
      '--delay:' + (Math.random()*3) + 's;';
    c.appendChild(s);
  }
})();

// ── Quick Access links ──
(function(){
  var host = window.location.hostname || '127.0.0.1';
  var am = document.getElementById('lnk-agentmemory');
  if (am) am.href = 'http://' + host + ':3113';
  var fb = document.getElementById('lnk-files');
  if (fb) fb.href = 'http://' + host + ':8501/files';
})();

// ── Health check ──
async function checkHealth() {
  var dot = document.getElementById('status-dot');
  var txt = document.getElementById('status-text');
  try {
    var r = await fetch('/health');
    var d = await r.json();
    if (d.subserver && d.subserver.healthy) {
      dot.classList.remove('dead');
      txt.textContent = 'subserver ok (pid ' + d.subserver.pid + ') | :' + d.subserver.port;
    } else {
      dot.classList.add('dead');
      txt.textContent = 'subserver not ready';
    }
  } catch(e) {
    dot.classList.add('dead');
    txt.textContent = 'health check error';
  }
}
checkHealth();
setInterval(checkHealth, 10000);
</script>
</body>
</html>"""


@app.get("/")
async def dashboard():
    return HTMLResponse(
        content=DASHBOARD_HTML,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ── Hermes SPA redirect ───────────────────────────────────────────────────────
# /hermes → 302 redirect to /_hermes_spa so the SPA loads directly in the
# browser tab rather than inside an iframe. Iframes break the SPA's Service
# Worker, cross-frame API calls, and cause "refused to connect" errors.

@app.get("/hermes")
async def hermes_spa_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/_hermes_spa", status_code=302)


# ── Hermes iframe proxy (strips /_hermes_spa prefix → subserver root) ──────────────
@app.get("/_hermes_spa")
async def proxy_hermes_root(request: Request):
    """Proxy /hermes → subserver / to serve the SPA shell in the iframe."""
    if not _healthy or _client is None:
        raise HTTPException(status_code=503, detail="Hermes WebUI not ready")
    try:
        resp = await _client.get("/", headers={
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "transfer-encoding")
        })
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Subserver unreachable: {e}")
    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("transfer-encoding", "content-encoding")
    }
    try:
        raw_body = await resp.aread()
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Failed to read subserver response")
    resp_headers.pop("content-length", None)
    return Response(content=raw_body, status_code=resp.status_code, headers=resp_headers,
                    media_type=resp.headers.get("content-type", "text/html"))


@app.api_route("/_hermes_spa/{path:path}", methods=PROXIED_METHODS)
async def proxy_hermes_path(request: Request, path: str):
    """Proxy /_hermes_spa/* → subserver /* for iframe static assets and API calls."""
    if not _healthy or _client is None:
        raise HTTPException(status_code=503, detail="Hermes WebUI not ready")
    url = f"/{path}" if path else "/"
    if request.url.query:
        url = f"{url}?{request.url.query.decode() if isinstance(request.url.query, bytes) else request.url.query}"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "transfer-encoding")
    }
    original_host = request.headers.get("host", "")
    if original_host:
        headers["X-Forwarded-Host"] = original_host
    body = await request.body()
    from starlette.background import BackgroundTask
    try:
        upstream_method = "GET" if request.method == "HEAD" else request.method
        req = _client.build_request(method=upstream_method, url=url, headers=headers, content=body)
        resp = await _client.send(req, stream=True, follow_redirects=False)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Subserver unreachable: {e}")
    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("transfer-encoding", "content-encoding")
    }
    if request.method == "HEAD":
        await resp.aclose()
        resp_headers.pop("content-length", None)
        return Response(content=b"", status_code=resp.status_code, headers=resp_headers,
                        media_type=resp.headers.get("content-type", ""))
    content_length = resp.headers.get("content-length")
    if content_length is not None:
        try:
            raw_body = await resp.aread()
        except httpx.RequestError:
            raise HTTPException(status_code=502, detail="Failed to read subserver response")
        resp_headers.pop("content-length", None)
        return Response(content=raw_body, status_code=resp.status_code, headers=resp_headers,
                        media_type=resp.headers.get("content-type", ""))
    else:
        resp_headers.pop("content-length", None)
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code,
                                 headers=resp_headers,
                                 media_type=resp.headers.get("content-type", ""),
                                 background=BackgroundTask(resp.aclose))


# ── File Browser API ─────────────────────────────────────────────────────────
import stat as _stat
import datetime as _datetime

_FILE_BROWSER_ROOT = os.environ.get("FILE_BROWSER_ROOT", "/")
_FILE_BROWSER_DENY = {"/proc", "/sys", "/dev", "/run", "/etc/shadow", "/etc/passwd"}

# ── AI Permissions config ────────────────────────────────────────────────────
# Stores path → level mappings in a JSON file under the browsable root so
# the host can manage it through the file-browser UI.  Each entry controls
# what AI agents inside Docker are allowed to do with that path.
_PERMISSIONS_CONFIG = os.environ.get(
    "FILE_BROWSER_PERMISSIONS_CONFIG",
    "/tmp/aios-permissions.json",
)

_PERMISSION_MODES = {
    "rw":   {"file": "644", "dir": "755"},
    "ro":   {"file": "444", "dir": "555"},
    "none": {"file": "600", "dir": "700"},
}


def _load_ai_permissions() -> dict[str, str]:
    """Load AI permission config → {path_prefix: rw|ro|none}."""
    try:
        cfg = Path(_PERMISSIONS_CONFIG)
        if cfg.exists():
            return json.loads(cfg.read_text())
    except Exception:
        pass
    return {}


def _save_ai_permissions(data: dict[str, str]) -> None:
    """Persist AI permission config to disk."""
    Path(_PERMISSIONS_CONFIG).parent.mkdir(parents=True, exist_ok=True)
    Path(_PERMISSIONS_CONFIG).write_text(json.dumps(data, indent=2))


def _get_ai_permission_level(abs_path: str) -> str:
    """Return 'rw', 'ro', or 'none' for a path (longest-prefix match).

    Falls back to 'rw' when no entry matches.
    """
    perms = _load_ai_permissions()
    best_match = "rw"
    best_len = 0
    for prefix, level in perms.items():
        if abs_path == prefix or abs_path.startswith(prefix.rstrip("/") + "/"):
            if len(prefix) > best_len:
                best_match = level
                best_len = len(prefix)
    return best_match


def _check_ai_permission(target: Path, allow: list[str]) -> None:
    """Raise HTTP 403 if the path's AI permission level is not in *allow*."""
    level = _get_ai_permission_level(str(target))
    if level not in allow:
        raise HTTPException(
            status_code=403,
            detail=f"AI permission '{level}' — operation not allowed",
        )


def _resolve_safe_path(rel: str) -> Path:
    """Resolve and jail a relative path to FILE_BROWSER_ROOT."""
    root = Path(_FILE_BROWSER_ROOT).resolve()
    rel_path = rel.lstrip("/")
    # If the client sends an absolute path under the root, strip the root prefix
    root_str = str(root)
    if rel_path.startswith(root_str.lstrip("/")):
        rel_path = rel_path[len(root_str.lstrip("/")):].lstrip("/")
    target = (root / rel_path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path outside file browser root")
    if str(target) in _FILE_BROWSER_DENY or any(
        str(target).startswith(d + "/") for d in _FILE_BROWSER_DENY
    ):
        raise HTTPException(status_code=403, detail="Path is blocked")
    return target


@app.get("/api/files/list")
async def files_list(path: str = ""):
    """List directory contents."""
    target = _resolve_safe_path(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")

    entries = []
    try:
        for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            try:
                st = entry.stat()
            except PermissionError:
                entries.append({
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": None,
                    "modified": None,
                    "mode": None,
                    "mode_octal": None,
                    "unreadable": True,
                })
                continue
            abs_path = str(target / entry.name)
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size": st.st_size,
                "modified": _datetime.datetime.fromtimestamp(
                    st.st_mtime, tz=_datetime.timezone.utc
                ).isoformat(),
                "mode": _stat.filemode(st.st_mode),
                "mode_octal": oct(st.st_mode)[-3:],
                "uid": st.st_uid,
                "gid": st.st_gid,
                "ai_level": _get_ai_permission_level(abs_path),
            })
    except PermissionError:
        raise HTTPException(status_code=403, detail="Cannot read directory")

    # ── Filter out entries the AI is not allowed to see ──
    entries = [e for e in entries if e.get("ai_level") != "none"]

    return {"path": str(target), "entries": entries}


@app.get("/api/files/read")
async def files_read(path: str = ""):
    """Read file content."""
    target = _resolve_safe_path(path)
    _check_ai_permission(target, allow=["rw", "ro"])
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    try:
        content = target.read_text()
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Binary file — cannot display as text")
    return {"path": str(target), "content": content, "size": target.stat().st_size}


@app.post("/api/files/write")
async def files_write(request: Request):
    """Write file content."""
    body = await request.json()
    rel = body.get("path", "")
    content = body.get("content", "")
    target = _resolve_safe_path(rel)
    _check_ai_permission(target, allow=["rw"])
    try:
        target.write_text(content)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    return {"path": str(target), "written": len(content)}


@app.post("/api/files/chmod")
async def files_chmod(request: Request):
    """Change file permissions. Mode is an octal string like '644' or '755'."""
    body = await request.json()
    rel = body.get("path", "")
    mode_str = body.get("mode", "")
    if not mode_str or not all(c in "01234567" for c in mode_str) or len(mode_str) != 3:
        raise HTTPException(status_code=400, detail="Invalid mode — must be 3-digit octal (e.g. 644)")

    target = _resolve_safe_path(rel)
    _check_ai_permission(target, allow=["rw"])
    try:
        new_mode = int(mode_str, 8)
        os.chmod(target, new_mode)
        st = target.stat()
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except OSError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "path": str(target),
        "mode": _stat.filemode(st.st_mode),
        "mode_octal": oct(st.st_mode)[-3:],
    }


@app.post("/api/files/copy")
async def files_copy(request: Request):
    """Copy a file or directory. {src, dst}"""
    import shutil as _shutil
    body = await request.json()
    src = _resolve_safe_path(body.get("src", ""))
    dst = _resolve_safe_path(body.get("dst", ""))
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source not found")
    _check_ai_permission(src, allow=["rw"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_dir():
        dst = dst / src.name
    # Avoid clobbering: suffix if dst already exists
    stem, suffix = dst.stem, dst.suffix
    counter = 1
    while dst.exists():
        dst = dst.parent / f"{stem} (copy {counter}){suffix}"
        counter += 1
    try:
        if src.is_dir():
            _shutil.copytree(str(src), str(dst))
        else:
            _shutil.copy2(str(src), str(dst))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except OSError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"src": str(src), "dst": str(dst)}


@app.post("/api/files/move")
async def files_move(request: Request):
    """Move/rename a file or directory. {src, dst} — both relative to FILE_BROWSER_ROOT."""
    import shutil as _shutil
    body = await request.json()
    src = _resolve_safe_path(body.get("src", ""))
    dst = _resolve_safe_path(body.get("dst", ""))
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source not found")
    _check_ai_permission(src, allow=["rw"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    # If dst is a directory, move src inside it
    if dst.is_dir():
        dst = dst / src.name
    try:
        _shutil.move(str(src), str(dst))
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except OSError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"src": str(src), "dst": str(dst)}


@app.post("/api/files/delete")
async def files_delete(request: Request):
    """Permanently delete a file or directory."""
    import shutil as _shutil
    body = await request.json()
    target = _resolve_safe_path(body.get("path", ""))
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    _check_ai_permission(target, allow=["rw"])
    try:
        if target.is_dir():
            _shutil.rmtree(str(target))
        else:
            target.unlink()
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except OSError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"deleted": str(target)}


@app.get("/api/files/watch")
async def files_watch(path: str = ""):
    """SSE stream that tails a file. Emits new lines as they appear."""
    from asyncio import sleep as _async_sleep

    target = _resolve_safe_path(path)
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    async def _tail():
        yield "data: {\"status\": \"watching\", \"path\": \"" + str(target) + "\"}\n\n"
        try:
            with open(target, "r") as f:
                f.seek(0, 2)  # end of file
                while True:
                    line = f.readline()
                    if line:
                        yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
                    else:
                        await _async_sleep(0.5)
        except PermissionError:
            yield "data: {\"error\": \"Permission denied\"}\n\n"
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(
        _tail(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── AI Permissions Management API ────────────────────────────────────────────
@app.get("/api/files/permissions")
async def files_permissions_get(path: str = ""):
    """Get AI permissions config. Pass ?path= to query a single path."""
    if path:
        target = _resolve_safe_path(path)
        level = _get_ai_permission_level(str(target))
        return {"path": str(target), "level": level, "mode": _PERMISSION_MODES.get(level)}
    return _load_ai_permissions()


@app.post("/api/files/permissions/set")
async def files_permissions_set(request: Request):
    """Set AI permission level for a path + apply matching chmod.

    Body: {path, level, type}  — level ∈ {rw, ro, none}, type ∈ {file, dir}
    """
    body = await request.json()
    rel = body.get("path", "")
    level = body.get("level", "rw")
    entry_type = body.get("type", "file")

    if level not in ("rw", "ro", "none"):
        raise HTTPException(status_code=400, detail="Invalid level: must be rw, ro, or none")
    if entry_type not in ("file", "dir"):
        raise HTTPException(status_code=400, detail="Invalid type: must be 'file' or 'dir'")

    target = _resolve_safe_path(rel)
    abs_path = str(target)

    # ── Persist permission level ──
    perms = _load_ai_permissions()
    perms[abs_path] = level
    _save_ai_permissions(perms)

    # ── Apply filesystem mode and ownership directly (backend runs as root) ──
    mode_octal = _PERMISSION_MODES[level][entry_type]
    chmod_error = None
    chmod_applied = False

    if not target.exists():
        chmod_error = "target does not exist"
    else:
        try:
            # Change file/folder ownership to root or ai_user based on permission level:
            # - rw (read-write): owned by ai_user (UID 2000, GID 2000 by default)
            # - ro (read-only) or none (hidden): owned by root (UID 0, GID 0)
            import shutil
            target_uid = int(os.environ.get("USER_ID", "2000"))
            target_gid = int(os.environ.get("GROUP_ID", "2000"))

            if level == "rw":
                shutil.chown(target, user=target_uid, group=target_gid)
            else:
                shutil.chown(target, user=0, group=0)

            # Change filesystem permission mode
            os.chmod(target, int(mode_octal, 8))
            applied_mode = oct(target.stat().st_mode)[-3:]
            if applied_mode == mode_octal:
                chmod_applied = True
            else:
                chmod_error = f"chmod succeeded but mode is {applied_mode}, expected {mode_octal}"
        except OSError as e:
            chmod_error = str(e)

    return {
        "path": abs_path, "level": level, "mode": mode_octal,
        "chmod_applied": chmod_applied,
        "chmod_error": chmod_error,
    }


@app.post("/api/files/permissions/remove")
async def files_permissions_remove(request: Request):
    """Remove AI permission entry for a path and its children."""
    body = await request.json()
    rel = body.get("path", "")
    target = _resolve_safe_path(rel)
    abs_path = str(target)

    perms = _load_ai_permissions()
    to_remove = [k for k in perms if k == abs_path or k.startswith(abs_path.rstrip("/") + "/")]
    for k in to_remove:
        del perms[k]
    _save_ai_permissions(perms)

    return {"path": abs_path, "removed": to_remove}


# ── File Browser SPA ─────────────────────────────────────────────────────────
_FILE_BROWSER_HTML_PATH = Path(__file__).resolve().parent / "static" / "file-browser" / "index.html"


def _load_file_browser_html() -> str:
    """Load the file browser SPA HTML, with a fallback for missing file."""
    try:
        return _FILE_BROWSER_HTML_PATH.read_text()
    except Exception:
        return "<html><body><h1>File browser not found</h1></body></html>"


@app.get("/files")
async def file_browser_spa():
    """Serve the file browser single-page app."""
    return HTMLResponse(
        content=_load_file_browser_html(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ── Reverse proxy (health-gated) ─────────────────────────────────────────────
# MUST be the LAST route registered — Starlette sorts exact paths first,
# but placing the catch-all last is defensive and clearer.


@app.api_route("/{path:path}", methods=PROXIED_METHODS)
async def proxy_to_hermes(request: Request, path: str):
    """Forward all unmatched requests to the hermes-webui subserver."""
    if not _healthy or _client is None:
        raise HTTPException(status_code=503, detail="Hermes WebUI not ready")
    url = f"/{path}" if path else "/"
    if request.url.query:
        url = f"{url}?{request.url.query.decode() if isinstance(request.url.query, bytes) else request.url.query}"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "transfer-encoding")
    }
    # Preserve original Host for CSRF same-origin checks in the subserver
    original_host = request.headers.get("host", "")
    if original_host:
        headers["X-Forwarded-Host"] = original_host
    body = await request.body()
    try:
        from starlette.background import BackgroundTask
        # The sync subserver doesn't support HEAD; convert to GET internally
        # but return a HEAD-style response (no body) to the client.
        upstream_method = "GET" if request.method == "HEAD" else request.method
        req = _client.build_request(
            method=upstream_method,
            url=url,
            headers=headers,
            content=body,
        )
        resp = await _client.send(req, stream=True, follow_redirects=False)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Subserver unreachable: {e}")

    # Build response headers — preserve subserver headers except hop-by-hop.
    # IMPORTANT: httpx auto-decompresses the body, so we MUST strip
    # content-encoding to prevent browsers from trying to gunzip plaintext.
    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("transfer-encoding", "content-encoding")
    }

    # Determine streaming vs buffered: SSE/text-event-stream responses have no
    # Content-Length and must be streamed. Static assets (CSS/JS/images) have
    # Content-Length and should be buffered to avoid TaskGroup/ClientDisconnected
    # exceptions corrupting the response body.
    content_length = resp.headers.get("content-length")
    content_type = resp.headers.get("content-type", "")

    if request.method == "HEAD":
        await resp.aclose()
        # For HEAD, return headers only (no body) per HTTP spec
        resp_headers.pop("content-length", None)
        return Response(
            content=b"",
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=content_type,
        )

    if content_length is not None:
        # Buffered — read full body, return plain Response (no TaskGroup risk).
        # IMPORTANT: httpx auto-decompresses the response body, so the subserver's
        # Content-Length may not match the decompressed size. Strip Content-Length
        # and let Starlette/Uvicorn set the correct one from the actual body.
        try:
            raw_body = await resp.aread()
        except httpx.RequestError:
            raise HTTPException(status_code=502, detail="Failed to read subserver response")
        resp_headers.pop("content-length", None)
        return Response(
            content=raw_body,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=content_type,
        )
    else:
        # Streaming — SSE, long-polling, or other chunked responses
        resp_headers.pop("content-length", None)  # double-ensure no stale CL
        return StreamingResponse(
            resp.aiter_bytes(),
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=content_type,
            background=BackgroundTask(resp.aclose),
        )


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting BFF on {SIDECAR_HOST}:{SIDECAR_PORT}")
    logger.info(f"Hermes subserver target: {HERMES_SUB_HOST}:{HERMES_SUB_PORT}")
    logger.info(f"HERMES_HOME: {HERMES_HOME}")

    uvicorn.run(
        "main:app",
        host=SIDECAR_HOST,
        port=SIDECAR_PORT,
        log_level="info",
        reload=False,
    )
