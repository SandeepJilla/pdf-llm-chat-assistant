"""
PDF Chat Assistant with free OpenRouter models + PDF processing — Flask Backend 

Usage:
    Replace XXXXX_API_KEY_XXXXX with your API key from openrouter.ai
    python pdf_chat_app.py
    Open browser to http://localhost:5000
"""

import os
import re
import json
import time
from pathlib import Path
from flask import Flask, request, jsonify, Response, make_response, stream_with_context
import requests
from io import BytesIO

# ── Configuration ──────────────────────────────────────────────────────────────
# OpenRouter settings
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "XXXXX_API_KEY_XXXXX")

# Fallback models list
_FALLBACK_MODELS = [
    {"id": "meta-llama/llama-3.2-3b-instruct:free",  "label": "Llama 3.2 3B (free)"},
    {"id": "meta-llama/llama-3.2-1b-instruct:free",  "label": "Llama 3.2 1B (free)"},
    {"id": "google/gemma-2-9b-it:free",              "label": "Gemma 2 9B (free)"},
    {"id": "qwen/qwen-2-7b-instruct:free",           "label": "Qwen 2 7B (free)"},
]
FREE_MODELS = _FALLBACK_MODELS
MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_FILE_BYTES = 10 * 1024 * 1024
ALLOWED_EXT = {".pdf", ".txt", ".csv", ".json", ".md"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

pdf_storage = {}

# ── OpenRouter Helpers ─────────────────────────────────────────────────────────
def get_headers():
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "PDF Chat Assistant",
    }


def call_openrouter(payload, max_retries=3):
    """POST to OpenRouter with retry logic"""
    base_model = payload.get("model", MODEL)
    candidates = [base_model]
    
    # Add fallback models
    for m in FREE_MODELS:
        if m["id"] != base_model and m["id"] not in candidates:
            candidates.append(m["id"])
    
    tried = set()
    for candidate in candidates:
        if candidate in tried:
            continue
        tried.add(candidate)
        payload["model"] = candidate
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    OPENROUTER_URL,
                    headers=get_headers(),
                    json=payload,
                    timeout=120
                )
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None, candidate, "Request timed out"
            except Exception as e:
                return None, candidate, str(e)
            
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                retry_after = min(retry_after, 10)
                if attempt < max_retries - 1:
                    time.sleep(retry_after)
                    continue
                else:
                    break
            
            if not resp.ok:
                try:
                    err_body = resp.json()
                    err_msg = err_body.get("error", {}).get("message", "") or str(err_body)
                except Exception:
                    err_msg = resp.text or f"HTTP {resp.status_code}"
                if resp.status_code == 404:
                    break
                return None, candidate, f"{resp.status_code}: {err_msg}"
            
            data = resp.json()
            if "error" in data:
                return None, candidate, data["error"].get("message", str(data["error"]))
            
            return data, candidate, None
    
    return None, base_model, "All models rate-limited. Please wait and try again."


def fetch_free_models():
    """Fetch live free models from OpenRouter"""
    global FREE_MODELS
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=10
        )
        if not resp.ok:
            print(f"[models] Using fallback list (API {resp.status_code})")
            return
        
        data = resp.json().get("data", [])
        free = []
        for m in data:
            pricing = m.get("pricing", {})
            if str(pricing.get("prompt", "1")) != "0":
                continue
            mid = m.get("id", "")
            name = m.get("name", mid)
            ctx = m.get("context_length", 0)
            free.append({"id": mid, "label": f"{name} [{ctx//1000}k]", "ctx": ctx})
        
        if free:
            free.sort(key=lambda x: x["ctx"], reverse=True)
            FREE_MODELS = [{"id": f["id"], "label": f["label"]} for f in free]
            print(f"[models] Loaded {len(FREE_MODELS)} free models")
        else:
            print("[models] No free models found, using fallback")
    except Exception as e:
        print(f"[models] Fetch failed: {e}")


# ── PDF Processing ─────────────────────────────────────────────────────────────
def parse_pdf(raw_bytes: bytes) -> str:
    """Extract text from PDF"""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(BytesIO(raw_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"--- Page {i+1} ---\n{text.strip()}")
        
        if not pages:
            return "[PDF contains no extractable text]"
        
        full = "\n\n".join(pages)
        words = full.split()
        if len(words) > 10000:
            full = " ".join(words[:10000]) + "\n\n[PDF TRUNCATED - showing first 10000 words]"
        return full
    except ImportError:
        return "[PyPDF2 not installed. Run: pip install PyPDF2]"
    except Exception as e:
        return f"[Could not extract PDF: {str(e)}]"


def parse_uploaded_file(file) -> tuple:
    """Parse uploaded file and return (filename, content)"""
    filename = file.filename
    ext = Path(filename).suffix.lower()
    
    if ext not in ALLOWED_EXT:
        return filename, f"[Unsupported file type: {ext}]"
    
    raw = file.read(MAX_FILE_BYTES)
    
    if ext == ".pdf":
        return filename, parse_pdf(raw)
    
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return filename, "[Could not decode file]"
    
    if len(raw) == MAX_FILE_BYTES:
        text += "\n\n[FILE TRUNCATED]"
    
    return filename, text


# ── Conversational Detection ───────────────────────────────────────────────────
CONVERSATIONAL_PATTERNS = re.compile(
    r"^\s*(hi+|hello+|hey+|thanks?|thank\s*you|ok+ay?|cool|nice|bye|"
    r"what\s*can\s*you\s*do|help|who\s*are\s*you)\s*[!?.]*\s*$",
    re.IGNORECASE
)


# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful PDF document analyst assistant.

When users upload PDF files, you can:
1. Summarize the content
2. Answer questions about specific information in the PDFs
3. Extract key points, dates, names, numbers
4. Compare information across multiple PDFs
5. Search for specific topics or keywords

IMPORTANT RULES:
- Be concise but thorough in your answers
- Quote relevant sections from the PDFs when answering
- If information is not in the uploaded PDFs, clearly state that
- For greetings and casual conversation, respond naturally without referring to PDFs
- Use bullet points and formatting to make answers clear
- Include page numbers when referencing specific parts of PDFs

When analyzing PDFs:
- Focus on the most relevant information to the user's question
- Provide context around quotes and excerpts
- Organize information logically
- Highlight important numbers, dates, and names
"""


# ── HTML Template ──────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>PDF Chat Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg: #0a0e17; --surface: #0f141f; --panel: #151b2b;
  --border: #1a2332; --border2: #243447;
  --primary: #6366f1; --primary-dim: #4f46e5; --primary-bright: #818cf8;
  --secondary: #06b6d4; --accent: #f59e0b;
  --text: #e2e8f0; --text-dim: #94a3b8; --text-mute: #475569;
  --success: #10b981; --error: #ef4444;
  --mono: 'Space Mono', monospace; --sans: 'Inter', sans-serif;
  --radius: 12px; --shadow: 0 4px 6px rgba(0,0,0,.3);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.6; }
body::before {
  content: ''; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background: radial-gradient(ellipse 800px 600px at 30% 20%, rgba(99,102,241,.08), transparent),
              radial-gradient(ellipse 600px 500px at 80% 80%, rgba(6,182,212,.06), transparent);
}
#app { position: relative; z-index: 1; display: grid; grid-template-columns: 320px 1fr; grid-template-rows: 64px 1fr; height: 100vh; }

/* Header */
header { grid-column: 1/-1; display: flex; align-items: center; gap: 16px; padding: 0 24px; background: rgba(15,20,31,.95); border-bottom: 1px solid var(--border); backdrop-filter: blur(12px); }
.logo { display: flex; align-items: center; gap: 12px; }
.logo-icon { width: 36px; height: 36px; background: linear-gradient(135deg, var(--primary), var(--secondary)); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px; }
.logo-text h1 { font-size: 18px; font-weight: 700; color: var(--text); }
.logo-text p { font-size: 11px; color: var(--text-dim); }
.status-bar { margin-left: auto; display: flex; align-items: center; gap: 20px; }
.status-chip { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--text-dim); font-family: var(--mono); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-mute); }
.status-dot.ok { background: var(--success); box-shadow: 0 0 8px var(--success); }
.status-dot.fail { background: var(--error); box-shadow: 0 0 8px var(--error); }
#model-select {
  background: var(--panel); border: 1px solid var(--border2); border-radius: 8px;
  color: var(--text); font-family: var(--mono); font-size: 11px;
  padding: 6px 32px 6px 12px; cursor: pointer; outline: none;
  appearance: none; -webkit-appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M0 0l6 8 6-8z' fill='%236366f1'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: right 10px center;
  transition: border-color .2s;
}
#model-select:hover { border-color: var(--primary); }

/* Sidebar */
aside { background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
.sidebar-section { padding: 20px; border-bottom: 1px solid var(--border); }
.sidebar-label { font-size: 11px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; color: var(--text-mute); margin-bottom: 12px; }

/* Drop Zone */
#drop-zone { border: 2px dashed var(--border2); border-radius: var(--radius); padding: 24px 16px; text-align: center; cursor: pointer; transition: all .2s; position: relative; background: var(--panel); }
#drop-zone:hover, #drop-zone.drag-over { border-color: var(--primary); background: rgba(99,102,241,.05); }
#drop-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.drop-icon { font-size: 32px; margin-bottom: 8px; color: var(--text-dim); }
.drop-text { font-size: 12px; color: var(--text-dim); line-height: 1.5; }
.drop-text strong { color: var(--primary-bright); }

/* File List */
#file-list { margin-top: 12px; display: flex; flex-direction: column; gap: 8px; max-height: 200px; overflow-y: auto; }
.file-item { display: flex; align-items: center; gap: 10px; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; animation: slideIn .2s ease; }
@keyframes slideIn { from { opacity: 0; transform: translateY(-4px); } }
.file-icon { font-size: 16px; flex-shrink: 0; }
.file-name { font-size: 12px; color: var(--text); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-size { font-family: var(--mono); font-size: 10px; color: var(--text-dim); }
.file-remove { width: 20px; height: 20px; border-radius: 50%; background: none; border: none; cursor: pointer; color: var(--text-dim); font-size: 16px; display: flex; align-items: center; justify-content: center; transition: all .2s; }
.file-remove:hover { color: var(--error); background: rgba(239,68,68,.1); }

/* Suggestions */
.suggestions { padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; }
.suggest-btn { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; text-align: left; cursor: pointer; font-size: 12px; color: var(--text-dim); transition: all .2s; line-height: 1.5; }
.suggest-btn:hover { border-color: var(--primary); color: var(--text); background: rgba(99,102,241,.05); transform: translateX(2px); }
.suggest-icon { display: inline-block; margin-right: 8px; color: var(--primary-bright); font-size: 14px; }

/* Sidebar Info */
.sidebar-info { margin-top: auto; padding: 16px 20px; border-top: 1px solid var(--border); font-family: var(--mono); font-size: 10px; color: var(--text-mute); line-height: 1.8; }
.sidebar-info span { color: var(--text-dim); font-weight: 600; }

/* Main Content */
main { display: flex; flex-direction: column; overflow: hidden; }
#messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; scroll-behavior: smooth; }
#messages::-webkit-scrollbar { width: 6px; }
#messages::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

/* Welcome Screen */
#welcome { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 16px; text-align: center; padding: 40px; animation: fadeIn .6s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } }
.welcome-icon { font-size: 64px; margin-bottom: 8px; }
.welcome-title { font-size: 24px; font-weight: 700; color: var(--text); margin-bottom: 8px; }
.welcome-sub { font-size: 14px; color: var(--text-dim); max-width: 480px; line-height: 1.6; }

/* Messages */
.msg { display: flex; gap: 12px; max-width: 900px; animation: fadeIn .3s ease; }
.msg.user { flex-direction: row-reverse; align-self: flex-end; }
.msg.assistant { align-self: flex-start; }
.msg-avatar { width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; margin-top: 2px; }
.msg.user .msg-avatar { background: linear-gradient(135deg, var(--primary-dim), var(--primary)); }
.msg.assistant .msg-avatar { background: var(--panel); border: 1px solid var(--border2); }
.msg-body { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.msg-bubble { padding: 14px 18px; border-radius: var(--radius); font-size: 14px; line-height: 1.7; word-break: break-word; }
.msg.user .msg-bubble { background: linear-gradient(135deg, var(--primary-dim), var(--primary)); color: #f8fafc; border-bottom-right-radius: 4px; box-shadow: var(--shadow); }
.msg.assistant .msg-bubble { background: var(--panel); border: 1px solid var(--border); color: var(--text); border-bottom-left-radius: 4px; }

/* Message Formatting */
.msg-bubble code { font-family: var(--mono); font-size: 12px; background: rgba(99,102,241,.15); color: var(--primary-bright); padding: 2px 6px; border-radius: 4px; }
.msg-bubble pre { background: rgba(0,0,0,.5); border: 1px solid var(--border2); border-radius: 8px; padding: 14px 16px; overflow-x: auto; margin: 10px 0; }
.msg-bubble pre code { background: none; padding: 0; color: #8dd4f7; font-size: 12px; }
.msg-bubble strong { color: var(--primary-bright); font-weight: 600; }
.msg-bubble em { color: var(--accent); font-style: italic; }
.msg-bubble p + p { margin-top: 10px; }
.msg-bubble ul, .msg-bubble ol { padding-left: 20px; margin: 8px 0; }
.msg-bubble li { margin-bottom: 4px; }
.msg-bubble h3 { font-size: 13px; font-weight: 600; color: var(--primary-bright); margin: 12px 0 6px; }

/* Attached Files Chips */
.attached-files { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
.attach-chip { background: rgba(99,102,241,.15); border: 1px solid rgba(99,102,241,.3); border-radius: 8px; padding: 4px 10px; font-size: 11px; color: var(--primary-bright); font-family: var(--mono); }

/* Typing Animation */
.typing-dots { display: flex; gap: 6px; align-items: center; padding: 16px 18px; }
.typing-dots span { width: 8px; height: 8px; border-radius: 50%; background: var(--primary); animation: blink 1.2s infinite; }
.typing-dots span:nth-child(2) { animation-delay: .2s; }
.typing-dots span:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%,80%,100% { opacity:.3; transform: scale(.8); } 40% { opacity:1; transform: scale(1); } }

/* Input Bar */
#input-bar { padding: 16px 24px; background: rgba(15,20,31,.98); border-top: 1px solid var(--border); backdrop-filter: blur(12px); }
.input-wrapper { display: flex; align-items: flex-end; gap: 12px; background: var(--panel); border: 2px solid var(--border2); border-radius: var(--radius); padding: 12px 14px; transition: border-color .2s; }
.input-wrapper:focus-within { border-color: var(--primary); }
#msg-input { flex: 1; background: none; border: none; outline: none; color: var(--text); font-family: var(--sans); font-size: 14px; resize: none; max-height: 150px; line-height: 1.6; scrollbar-width: none; }
#msg-input::placeholder { color: var(--text-mute); }
#msg-input::-webkit-scrollbar { display: none; }
.send-btn { width: 40px; height: 40px; border-radius: 10px; flex-shrink: 0; background: linear-gradient(135deg, var(--primary-dim), var(--primary)); border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all .2s; color: white; box-shadow: var(--shadow); }
.send-btn:hover { background: linear-gradient(135deg, var(--primary), var(--primary-bright)); transform: translateY(-1px); box-shadow: 0 6px 12px rgba(0,0,0,.4); }
.send-btn:active { transform: scale(.95); }
.send-btn:disabled { background: var(--border2); cursor: not-allowed; box-shadow: none; }
.send-btn svg { width: 18px; height: 18px; }

/* Input Hint */
.input-hint { margin-top: 8px; font-size: 11px; color: var(--text-mute); display: flex; gap: 16px; padding: 0 2px; }
.input-hint kbd { font-family: var(--mono); font-size: 10px; background: var(--panel); border: 1px solid var(--border2); border-radius: 4px; padding: 2px 6px; color: var(--text-dim); }
.clear-btn { background: none; border: 1px solid var(--border); border-radius: 8px; color: var(--text-dim); font-size: 11px; padding: 4px 10px; cursor: pointer; font-family: var(--mono); transition: all .2s; margin-left: auto; }
.clear-btn:hover { border-color: var(--error); color: var(--error); }

/* Error Toast */
#error-toast { position: fixed; bottom: 90px; right: 24px; background: rgba(239,68,68,.15); border: 1px solid var(--error); border-radius: 10px; padding: 12px 16px; font-size: 13px; color: var(--error); display: none; z-index: 100; animation: fadeIn .3s ease; box-shadow: var(--shadow); }
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="logo">
      <div class="logo-icon">📄</div>
      <div class="logo-text">
        <h1>PDF Chat Assistant</h1>
        <p>Ask questions about your documents</p>
      </div>
    </div>
    <div class="status-bar">
      <div class="status-chip"><div class="status-dot" id="dot-api"></div><span>API</span></div>
      <select id="model-select" onchange="changeModel(this.value)"></select>
    </div>
  </header>

  <aside>
    <div class="sidebar-section">
      <div class="sidebar-label">Upload Documents</div>
      <div id="drop-zone">
        <input type="file" id="file-input" multiple accept=".pdf,.txt,.csv,.json,.md"/>
        <div class="drop-icon">📎</div>
        <div class="drop-text">Drop files or <strong>click to browse</strong><br>PDF • TXT • CSV • JSON • MD</div>
      </div>
      <div id="file-list"></div>
    </div>
    <div class="sidebar-label" style="padding:16px 20px 8px;">Quick Examples</div>
    <div class="suggestions">
      <button class="suggest-btn" onclick="suggest('Summarize the main points of this document')"><span class="suggest-icon">📋</span>Summarize the document</button>
      <button class="suggest-btn" onclick="suggest('What are the key dates mentioned?')"><span class="suggest-icon">📅</span>Extract key dates</button>
      <button class="suggest-btn" onclick="suggest('List all the names mentioned')"><span class="suggest-icon">👤</span>Find all names</button>
      <button class="suggest-btn" onclick="suggest('What are the main topics covered?')"><span class="suggest-icon">💡</span>Identify main topics</button>
      <button class="suggest-btn" onclick="suggest('Extract all numbers and statistics')"><span class="suggest-icon">📊</span>Get numbers & stats</button>
    </div>
    <div class="sidebar-info">
      Model: <span id="sidebar-model">loading...</span><br>
      Status: <span>Ready</span>
    </div>
  </aside>

  <main>
    <div id="messages">
      <div id="welcome">
        <div class="welcome-icon">📚</div>
        <div class="welcome-title">Welcome to PDF Chat</div>
        <div class="welcome-sub">Upload your PDF documents and ask questions. I can summarize content, extract information, find specific details, and help you understand your documents better.</div>
      </div>
    </div>
    <div id="input-bar">
      <div class="input-wrapper">
        <textarea id="msg-input" rows="1" placeholder="Ask a question about your PDFs..."></textarea>
        <button class="send-btn" id="send-btn" onclick="sendMessage()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
      <div class="input-hint">
        <span><kbd>Enter</kbd> send • <kbd>Shift+Enter</kbd> newline</span>
        <button class="clear-btn" onclick="clearChat()">✕ Clear chat</button>
      </div>
    </div>
  </main>
</div>
<div id="error-toast"></div>

<script>
let chatHistory = [], attachedFiles = [], isStreaming = false;

const input = document.getElementById('msg-input');
input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 150) + 'px'; });
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileListEl = document.getElementById('file-list');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag-over'); addFiles([...e.dataTransfer.files]); });
fileInput.addEventListener('change', () => { addFiles([...fileInput.files]); fileInput.value = ''; });

function addFiles(files) {
  const allowed = ['.pdf','.txt','.csv','.json','.md'];
  files.forEach(f => {
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) { showError('Unsupported: ' + f.name); return; }
    if (attachedFiles.some(af => af.name === f.name)) return;
    attachedFiles.push(f);
  });
  renderFileList();
}

function renderFileList() {
  fileListEl.innerHTML = '';
  attachedFiles.forEach((f, i) => {
    const size = f.size > 1048576 ? (f.size/1048576).toFixed(1)+'MB' : Math.round(f.size/1024)+'KB';
    const icons = {pdf:'📕', txt:'📄', csv:'📊', json:'📋', md:'📝'};
    const icon = icons[f.name.split('.').pop().toLowerCase()] || '📄';
    const div = document.createElement('div');
    div.className = 'file-item';
    div.innerHTML = '<span class="file-icon">'+icon+'</span><span class="file-name" title="'+esc(f.name)+'">'+esc(f.name)+'</span><span class="file-size">'+size+'</span><button class="file-remove" onclick="removeFile('+i+')">×</button>';
    fileListEl.appendChild(div);
  });
}
function removeFile(i) { attachedFiles.splice(i, 1); renderFileList(); }
function suggest(text) { input.value = text; input.dispatchEvent(new Event('input')); input.focus(); }

async function sendMessage() {
  if (isStreaming) return;
  const text = input.value.trim();
  if (!text && attachedFiles.length === 0) return;
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();
  const filesCopy = [...attachedFiles];
  appendUserMsg(text, filesCopy);
  attachedFiles = []; renderFileList();
  input.value = ''; input.style.height = 'auto';
  const typingEl = appendTyping();
  setStreaming(true);

  const fd = new FormData();
  fd.append('message', text);
  fd.append('history', JSON.stringify(chatHistory));
  filesCopy.forEach((f, i) => fd.append('file_'+i, f));

  try {
    const res = await fetch('/chat', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Server error ' + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    typingEl.remove();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let evt; try { evt = JSON.parse(line.slice(6)); } catch { continue; }
        if (evt.type === 'text') {
          appendAssistantMsg(evt.content);
        } else if (evt.type === 'done') {
          chatHistory = evt.history || [];
        } else if (evt.type === 'error') {
          showError(evt.content);
        }
      }
    }
  } catch(err) {
    typingEl?.remove();
    showError('Connection error: ' + err.message);
  }
  setStreaming(false); scrollBottom();
}

const msgs = document.getElementById('messages');
function scrollBottom() { msgs.scrollTop = msgs.scrollHeight; }

function appendUserMsg(text, files) {
  const div = document.createElement('div');
  div.className = 'msg user';
  const chips = files.length ? '<div class="attached-files">'+files.map(f=>'<span class="attach-chip">📎 '+esc(f.name)+'</span>').join('')+'</div>' : '';
  div.innerHTML = '<div class="msg-avatar">👤</div><div class="msg-body">'+chips+(text?'<div class="msg-bubble">'+esc(text)+'</div>':'')+'</div>';
  msgs.appendChild(div); scrollBottom();
  if (text) chatHistory.push({ role: 'user', content: text });
}

function appendTyping() {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-avatar">🤖</div><div class="msg-body"><div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div></div>';
  msgs.appendChild(div); scrollBottom(); return div;
}

function appendAssistantMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.innerHTML = '<div class="msg-avatar">🤖</div><div class="msg-body"><div class="msg-bubble">'+renderMarkdown(text)+'</div></div>';
  msgs.appendChild(div); scrollBottom(); return div;
}

function renderMarkdown(text) {
  let h = esc(text);
  h = h.replace(/```(\w*)\n([\s\S]*?)```/g, (_, l, code) => '<pre><code>'+code.trim()+'</code></pre>');
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  h = h.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  h = h.replace(/^[*-] (.+)$/gm, '<li>$1</li>');
  h = h.replace(/(<li>.*<\/li>(\n|$))+/g, m => '<ul>'+m+'</ul>');
  h = h.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  h = h.split(/\n\n+/).map(p => p.trim().startsWith('<') ? p : '<p>'+p.replace(/\n/g,'<br>')+'</p>').join('');
  return h;
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function setStreaming(val) { isStreaming = val; document.getElementById('send-btn').disabled = val; }
function clearChat() {
  chatHistory = []; attachedFiles = []; renderFileList();
  msgs.innerHTML = '<div id="welcome" style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:16px;text-align:center;padding:40px;"><div class="welcome-icon">📚</div><div class="welcome-title">Welcome to PDF Chat</div><div class="welcome-sub">Upload your PDF documents and ask questions.</div></div>';
}
function showError(msg) {
  const t = document.getElementById('error-toast');
  t.textContent = '⚠ ' + msg; t.style.display = 'block';
  setTimeout(() => { t.style.display = 'none'; }, 4000);
}

async function loadModels() {
  try {
    const d = await (await fetch('/models')).json();
    const sel = document.getElementById('model-select');
    sel.innerHTML = '';
    d.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id; opt.textContent = m.label;
      if (m.id === d.current) opt.selected = true;
      sel.appendChild(opt);
    });
    const cur = d.models.find(m => m.id === d.current);
    if (cur) document.getElementById('sidebar-model').textContent = cur.label;
  } catch(e) { console.warn('loadModels failed', e); }
}

async function changeModel(modelId) {
  try {
    const r = await fetch('/set-model', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: modelId})
    });
    const d = await r.json();
    if (d.ok) {
      const sel = document.getElementById('model-select');
      const label = sel.options[sel.selectedIndex].textContent;
      document.getElementById('sidebar-model').textContent = label;
    } else { showError(d.error || 'Could not switch model'); }
  } catch(e) { showError('Model switch failed: ' + e.message); }
}

async function checkHealth() {
  try {
    const d = await (await fetch('/health')).json();
    document.getElementById('dot-api').className = 'status-dot ' + (d.api_ok ? 'ok' : 'fail');
  } catch {}
}

loadModels();
checkHealth();
setInterval(checkHealth, 15000);
</script>
</body>
</html>"""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML_TEMPLATE


@app.route("/chat", methods=["POST"])
def chat():
    history_raw = request.form.get("history", "[]")
    user_message = request.form.get("message", "").strip()
    
    try:
        history = json.loads(history_raw)
    except Exception:
        history = []
    
    file_contexts = []
    for key in request.files:
        f = request.files[key]
        if f and f.filename:
            fname, ftext = parse_uploaded_file(f)
            file_contexts.append((fname, ftext))
            # Store in memory
            file_id = f"{fname}_{int(time.time())}"
            pdf_storage[file_id] = ftext
    
    content_parts = []
    if file_contexts:
        for fname, ftext in file_contexts:
            content_parts.append(f"[DOCUMENT: {fname}]\n{ftext}\n")
    if user_message:
        content_parts.append(user_message)
    
    full_user_content = "\n\n".join(content_parts)
    
    if not full_user_content:
        return jsonify({"error": "No message or files provided"}), 400
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": full_user_content})
    
    is_conversational = (
        len(file_contexts) == 0 and
        bool(CONVERSATIONAL_PATTERNS.match(user_message))
    )
    
    def generate():
        nonlocal messages
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2000,
        }
        
        data, used_model, err = call_openrouter(payload)
        if err:
            evt = json.dumps({"type": "error", "content": err})
            yield f"data: {evt}\n\n"
            return
        
        choice = data["choices"][0]
        msg = choice["message"]
        messages.append(msg)
        
        final_text = msg.get("content", "").strip()
        if final_text:
            yield f"data: {json.dumps({'type': 'text', 'content': final_text})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done', 'history': messages[1:]})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"}
    )


@app.route("/models")
def get_models():
    current = MODEL
    ids = [m["id"] for m in FREE_MODELS]
    if current not in ids and FREE_MODELS:
        current = FREE_MODELS[0]["id"]
    return jsonify({"models": FREE_MODELS, "current": current})


@app.route("/set-model", methods=["POST"])
def set_model():
    global MODEL
    data = request.get_json()
    new_model = data.get("model", "").strip()
    if not new_model:
        return jsonify({"error": "No model specified"}), 400
    MODEL = new_model
    return jsonify({"ok": True, "model": MODEL})


@app.route("/health")
def health():
    api_ok = bool(OPENROUTER_API_KEY and OPENROUTER_API_KEY != "XXXXX_API_KEY_XXXXX")
    return jsonify({
        "api_ok": api_ok,
        "model": MODEL,
        "status": "ok" if api_ok else "degraded"
    })


if __name__ == "__main__":
    fetch_free_models()
    print("=" * 60)
    print("  PDF Chat Assistant")
    print("=" * 60)
    print(f"  Model: {MODEL}")
    key_hint = OPENROUTER_API_KEY[:12] + "..." if len(OPENROUTER_API_KEY) > 12 else "(not set)"
    print(f"  API Key: {key_hint}")
    print(f"  URL: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000, threaded=True)
