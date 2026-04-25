"""
NutriBot Kendari — Streamlit App (Standalone, tanpa FastAPI)
Langsung memanggil generate_response() dari rag_pipeline/generation.py
"""
import os
import sys
import time
import streamlit as st

# ── PATH SETUP ───────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── API KEY SETUP ─────────────────────────────────────────────────────────────
# Prioritas: st.secrets (Streamlit Cloud) → os.getenv (lokal / .env)
def _get_secret(key: str):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

# Inject dulu ke os.environ SEBELUM generation.py diimport,
# sehingga get_bot() di dalam generation.py langsung bisa membacanya.
_GEMINI_KEY = _get_secret("GEMINI_API_KEY")
if _GEMINI_KEY:
    os.environ["GEMINI_API_KEY"] = _GEMINI_KEY

# ── IMPORT RAG PIPELINE ───────────────────────────────────────────────────────
# generate_response() adalah wrapper function publik di generation.py.
# get_bot_cached() membungkus get_bot() dengan @st.cache_resource agar
# model SBERT dan database NPZ hanya dimuat SEKALI per sesi Streamlit.
try:
    from rag_pipeline.generation import generate_response, get_bot

    @st.cache_resource(show_spinner="Memuat model RAG (hanya sekali)...")
    def get_bot_cached():
        """Cache instance ResponseGenerator agar tidak reload setiap rerun."""
        return get_bot()

    # Panaskan cache saat startup
    get_bot_cached()

    RAG_AVAILABLE = True
    RAG_ERROR     = None
except Exception as _e:
    RAG_AVAILABLE = False
    RAG_ERROR     = str(_e)

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriBot Kendari",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Serif:wght@400;600&display=swap" rel="stylesheet">

<style>
/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
  --bg:       #f5f5f0;
  --surface:  #ffffff;
  --border:   #e2e2dc;
  --green-1:  #1c3d2e;
  --green-2:  #2d6147;
  --green-3:  #3d8460;
  --green-4:  #d4ece1;
  --green-5:  #edf6f1;
  --text-1:   #1a1a1a;
  --text-2:   #4a4a4a;
  --text-3:   #888880;
  --amber-bg: #fdf6e3;
  --amber-br: #e8d5a0;
  --amber-tx: #7a5c20;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}

/* Hide Streamlit default chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* ── Top bar ── */
.topbar {
  background: var(--green-1);
  padding: 14px 28px;
  display: flex;
  align-items: center;
  gap: 10px;
  margin: -1rem -1rem 1.5rem -1rem;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.topbar-brand {
  font-family: 'IBM Plex Serif', serif;
  font-size: 17px;
  font-weight: 600;
  color: #fff;
}
.topbar-sep { color: rgba(255,255,255,0.2); margin: 0 4px; font-size: 14px; }
.topbar-sub { font-size: 12px; color: rgba(255,255,255,0.4); }
.topbar-spacer { flex: 1; }
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: rgba(255,255,255,0.6);
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 100px;
  padding: 4px 13px;
  font-weight: 500;
}
.sdot {
  width: 6px; height: 6px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 2px;
  vertical-align: middle;
}
.sdot-ok     { background: #4ade80; }
.sdot-off    { background: #f87171; }
.sdot-wait   { background: #f59e0b; }

/* Text dalam input — semua input & textarea harus hitam */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stTextArea"] textarea {
  color: #1a1a1a !important;
}

/* Placeholder */
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder,
[data-testid="stSidebar"] [data-testid="stTextArea"] textarea::placeholder {
  color: #888880 !important;
  opacity: 1 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* Sidebar label text should be dark */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] small {
  color: var(--text-2) !important;
}

.panel-head {
  font-size: 10.5px;
  font-weight: 700;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.7px;
  padding: 12px 16px 8px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 8px;
}

/* Bot identity card */
.bot-card {
  background: var(--green-1);
  padding: 18px 16px 16px;
  margin-bottom: 8px;
}
.bot-name {
  font-family: 'IBM Plex Serif', serif;
  font-size: 17px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 5px;
}
.bot-desc {
  font-size: 12px;
  color: rgba(255,255,255,0.48);
  line-height: 1.55;
  margin-bottom: 12px;
}
.bot-tags { display: flex; flex-wrap: wrap; gap: 5px; }
.bot-tag {
  font-size: 10px;
  font-weight: 600;
  color: rgba(255,255,255,0.55);
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 4px;
  padding: 2px 8px;
  letter-spacing: 0.3px;
}

/* Stat boxes */
.stat-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 7px;
  padding: 4px 0;
}
.stat-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 9px 10px;
  text-align: center;
}
.stat-val {
  font-family: 'IBM Plex Serif', serif;
  font-size: 20px;
  font-weight: 600;
  color: var(--green-2);
  display: block;
}
.stat-lbl {
  font-size: 10px;
  color: var(--text-3);
  margin-top: 2px;
  display: block;
  line-height: 1.3;
}

/* ── Chat messages ── */
.chat-area {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  min-height: 420px;
  margin-bottom: 12px;
}

.msg-row-bot, .msg-row-user {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  margin-bottom: 16px;
  animation: fadeup 0.25s ease;
}
.msg-row-user { flex-direction: row-reverse; }
@keyframes fadeup { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

.msg-av {
  width: 28px; height: 28px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700;
  flex-shrink: 0;
  margin-top: 2px;
}
.msg-av-bot  { background: var(--green-4); color: var(--green-1); }
.msg-av-user { background: var(--green-1); color: #fff; }

.bubble-bot, .bubble-user {
  max-width: 78%;
  padding: 11px 14px;
  border-radius: 10px;
  font-size: 13.5px;
  line-height: 1.72;
}
.bubble-bot {
  background: var(--bg);
  border: 1px solid var(--border);
  border-top-left-radius: 2px;
  color: var(--text-1);
}
.bubble-user {
  background: var(--green-1);
  color: #fff;
  border-top-right-radius: 2px;
}
.msg-time {
  font-size: 10px;
  color: var(--text-3);
  margin-top: 4px;
}
.msg-row-user .msg-time { text-align: right; }

/* Rich text inside bot bubble */
.bh {
  font-family: 'IBM Plex Serif', serif;
  font-size: 14px; font-weight: 600;
  color: var(--green-1);
  display: block;
  margin: 12px 0 5px;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--green-4);
}
.bh:first-child { margin-top: 0; }
.bp { display: block; margin: 3px 0; }
.bg { display: block; height: 7px; }

.bul { list-style: none; margin: 5px 0; padding: 0; }
.bul li { position: relative; padding: 2px 0 2px 17px; line-height: 1.65; }
.bul li::before {
  content: ''; position: absolute;
  left: 4px; top: 10px;
  width: 5px; height: 5px;
  border-radius: 50%; background: var(--green-3);
}
.bul li.sub { padding-left: 30px; font-size: 13px; color: var(--text-2); }
.bul li.sub::before { left: 16px; width: 4px; height: 4px; background: #d0d0c8; }

.bol { margin: 5px 0; padding-left: 20px; }
.bol li { margin: 3px 0; line-height: 1.65; }

.src-tag {
  display: inline-block;
  font-size: 10.5px; font-weight: 600;
  color: var(--amber-tx);
  background: var(--amber-bg);
  border: 1px solid var(--amber-br);
  border-radius: 4px;
  padding: 1px 7px;
  margin: 1px 2px;
  vertical-align: middle;
  max-width: 240px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ref-block {
  margin-top: 12px;
  padding: 10px 13px;
  background: var(--green-5);
  border: 1px solid var(--green-4);
  border-left: 3px solid var(--green-3);
  border-radius: 0 6px 6px 0;
}
.ref-head {
  font-size: 10.5px; font-weight: 700;
  color: var(--green-2);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
.ref-entry {
  font-size: 12px; color: var(--text-2);
  padding: 4px 0;
  border-top: 1px solid var(--green-4);
  line-height: 1.5;
}
.ref-entry:first-of-type { border-top: none; }
            
/* Empty state */
.empty-state {
  text-align: center;
  padding: 50px 20px;
}
.empty-state h2 {
  font-family: 'IBM Plex Serif', serif;
  font-size: 20px;
  color: var(--green-1);
  margin-bottom: 8px;
}
.empty-state p {
  font-size: 13px;
  color: var(--text-3);
  line-height: 1.65;
}

/* Typing indicator */
.typing-bubble {
  display: flex; gap: 5px; align-items: center;
  padding: 12px 15px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  border-top-left-radius: 2px;
  width: fit-content;
}
.typing-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green-3);
  animation: td 1.2s infinite;
  display: inline-block;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes td { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)} }

/* Spinner text jadi hitam */
[data-testid="stSpinner"] div {
  color: #000000 !important;
}

/* Streamlit widget overrides */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 13.5px !important;
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  color: var(--text-1) !important;
  -webkit-text-fill-color: var(--text-1) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--green-3) !important;
  box-shadow: 0 0 0 2px rgba(61,132,96,0.12) !important;
}
.stButton > button {
  font-family: 'IBM Plex Sans', sans-serif !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  transition: all 0.15s !important;
}
/* Primary buttons */
.stButton > button[kind="primary"] {
  background: var(--green-2) !important;
  color: #fff !important;
  border: none !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--green-1) !important;
}
/* Secondary / ghost buttons */
.stButton > button:not([kind="primary"]) {
  background: var(--bg) !important;
  color: var(--text-2) !important;
  border: 1px solid var(--border) !important;
}
.stButton > button:not([kind="primary"]):hover {
  background: var(--green-5) !important;
  border-color: var(--green-3) !important;
  color: var(--green-2) !important;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "connected"    not in st.session_state: st.session_state.connected    = RAG_AVAILABLE
if "n_questions"  not in st.session_state: st.session_state.n_questions  = 0
if "n_responses"  not in st.session_state: st.session_state.n_responses  = 0
if "last_time"    not in st.session_state: st.session_state.last_time    = "—"
if "pending_send" not in st.session_state: st.session_state.pending_send = None

# Tampilkan pesan sambutan otomatis jika RAG berhasil dimuat
if RAG_AVAILABLE and not st.session_state.messages:
    st.session_state.messages.append({
        "role": "bot",
        "content": (
            "Halo! Saya **NutriBot Kendari**.\n\n"
            "Saya siap membantu konsultasi seputar stunting, gizi anak, "
            "dan kesehatan ibu-balita di Kota Kendari. "
            "Silakan ajukan pertanyaan Anda."
        ),
        "time": time.strftime("%H:%M"),
    })

# ── HELPERS ──────────────────────────────────────────────────────────────────
def call_rag(query: str) -> str:
    """
    Delegasikan query ke ResponseGenerator yang sudah di-cache.
    Tidak ada HTTP request — semua berjalan in-process.
    """
    if not RAG_AVAILABLE:
        raise RuntimeError(f"Modul RAG tidak dapat dimuat: {RAG_ERROR}")
    bot = get_bot_cached()   # instance yang sudah warm, tidak reload ulang
    return bot.generate(query)


def render_md(raw: str) -> str:
    """Convert markdown-like bot response to clean HTML."""
    import re
    import html as htmllib

    # ── Extract reference block ──
    ref_html = ""
    rm = re.search(r"(📚\s*Referensi[\s\S]*?)$", raw, re.IGNORECASE)
    if rm:
        raw = raw[:rm.start()].rstrip()
        lines = [l.strip() for l in rm.group(1).split("\n") if l.strip()]
        entries = [
            f'<div class="ref-entry">{_inl(re.sub(r"^[*\\-\\d.]\\s*", "", l))}</div>'
            for l in lines if not re.search(r"📚\s*Referensi", l, re.I)
        ]
        if entries:
            ref_html = (
                '<div class="ref-block">'
                '<div class="ref-head">Referensi</div>'
                + "".join(entries)
                + "</div>"
            )

    # ── Line-by-line parse ──
    lines = raw.split("\n")
    out = []
    i = 0
    while i < len(lines):
        ln  = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""

        if re.match(r"^#{1,3}\s", ln):
            out.append(f'<span class="bh">{_inl(re.sub(r"^#+\\s*", "", ln))}</span>')
            i += 1; continue

        if re.match(r"^\d+\.\s+\S.{0,80}$", ln) and not re.match(r"^\s*[\*\-\d]", nxt):
            out.append(f'<span class="bh">{_inl(re.sub(r"^\\d+\\.\\s+", "", ln))}</span>')
            i += 1; continue

        if re.match(r"^\d+\.\s", ln):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i]):
                items.append(f"<li>{_inl(re.sub(r'^\\d+\\.\\s+', '', lines[i]))}</li>")
                i += 1
            out.append('<ol class="bol">' + "".join(items) + "</ol>")
            continue

        if re.match(r"^\s*[\*\-]\s", ln):
            items = []
            while i < len(lines) and re.match(r"^\s*[\*\-]\s", lines[i]):
                indent = len(lines[i]) - len(lines[i].lstrip())
                cls = ' class="sub"' if indent >= 2 else ""
                items.append(f'<li{cls}>{_inl(re.sub(r"^\\s*[\\*\\-]\\s+", "", lines[i]))}</li>')
                i += 1
            out.append('<ul class="bul">' + "".join(items) + "</ul>")
            continue

        if not ln.strip():
            out.append('<span class="bg"></span>')
            i += 1; continue

        out.append(f'<span class="bp">{_inl(ln)}</span>')
        i += 1

    return "".join(out) + ref_html


def _inl(t: str) -> str:
    import re
    import html as h
    t = h.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    def src(m):
        s = m.group(1)
        d = s[:52] + "…" if len(s) > 55 else s
        return f'<span class="src-tag" title="{s}">Sumber: {d}</span>'
    t = re.sub(r"→?\s*\(Sumber:\s*([^)]+)\)", src, t)
    return t


def now_str() -> str:
    return time.strftime("%H:%M")

# ── TOPBAR ───────────────────────────────────────────────────────────────────
dot_cls = "sdot-ok"   if st.session_state.connected else "sdot-off"
dot_txt = "Siap digunakan" if st.session_state.connected else "Modul RAG tidak tersedia"

st.markdown(f"""
<div class="topbar">
  <span class="topbar-brand">NutriBot Kendari</span>
  <span class="topbar-sep">/</span>
  <span class="topbar-sub">Konsultasi Risiko Stunting</span>
  <div class="topbar-spacer"></div>
  <div class="status-pill">
    <span class="sdot {dot_cls}"></span>
    {dot_txt}
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Bot identity card
    st.markdown("""
    <div class="bot-card">
      <div class="bot-name">NutriBot</div>
      <div class="bot-desc">Asisten konsultasi risiko stunting berbasis AI untuk Kota Kendari, Sulawesi Tenggara.</div>
      <div class="bot-tags">
        <span class="bot-tag">RAG</span>
        <span class="bot-tag">IndoBERT</span>
        <span class="bot-tag">SBERT</span>
        <span class="bot-tag">Gemini</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status RAG Pipeline ──
    st.markdown('<div class="panel-head">Status Pipeline</div>', unsafe_allow_html=True)
    if RAG_AVAILABLE:
        st.success("✅ RAG pipeline aktif", icon=None)
    else:
        st.error(f"❌ RAG pipeline gagal dimuat:\n\n`{RAG_ERROR}`")
        st.info(
            "Periksa:\n"
            "- `rag_pipeline/generation.py` ada dan bisa diimport\n"
            "- `GEMINI_API_KEY` sudah di-set di **Secrets** (Streamlit Cloud)\n"
            "  atau di file `.env` (lokal)"
        )

    st.markdown("---")

    # ── Topics ──
    st.markdown('<div class="panel-head">Topik Konsultasi</div>', unsafe_allow_html=True)

    TOPICS = [
        ("Definisi Stunting",         "Penyebab dan pencegahan",            "Apa itu stunting dan bagaimana cara mencegahnya?"),
        ("MPASI",                     "Panduan makanan pendamping ASI",      "Kapan dan bagaimana cara memberikan MPASI yang benar?"),
        ("Gizi Ibu Hamil",            "Nutrisi dan suplemen kehamilan",      "Apa saja kebutuhan gizi ibu hamil untuk mencegah stunting?"),
        ("ASI dan Menyusui",          "Manfaat dan teknik menyusui",         "Bagaimana manfaat ASI dan cara menyusui yang benar?"),
        ("Deteksi Dini",              "Tanda dan pemantauan tumbuh kembang", "Apa saja tanda stunting pada anak dan cara mendeteksinya?"),
        ("Layanan Kesehatan Kendari", "Posyandu dan puskesmas terdekat",     "Di mana layanan posyandu dan puskesmas di Kota Kendari?"),
    ]

    for label, sub, query in TOPICS:
        if st.button(f"**{label}**\n\n{sub}", use_container_width=True, key=f"topic_{label}"):
            if st.session_state.connected:
                st.session_state.pending_send = query
                st.rerun()

    st.markdown("---")

    # ── Stats ──
    st.markdown('<div class="panel-head">Statistik Sesi</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-box">
        <span class="stat-val">{st.session_state.n_questions}</span>
        <span class="stat-lbl">Pertanyaan</span>
      </div>
      <div class="stat-box">
        <span class="stat-val">{st.session_state.n_responses}</span>
        <span class="stat-lbl">Respons</span>
      </div>
      <div class="stat-box">
        <span class="stat-val">{st.session_state.last_time}</span>
        <span class="stat-lbl">Waktu (s)</span>
      </div>
      <div class="stat-box">
        <span class="stat-val">{"OK" if st.session_state.connected else "ERR"}</span>
        <span class="stat-lbl">Pipeline</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Clear chat
    if st.button("Hapus percakapan", use_container_width=True):
        st.session_state.messages    = []
        st.session_state.n_questions = 0
        st.session_state.n_responses = 0
        st.session_state.last_time   = "—"
        st.rerun()

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
hdr_status = "Online · Siap berkonsultasi" if st.session_state.connected else "Pipeline tidak tersedia"
st.markdown(f"""
<div style="
  background:#f5f5f0;
  border:1px solid #e2e2dc;
  border-radius:12px 12px 0 0;
  padding:12px 20px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom:-1px;
">
  <div>
    <strong style="font-size:14px;display:block;color:#000000;">NutriBot Kendari</strong>
    <span style="font-size:11px;color:#888880;">{hdr_status}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Messages
with st.container():
    if not st.session_state.messages:
        st.markdown("""
        <div class="chat-area">
          <div class="empty-state">
            <h2>Selamat datang</h2>
            <p>Saya siap membantu konsultasi seputar stunting, gizi anak, dan kesehatan ibu-balita di Kota Kendari.<br>
            Pilih topik di sidebar atau ketik pertanyaan Anda.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        bubbles_html = '<div class="chat-area">'
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                import html as _h
                safe = _h.escape(msg["content"]).replace("\n", "<br>")
                bubbles_html += f"""
                <div class="msg-row-user">
                  <div class="msg-av msg-av-user">S</div>
                  <div>
                    <div class="bubble-user">{safe}</div>
                    <div class="msg-time" style="text-align:right">{msg["time"]}</div>
                  </div>
                </div>"""
            else:
                rendered = render_md(msg["content"])
                elapsed  = msg.get("elapsed", "")
                time_str = f"{msg['time']} · {elapsed}s" if elapsed else msg["time"]
                bubbles_html += f"""
                <div class="msg-row-bot">
                  <div class="msg-av msg-av-bot">N</div>
                  <div style="max-width:78%">
                    <div class="bubble-bot">{rendered}</div>
                    <div class="msg-time">{time_str}</div>
                  </div>
                </div>"""
        bubbles_html += "</div>"
        st.markdown(bubbles_html, unsafe_allow_html=True)

# ── INPUT BAR ─────────────────────────────────────────────────────────────────
if not st.session_state.connected:
    st.warning(
        "RAG pipeline tidak aktif. Periksa log di sidebar untuk detail error.",
        icon="⚠️",
    )

with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([11, 1])
    with c1:
        user_input = st.text_area(
            "Pesan",
            placeholder="Tanyakan seputar stunting, gizi, atau kesehatan anak...",
            label_visibility="collapsed",
            height=68,
            disabled=not st.session_state.connected,
        )
    with c2:
        submitted = st.form_submit_button(
            "→",
            use_container_width=True,
            type="primary",
            disabled=not st.session_state.connected,
        )

st.markdown(
    '<div style="text-align:center;font-size:10.5px;color:#888880;margin-top:4px;">'
    'Enter untuk baris baru &nbsp;·&nbsp; Klik tombol kirim untuk mengirim pesan'
    '</div>',
    unsafe_allow_html=True,
)

# ── HANDLE SEND ───────────────────────────────────────────────────────────────
def process_query(query: str):
    query = query.strip()
    if not query:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "time": now_str(),
    })
    st.session_state.n_questions += 1

    with st.spinner("NutriBot sedang memproses..."):
        try:
            t0      = time.time()
            response = call_rag(query)          # ← panggil langsung, bukan HTTP
            elapsed  = f"{time.time() - t0:.1f}"
        except Exception as e:
            response = (
                f"⚠️ Terjadi kesalahan saat memproses pertanyaan:\n\n"
                f"`{e}`\n\n"
                f"Periksa konfigurasi `rag_pipeline/generation.py` dan API key."
            )
            elapsed = ""

    st.session_state.messages.append({
        "role": "bot",
        "content": response,
        "time": now_str(),
        "elapsed": elapsed,
    })
    st.session_state.n_responses += 1
    if elapsed:
        st.session_state.last_time = elapsed
    st.rerun()


if submitted and user_input and user_input.strip():
    process_query(user_input)

if st.session_state.pending_send:
    q = st.session_state.pending_send
    st.session_state.pending_send = None
    process_query(q)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  text-align:center;
  font-size:11px;
  color:#888880;
  border-top:1px solid #e2e2dc;
  padding:12px 0;
  margin-top:24px;
  background:#fff;
">
  NutriBot Kendari &nbsp;·&nbsp; Sistem RAG Konsultasi Stunting &nbsp;·&nbsp;
  IndoBERT + SBERT + Gemini &nbsp;·&nbsp; Universitas Halu Oleo, Kota Kendari
</div>
""", unsafe_allow_html=True)