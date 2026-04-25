"""
generation.py
=============
Modul Generasi Respons NutriBot Kendari (Versi Hybrid RAG).
Mampu mengambil data dari database lokal DAN menggunakan pengetahuan AI jika data kosong.
"""

import logging
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# ─── LOAD .env (hanya efektif di lokal; diabaikan di Streamlit Cloud) ────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv tidak wajib di Cloud

# ─── PATH SETUP DINAMIS ───────────────────────────────────────────────────────
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent   # .../Stunting-chatbot-fix/

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ─── IMPORT MODUL ─────────────────────────────────────────────────────────────
from google import genai
from google.genai import types

try:
    from retrieval import StuntingRetriever
    from embedding.create_embedding import EmbeddingCreator
except ImportError:
    try:
        from rag_pipeline.retrieval import StuntingRetriever
        from embedding.create_embedding import EmbeddingCreator
    except ImportError as e:
        raise ImportError(
            f"Gagal mengimpor StuntingRetriever / EmbeddingCreator: {e}\n"
            "Pastikan __init__.py ada di setiap folder (rag_pipeline/, embedding/)."
        )

logger = logging.getLogger("stunting.generation")

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """Anda adalah NutriBot Kendari, asisten ahli stunting di Kota Kendari. 
Tugas Anda adalah memberikan edukasi gizi dan pencegahan stunting secara ramah dan berbasis sumber.

ATURAN MENJAWAB (WAJIB DIIKUTI):
1. Jika ada 'Konteks Database', SELALU gunakan sebagai referensi utama jawaban Anda.
2. Setiap informasi penting yang Anda sampaikan dari konteks HARUS diikuti dengan kutipan sumber dalam format:
   → (Sumber: [nama seksi dari konteks, contoh: SEKSI 5: PENCEGAHAN STUNTING])
3. Jika ada beberapa konteks yang relevan, sebutkan semua sumbernya secara berurutan.
4. Jika 'Konteks Database' kosong atau tidak relevan, gunakan pengetahuan medis umum Anda dan tandai dengan:
   → (Sumber: Pengetahuan Umum Medis)
5. Di akhir jawaban, selalu tambahkan bagian "📚 Referensi:" yang merangkum semua seksi yang dikutip.
6. Selalu sarankan kunjungan ke Puskesmas di Kendari jika kondisi mendesak.
7. Gunakan bahasa Indonesia yang mudah dipahami oleh ibu rumah tangga."""


# ─────────────────────────────────────────────
# CLASS UTAMA
# ─────────────────────────────────────────────
class ResponseGenerator:
    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.5-flash",
        max_tokens: int = 4096,
        temperature: float = 0.5,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.provider      = provider.lower()
        self.model_id      = model
        self.max_tokens    = max_tokens
        self.temperature   = temperature
        self.system_prompt = system_prompt
        self.client        = None
        self.retriever     = None
        self.embedder      = None

    def setup_engine(self, api_key: str, emb_path: str, meta_path: str):
        """Inisialisasi Client Gemini, SBERT, dan Database NPZ."""
        self.client = genai.Client(api_key=api_key)

        from embedding.create_embedding import SBERT_INDONESIAN
        self.embedder = EmbeddingCreator(
            model_type="sbert", model_name=SBERT_INDONESIAN, device="cpu"
        )
        self.embedder.load_model()

        self.retriever = StuntingRetriever(top_k=5)
        self.retriever.load_index(emb_path, meta_path)

    def _is_out_of_scope(self, query: str) -> bool:
        query_lower = query.lower()

        out_of_scope_terms = [
            "presiden", "politik", "pemilu", "saham", "kripto", "bitcoin",
            "film", "musik", "lagu", "artis", "sepak bola", "bola basket",
            "cuaca", "ramalan", "togel", "judi", "resep masakan dewasa",
            "hukum", "pengacara", "pajak", "kode program", "coding",
            "sejarah dunia", "geografi", "matematika", "fisika", "kimia",
        ]

        in_scope_terms = [
            "stunting", "gizi", "nutrisi", "bayi", "balita", "anak", "ibu hamil",
            "asi", "mpasi", "posyandu", "puskesmas", "kesehatan", "tumbuh",
            "berat badan", "tinggi badan", "menyusui", "kehamilan", "kendari",
            "anemia", "protein", "vitamin", "mineral", "kek", "kalori",
        ]

        has_out_of_scope = any(term in query_lower for term in out_of_scope_terms)
        has_in_scope     = any(term in query_lower for term in in_scope_terms)

        return has_out_of_scope and not has_in_scope

    def generate(self, query: str) -> str:
        if not self.client or not self.retriever:
            return "Engine belum siap. Mohon inisialisasi setup_engine() dahulu."

        try:
            search_results = self.retriever.retrieve(query, self.embedder)

            CONTEXT_THRESHOLD  = 0.40
            RELEVANT_MIN_COUNT = 1

            relevant_results = [
                r for r in search_results
                if r.get("similarity_score", 0) >= CONTEXT_THRESHOLD
            ]

            out_of_scope = self._is_out_of_scope(query)
            context      = self.retriever.format_context(relevant_results) if relevant_results else ""

            if out_of_scope and not relevant_results:
                prompt_text = f"""[MODE: OUT OF SCOPE]
Pertanyaan pengguna: {query}

Pertanyaan ini berada di luar topik yang NutriBot Kendari kuasai (stunting, gizi anak, kesehatan ibu dan balita).
Sampaikan dengan ramah bahwa Anda hanya dapat membantu seputar topik stunting dan gizi anak di Kota Kendari,
lalu tawarkan untuk menjawab pertanyaan yang berkaitan dengan topik tersebut.
Jangan mencoba menjawab pertanyaan di luar domain."""

            elif not relevant_results:
                prompt_text = f"""[MODE: GENERAL KNOWLEDGE]
Pertanyaan pengguna: {query}

Basis pengetahuan tidak memiliki informasi spesifik untuk pertanyaan ini.
Jawablah menggunakan pengetahuan medis Anda mengenai stunting dan gizi anak sebagai NutriBot Kendari.
Tandai setiap informasi dengan → (Sumber: Pengetahuan Umum Medis).
Di akhir jawaban, tambahkan bagian "📚 Referensi:" berisi daftar topik yang Anda rujuk."""

            else:
                prompt_text = f"""[MODE: DATABASE RAG]
Berikut adalah konteks dari basis pengetahuan stunting Kota Kendari:

{context}

---
Pertanyaan pengguna: {query}

INSTRUKSI MENJAWAB:
- Jawab pertanyaan berdasarkan konteks di atas.
- Setiap fakta/informasi penting HARUS diikuti kutipan sumber dalam format: → (Sumber: [nama di baris "Sumber   :" pada konteks])
- Di akhir jawaban, tambahkan bagian "📚 Referensi:" yang merangkum semua sumber yang dikutip.
- Jika ada informasi yang tidak ada dalam konteks, tandai dengan → (Sumber: Pengetahuan Umum Medis)."""

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
            return response.text

        except Exception as e:
            logger.error(f"Generate Error: {e}")
            return f"NutriBot mengalami kendala teknis: {e}. Silakan coba beberapa saat lagi."


# ─────────────────────────────────────────────
# WRAPPER FUNCTION — dipanggil langsung dari app.py
# ─────────────────────────────────────────────
# Instance tunggal yang di-cache oleh Streamlit (@st.cache_resource di app.py).
# Tidak perlu membuat ulang bot setiap kali ada pertanyaan baru.
_bot_instance: Optional[ResponseGenerator] = None


def get_bot() -> ResponseGenerator:
    """
    Kembalikan instance ResponseGenerator yang sudah diinisialisasi.
    Fungsi ini dipanggil oleh app.py via @st.cache_resource sehingga
    model dan database hanya dimuat SEKALI per sesi Streamlit.
    """
    global _bot_instance
    if _bot_instance is not None:
        return _bot_instance

    # Tentukan path embedding secara otomatis relatif terhadap lokasi file ini
    emb_path  = str(project_root / "embedding" / "stunting_embeddings.npz")
    meta_path = str(project_root / "embedding" / "stunting_embeddings_metadata.json")

    # Ambil API key: env var (lokal) atau sudah di-inject oleh app.py (Cloud)
    api_key = os.environ.get("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY tidak ditemukan. "
            "Set di Streamlit Secrets (Cloud) atau file .env (lokal)."
        )

    bot = ResponseGenerator()
    bot.setup_engine(api_key, emb_path, meta_path)
    _bot_instance = bot
    return bot


def generate_response(query: str) -> str:
    """
    Fungsi wrapper publik yang dipanggil oleh app.py.

    Signature sederhana:
        generate_response(query: str) -> str

    Cara kerja:
        1. Panggil get_bot() untuk mendapatkan instance yang sudah siap.
        2. Delegasikan ke ResponseGenerator.generate(query).
    """
    bot = get_bot()
    return bot.generate(query)


# ─────────────────────────────────────────────
# TEST LANGSUNG (python generation.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    API_KEY = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")

    print("[*] Menyiapkan NutriBot (Memuat model & database)...")
    bot = get_bot()

    print("\n" + "=" * 50)
    print("--- NUTRIBOT KENDARI SIAP BERKONSULTASI ---")
    print("      (Ketik 'keluar' untuk mengakhiri)      ")
    print("=" * 50 + "\n")

    while True:
        q = input("Anda: ")
        if q.lower() in ["exit", "keluar", "quit"]:
            print("Sampai jumpa! Jaga kesehatan si kecil.")
            break

        print("\nNutriBot sedang berpikir...")
        t0     = time.time()
        jawaban = bot.generate(q)
        print(f"\nNutriBot: {jawaban}")
        print(f" (Respons dalam {time.time() - t0:.2f} detik)")
        print("-" * 50)