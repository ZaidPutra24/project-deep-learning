"""
retrieval.py
============
Modul Retrieval untuk Pipeline RAG Stunting

Mengimplementasikan pencarian semantik berbasis cosine similarity
menggunakan embedding IndoBERT/SBERT + FAISS (opsional).

Proyek  : Chatbot Konsultasi Risiko Stunting Kota Kendari
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("stunting.retrieval")

# ─────────────────────────────────────────────
# KAMUS SINONIM DOMAIN STUNTING
# Digunakan oleh _expand_query() untuk meningkatkan recall
# pada query expansion. Setiap kunci merepresentasikan
# topik utama dan nilainya adalah sinonim/istilah terkait.
# ─────────────────────────────────────────────
QUERY_SYNONYMS = {
    "stunting": ["pendek", "gagal tumbuh", "kurang gizi kronis", "sangat pendek"],
    "gizi buruk": ["malnutrisi", "kurang gizi", "wasting", "marasmus", "kwashiorkor"],
    "ibu hamil": ["kehamilan", "hamil", "prenatal", "antenatal", "bumil"],
    "bayi": ["neonatus", "newborn", "anak baru lahir", "balita"],
    "asi": ["air susu ibu", "menyusui", "laktasi"],
    "mpasi": ["makanan pendamping", "makanan tambahan bayi"],
    "posyandu": ["pos pelayanan terpadu", "penimbangan", "pemantauan pertumbuhan"],
    "anemia": ["kurang darah", "hb rendah", "hemoglobin rendah"],
    "kek": ["kurang energi kronis", "lila rendah", "underweight ibu"],
    "kendari": ["kota kendari", "sulawesi tenggara", "sultra"],
    # --- Tambahan berdasarkan hasil evaluasi: 4 query yang gagal ---
    # Query "dampak stunting" salah retrieve karena kata "dampak" tidak ada sinonimnya
    "dampak": ["akibat", "konsekuensi", "efek", "pengaruh", "risiko jangka panjang",
               "perkembangan kognitif", "kecerdasan", "produktivitas"],
    # Query "intervensi gizi spesifik" ambil SEKSI 13 — perlu perkuat kata "spesifik"
    "intervensi spesifik": ["intervensi gizi", "suplementasi", "tablet tambah darah",
                            "vitamin mineral", "gizi langsung", "tatalaksana gizi"],
    # Query "program pencegahan sensitif" kalah ke SEKSI 7
    "sensitif": ["intervensi sensitif", "program pemerintah", "sanitasi", "air bersih",
                 "ketahanan pangan", "perlindungan sosial", "program keluarga"],
    # Query "kebutuhan gizi ibu hamil" ambil SEKSI 13 — perkuat konteks ibu hamil
    "kebutuhan gizi": ["gizi ibu hamil", "nutrisi kehamilan", "asupan ibu", "zat besi",
                       "asam folat", "protein ibu", "kalori ibu hamil"],
}


class StuntingRetriever:
    """
    Kelas retriever untuk mencari chunk relevan berdasarkan query pengguna.

    Strategi retrieval:
        1. Dense retrieval (cosine similarity / dot product)
        2. FAISS index untuk retrieval cepat pada dataset besar (opsional)
        3. Keyword filter sebagai pre/post filter

    Contoh:
        retriever = StuntingRetriever()
        retriever.load_index("embedding/stunting_embeddings.npz",
                             "embedding/stunting_embeddings_metadata.json")
        results = retriever.retrieve("apa itu stunting?", top_k=5)
    """

    def __init__(
        self,
        top_k: int = 5,
        similarity_threshold: float = 0.15,
        use_faiss: bool = False,
        rerank: bool = True,
    ):
        """
        Args:
            top_k               : Jumlah chunk teratas yang dikembalikan
            similarity_threshold: Ambang batas skor minimum (0-1)
            use_faiss           : Gunakan FAISS index untuk retrieval cepat
            rerank              : Reranking berdasarkan keyword match
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_faiss = use_faiss
        self.rerank = rerank

        self.embeddings: Optional[np.ndarray] = None
        self.chunk_ids: Optional[List[str]] = None
        self.metadata: Optional[Dict] = None
        self.chunks_map: Optional[Dict[str, Dict]] = None
        self.faiss_index = None

        logger.info(
            f"StuntingRetriever: top_k={top_k} | threshold={similarity_threshold} | "
            f"use_faiss={use_faiss} | rerank={rerank}"
        )

    def load_index(self, embeddings_path: str, metadata_path: str) -> None:
        """
        Memuat embedding index dan metadata chunk ke memori.

        Args:
            embeddings_path: Path ke file .npz embedding
            metadata_path  : Path ke file JSON metadata chunk
        """
        logger.info(f"Memuat embedding dari: {embeddings_path}")
        data = np.load(embeddings_path, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.chunk_ids = list(data["chunk_ids"])

        logger.info(f"Memuat metadata dari: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # Buat mapping chunk_id → chunk dict
        self.chunks_map = {
            c["chunk_id"]: c for c in meta_data.get("chunks", [])
        }

        logger.info(
            f"Index dimuat: {self.embeddings.shape[0]} chunk, "
            f"dim={self.embeddings.shape[1]}"
        )

        if self.use_faiss:
            self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        """Membangun FAISS index untuk pencarian cepat."""
        try:
            import faiss
            dim = self.embeddings.shape[1]
            # IndexFlatIP: Inner Product (untuk embedding yang sudah dinormalisasi = cosine)
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
            logger.info(f"FAISS index dibangun: {self.faiss_index.ntotal} vektor")
        except ImportError:
            logger.warning("FAISS tidak tersedia. Gunakan: pip install faiss-cpu")
            self.use_faiss = False

    def embed_query(self, query: str, embedder) -> np.ndarray:
        """
        Membuat embedding untuk query pengguna.

        Args:
            query   : Teks pertanyaan pengguna
            embedder: Instance EmbeddingCreator yang sudah di-load

        Returns:
            numpy array embedding query (1D)

        PERBAIKAN: Docstring dipindahkan ke atas return (sebelumnya ada
        setelah return sehingga menjadi dead code yang tidak terbaca).
        Normalisasi L2 tetap dipertahankan karena wajib untuk cosine similarity.
        """
        emb = embedder.embed_text(query)

        # Normalisasi L2 (wajib agar dot product = cosine similarity)
        norm = np.linalg.norm(emb)
        emb = emb / max(norm, 1e-9)

        return emb.astype(np.float32)

    def cosine_similarity_batch(
        self, query_emb: np.ndarray, corpus_embs: np.ndarray
    ) -> np.ndarray:
        """
        Menghitung cosine similarity antara query dan semua chunk.

        Args:
            query_emb  : Embedding query (1D, sudah ternormalisasi)
            corpus_embs: Embedding korpus (2D: N x dim, sudah ternormalisasi)

        Returns:
            Array skor similarity (1D, panjang N)
        """
        # Untuk embedding yang sudah dinormalisasi L2, dot product = cosine similarity
        scores = np.dot(corpus_embs, query_emb)
        return scores

    def retrieve(
        self,
        query: str,
        embedder,
        top_k: Optional[int] = None,
        keyword_filter: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Mencari chunk paling relevan untuk query pengguna.

        Args:
            query         : Pertanyaan pengguna dalam bahasa Indonesia
            embedder      : Instance EmbeddingCreator yang sudah di-load
            top_k         : Override jumlah chunk (default: self.top_k)
            keyword_filter: Filter berdasarkan keyword domain (opsional)

        Returns:
            List dict chunk yang relevan, diurutkan berdasarkan skor.
        """
        if self.embeddings is None:
            raise RuntimeError("Index belum dimuat. Panggil load_index() terlebih dahulu.")

        k = top_k or self.top_k
        start = time.time()

        # Query expansion berbasis sinonim domain stunting.
        # PERBAIKAN: Sebelumnya selalu menambahkan "stunting gizi anak" untuk query
        # pendek (<4 kata), sehingga query spesifik seperti "prevalensi Kendari" atau
        # "kapan MPASI" tidak mendapat ekspansi yang relevan dan skor retrieval
        # turun. Sekarang ekspansi berdasarkan sinonim topik yang terdeteksi di query,
        # sehingga lebih akurat dan tidak mendistorsi makna query asli.
        expanded_query = self._expand_query(query)
        query_emb = self.embed_query(expanded_query, embedder)

        # Cari dengan FAISS atau numpy
        if self.use_faiss and self.faiss_index is not None:
            scores, indices = self._faiss_search(query_emb, k * 2)
        else:
            scores, indices = self._numpy_search(query_emb, k * 2)

        # Bangun list hasil
        results = []
        for score, idx in zip(scores, indices):
            if score < self.similarity_threshold:
                continue
            chunk_id = self.chunk_ids[idx]
            chunk = dict(self.chunks_map.get(chunk_id, {}))
            chunk["similarity_score"] = float(score)
            chunk["rank"] = len(results) + 1
            # Bangun source_label eksplisit agar tersedia saat format_context() dipanggil
            chunk["source_label"] = self._build_source_label(chunk)
            results.append(chunk)

        # Keyword filter (post-filter)
        if keyword_filter:
            results = self._filter_by_keywords(results, keyword_filter)

        # Reranking
        if self.rerank and results:
            results = self._rerank(results, query)

        results = results[:k]

        elapsed = time.time() - start
        logger.info(
            f"Retrieval selesai: '{query[:60]}' → {len(results)} hasil ({elapsed*1000:.1f} ms)"
        )

        return results

    def _expand_query(self, query: str) -> str:
        """
        Memperluas query dengan sinonim domain stunting untuk meningkatkan recall.

        Berbeda dari pendekatan sebelumnya yang selalu menambahkan "stunting gizi anak"
        ke query pendek saja, metode ini mendeteksi topik yang sudah ada di query lalu
        menambahkan sinonimnya. Hasilnya: embedding expanded query lebih merepresentasikan
        intent pengguna, dan chunk yang secara semantik relevan lebih mudah ditemukan.

        Contoh:
            "anak pendek"   → "anak pendek stunting gagal tumbuh kurang gizi kronis"
            "kapan MPASI"   → "kapan MPASI makanan pendamping makanan tambahan bayi"
            "prevalensi Kendari" → "prevalensi Kendari kota kendari sulawesi tenggara"
        """
        query_lower = query.lower()
        extra_terms = []

        for term, synonyms in QUERY_SYNONYMS.items():
            all_forms = [term] + synonyms
            if any(f in query_lower for f in all_forms):
                for syn in synonyms:
                    if syn.lower() not in query_lower:
                        extra_terms.append(syn)

        if extra_terms:
            return (query + " " + " ".join(extra_terms)).strip()
        return query

    def _build_source_label(self, chunk: Dict) -> str:
        """
        Membangun label sumber eksplisit untuk citation dalam jawaban LLM.

        Label yang konsisten memudahkan LLM mengutip sumber dengan benar
        menggunakan nomor [Konteks N] yang ada di format_context().

        Format: "SEKSI X: Judul > Subseksi"  atau  "Referensi Ilmiah | Kategori"
        """
        section = chunk.get("section", "Basis Pengetahuan Stunting")
        subsection = chunk.get("subsection", "")
        source_file = chunk.get("source_file", "")

        if "reference" in source_file.lower():
            return f"Referensi Ilmiah | {section}"

        if subsection and subsection not in ("Umum", "Pendahuluan", "", section):
            return f"{section} > {subsection}"

        return section

    def _numpy_search(
        self, query_emb: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pencarian brute-force menggunakan numpy."""
        scores = self.cosine_similarity_batch(query_emb, self.embeddings)
        # Ambil top-k dengan argsort descending
        top_indices = np.argsort(scores)[::-1][:k]
        return scores[top_indices], top_indices

    def _faiss_search(
        self, query_emb: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pencarian menggunakan FAISS index."""
        query_2d = query_emb.reshape(1, -1)
        scores, indices = self.faiss_index.search(query_2d, k)
        return scores[0], indices[0]

    def _filter_by_keywords(
        self, results: List[Dict], keywords: List[str]
    ) -> List[Dict]:
        """
        Filter hasil berdasarkan keyword domain.
        Chunk yang mengandung keyword mendapat prioritas.
        """
        filtered = []
        for chunk in results:
            chunk_keywords = chunk.get("keywords", [])
            if any(kw in chunk_keywords for kw in keywords):
                filtered.append(chunk)
        # Jika tidak ada yang lolos filter, kembalikan semua
        return filtered if filtered else results

    # Kata-kata umum Bahasa Indonesia yang tidak diskriminatif untuk reranking.
    # Overlap pada kata-kata ini tidak boleh menambah bonus karena hampir semua
    # chunk mengandung kata ini → menyebabkan chunk panjang (SEKSI 13, dll.) selalu
    # menang meskipun bukan sumber yang paling relevan.
    _STOPWORDS = {
        "apa", "adalah", "yang", "di", "ke", "dari", "dan", "atau", "untuk",
        "dengan", "pada", "ini", "itu", "dalam", "oleh", "jika", "maka",
        "serta", "juga", "dapat", "harus", "akan", "telah", "sudah", "tidak",
        "anak", "bayi", "balita", "stunting", "gizi",  # terlalu umum di corpus ini
    }

    def _rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Reranking berdasarkan:
        1. Overlap kata DISKRIMINATIF (setelah filter stopword) antara query dan teks chunk
        2. Section-match bonus: jika kata kunci unik query muncul di nama seksi chunk

        PERBAIKAN v2 (dari evaluasi):
        - Sebelumnya kata umum seperti "stunting", "gizi", "anak" turut dihitung overlap,
          sehingga chunk panjang dari SEKSI 13 (Mitos/Info Tambahan) yang membahas banyak
          topik selalu mendapat bonus tinggi dan mengalahkan seksi yang lebih spesifik.
        - Sekarang stopword + kata corpus umum difilter dulu sebelum menghitung overlap.
        - Ditambah section_bonus: jika kata unik query cocok dengan judul seksi chunk,
          chunk itu mendapat prioritas lebih tinggi (membantu query "dampak stunting",
          "intervensi spesifik", "gizi ibu hamil" menemukan seksinya sendiri).
        """
        # Filter kata diskriminatif dari query (hilangkan stopword)
        query_words_all = set(query.lower().split())
        query_words = query_words_all - self._STOPWORDS
        query_len = max(len(query_words), 1)

        for chunk in results:
            text_lower = chunk.get("text", "").lower()
            text_words = set(text_lower.split()) - self._STOPWORDS

            # Overlap hanya pada kata diskriminatif
            overlap = len(query_words & text_words)
            overlap_density = overlap / query_len
            bonus = overlap_density * 0.12

            # Section-match bonus: kata unik query yang muncul di judul seksi chunk
            # Ini membantu query spesifik menemukan seksi yang tepat
            section_lower = chunk.get("section", "").lower()
            section_overlap = len(query_words & set(section_lower.split()))
            if section_overlap > 0:
                bonus += (section_overlap / query_len) * 0.10

            chunk["rerank_score"] = chunk["similarity_score"] + bonus

        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return results

    def retrieve_with_context(
        self,
        query: str,
        embedder,
        top_k: Optional[int] = None,
        context_window: int = 1,
    ) -> List[Dict]:
        """
        Retrieval dengan chunk tetangga untuk konteks tambahan.

        Mengambil chunk sebelum dan sesudah setiap chunk yang relevan
        untuk memberikan konteks yang lebih lengkap kepada generator.

        Args:
            query         : Query pengguna
            embedder      : EmbeddingCreator
            top_k         : Jumlah chunk utama
            context_window: Jumlah chunk tetangga di kiri dan kanan

        Returns:
            List chunk dengan konteks yang diperluas
        """
        primary_results = self.retrieve(query, embedder, top_k)

        if context_window == 0:
            return primary_results

        # Kumpulkan chunk ID yang sudah ada
        primary_ids = {r["chunk_id"] for r in primary_results}
        all_ids = list(self.chunks_map.keys())

        expanded = list(primary_results)
        for result in primary_results:
            current_id = result["chunk_id"]
            if current_id not in all_ids:
                continue
            current_idx = all_ids.index(current_id)

            for offset in range(-context_window, context_window + 1):
                if offset == 0:
                    continue
                neighbor_idx = current_idx + offset
                if 0 <= neighbor_idx < len(all_ids):
                    neighbor_id = all_ids[neighbor_idx]
                    if neighbor_id not in primary_ids:
                        neighbor_chunk = dict(self.chunks_map[neighbor_id])
                        neighbor_chunk["similarity_score"] = result["similarity_score"] * 0.7
                        neighbor_chunk["is_context"] = True
                        # Selalu set source_label agar format_context() tidak fallback ke section saja
                        neighbor_chunk["source_label"] = self._build_source_label(neighbor_chunk)
                        expanded.append(neighbor_chunk)
                        primary_ids.add(neighbor_id)

        # Urutkan kembali berdasarkan skor
        expanded.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return expanded

    def format_context(self, chunks: List[Dict], max_chars: int = 3000) -> str:
        """
        Memformat chunk hasil retrieval menjadi string konteks untuk LLM.
        Label sumber ditulis eksplisit agar LLM mudah mengutipnya dalam jawaban.

        PERBAIKAN: Format sebelumnya menggunakan header "Sumber:" biasa tanpa
        nomor urut yang konsisten, sehingga LLM sulit merujuk sumber tertentu
        dan cenderung menjawab tanpa menyebut sumber. Sekarang label "[Konteks N]"
        konsisten dengan nomor urut sehingga generation prompt bisa menginstruksikan
        LLM untuk menyebut "[Konteks 1]", "[Konteks 2]" dst. dalam jawabannya.
        Selain itu, prefix "[Sumber: ...]" dari preprocessing dibersihkan dulu
        agar tidak terjadi duplikasi label di dalam teks chunk.

        Args:
            chunks    : List chunk dari retrieve()
            max_chars : Batas maksimum karakter konteks

        Returns:
            String konteks terformat dengan label sumber yang jelas
        """
        context_parts = []
        total_chars = 0

        for i, chunk in enumerate(chunks, 1):
            # Gunakan source_label yang sudah dibangun di retrieve(),
            # fallback ke section jika belum ada (e.g. dari retrieve_with_context)
            source_label = chunk.get(
                "source_label",
                chunk.get("section", "Basis Pengetahuan Stunting")
            )
            text = chunk.get("text", "")
            score = chunk.get("similarity_score", 0)

            # chunk["text"] sudah bersih sejak preprocessing.py diperbaiki
            # (tidak lagi menyisipkan [Sumber:] ke dalam text).
            # Teks ditampilkan apa adanya tanpa perlu strip tambahan.

            part = (
                f"[Konteks {i}]\n"
                f"Sumber   : {source_label}\n"
                f"Relevansi: {score:.2f}\n"
                f"---\n"
                f"{text}"
            )

            if total_chars + len(part) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    part = part[:remaining] + "..."
                    context_parts.append(part)
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n\n" + ("=" * 40 + "\n\n").join(context_parts)


# ─────────────────────────────────────────────
# ENTRY POINT (TEST)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    BASE_DIR = Path(__file__).resolve().parent.parent
    EMB_PATH = BASE_DIR / "embedding" / "stunting_embeddings.npz"
    META_PATH = BASE_DIR / "embedding" / "stunting_embeddings_metadata.json"

    if not EMB_PATH.exists():
        print(f"ERROR: Embedding tidak ditemukan di {EMB_PATH}")
        print("Jalankan: python embedding/create_embedding.py")
        sys.exit(1)

    from embedding.create_embedding import EmbeddingCreator, SBERT_INDONESIAN

    # Muat model dan index — WAJIB menggunakan model yang sama dengan saat pembuatan embedding
    # (SBERT_INDONESIAN = LazarusNLP/all-indo-e5-small-v4), bukan SBERT_MULTILINGUAL.
    # Mismatch model menyebabkan skor cosine similarity jatuh drastis karena
    # query vector dan corpus vector berada di ruang vektor yang berbeda.
    embedder = EmbeddingCreator(model_type="sbert", model_name=SBERT_INDONESIAN, device="cpu")
    embedder.load_model()

    # retriever = StuntingRetriever(top_k=5, similarity_threshold=0.05)
    retriever = StuntingRetriever(top_k=5, similarity_threshold=0.2)
    retriever.load_index(str(EMB_PATH), str(META_PATH))

    # Test query agar bisa bertanya di terminal
    test_queries = [
        "Apa itu MPASI dan kapan sebaiknya diberikan?",
        "layanan kesehatan di kendari"

    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print("="*60)
        results = retriever.retrieve(query, embedder, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r['similarity_score']:.4f}")
            print(f"    Seksi : {r.get('section', '')}")
            print(f"    Sumber: {r.get('source_label', '')}")
            print(f"    Teks  : {r.get('text', '')}")