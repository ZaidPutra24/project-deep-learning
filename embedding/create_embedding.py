"""
create_embedding.py
===================
Modul Pembuatan Embedding Vector untuk Pipeline RAG Stunting

Menggunakan IndoBERT + Sentence-BERT untuk menghasilkan dense vector
representasi dari setiap chunk teks stunting.

Proyek  : Chatbot Konsultasi Risiko Stunting Kota Kendari
Model   : indobenchmark/indobert-base-p1  +  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("stunting.embedding")

# ─────────────────────────────────────────────
# KONSTANTA MODEL
# ─────────────────────────────────────────────
INDOBERT_MODEL = "indobenchmark/indobert-base-p1"
SBERT_MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SBERT_INDONESIAN = "LazarusNLP/all-indo-e5-small-v4"


class EmbeddingCreator:
    """
    Kelas untuk membuat embedding vector dari chunk teks stunting.

    Mendukung dua strategi:
        1. IndoBERT (transformers)  - embedding kontekstual, presisi tinggi
        2. Sentence-BERT (sbert)    - embedding semantik, lebih cepat

    Atau hybrid: rata-rata kedua embedding untuk retrieval terbaik.

    Contoh:
        creator = EmbeddingCreator(model_type="sbert")
        creator.load_model()
        embeddings = creator.embed_chunks(chunks)
        creator.save_embeddings(embeddings, "data/embeddings.npz")
    """

    def __init__(
        self,
        model_type: str = "sbert",
        model_name: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int = 512,
        normalize: bool = True,
    ):
        """
        Args:
            model_type    : "indobert", "sbert", atau "hybrid"
            model_name    : Override nama model (opsional)
            device        : "cpu" atau "cuda"
            batch_size    : Jumlah chunk per batch
            max_seq_length: Panjang token maksimum
            normalize     : Normalisasi L2 pada output embedding
        """
        self.model_type = model_type.lower()
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.normalize = normalize

        # Pilih nama model berdasarkan tipe
        if model_name:
            self.model_name = model_name
        elif self.model_type == "indobert":
            self.model_name = INDOBERT_MODEL
        elif self.model_type == "hybrid":
            self.model_name = SBERT_MULTILINGUAL
        else:
            self.model_name = SBERT_MULTILINGUAL

        self.model = None
        self.tokenizer = None

        logger.info(
            f"EmbeddingCreator: model_type={model_type} | model={self.model_name} | device={device}"
        )

    def load_model(self) -> None:
        """
        Memuat model embedding ke memori.
        Mendukung SentenceTransformer dan HuggingFace Transformers.
        """
        logger.info(f"Memuat model: {self.model_name}...")

        if self.model_type in ("sbert", "hybrid"):
            self._load_sbert()
        elif self.model_type == "indobert":
            self._load_indobert()
        else:
            raise ValueError(f"model_type tidak dikenal: {self.model_type}")

        logger.info("Model berhasil dimuat.")

    def _load_sbert(self) -> None:
        """Memuat Sentence-BERT menggunakan library sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.max_seq_length = self.max_seq_length
        except ImportError:
            raise ImportError("Install: pip install sentence-transformers")

    def _load_indobert(self) -> None:
        """Memuat IndoBERT menggunakan HuggingFace Transformers."""
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            import torch
            self.model.to(torch.device(self.device))
        except ImportError:
            raise ImportError("Install: pip install transformers torch")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Membuat embedding untuk satu teks.

        Args:
            text: Input teks

        Returns:
            numpy array 1D (vektor embedding)
        """
        if self.model is None:
            raise RuntimeError("Model belum dimuat. Panggil load_model() terlebih dahulu.")

        if self.model_type in ("sbert", "hybrid"):
            emb = self.model.encode(text, normalize_embeddings=self.normalize)
            return emb
        elif self.model_type == "indobert":
            return self._indobert_embed([text])[0]
        else:
            raise ValueError(f"model_type tidak dikenal: {self.model_type}")

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Membuat embedding untuk semua chunk secara batch.

        Args:
            chunks: List dictionary chunk (dari load_chunks_from_json)

        Returns:
            List chunk dengan tambahan field 'embedding'
        """
        if self.model is None:
            raise RuntimeError("Model belum dimuat. Panggil load_model() terlebih dahulu.")

        # Bangun teks embedding: section + subsection + teks bersih
        # chunk["text"] dari preprocessing tidak lagi mengandung [Sumber:] prefix
        # sehingga penggabungan dengan section tidak menyebabkan redundansi
        def _build_embed_text(chunk: Dict) -> str:
            section = chunk.get("section", "")
            subsection = chunk.get("subsection", "")
            body = chunk.get("text", "")
            prefix = f"{section} {subsection}".strip()
            return f"{prefix} {body}".strip() if prefix else body

        texts = [_build_embed_text(c) for c in chunks]
        total = len(texts)
        logger.info(f"Membuat embedding untuk {total} chunk (batch_size={self.batch_size})...")

        start_time = time.time()

        if self.model_type in ("sbert", "hybrid"):
            embeddings = self._sbert_embed_batch(texts)
        elif self.model_type == "indobert":
            embeddings = self._indobert_embed(texts)
        else:
            raise ValueError(f"model_type tidak dikenal: {self.model_type}")

        elapsed = time.time() - start_time
        logger.info(f"Embedding selesai: {total} chunk dalam {elapsed:.2f} detik")
        logger.info(f"Dimensi embedding: {embeddings.shape[1]}")

        # Gabungkan embedding dengan chunk data
        enriched = []
        for i, chunk in enumerate(chunks):
            enriched_chunk = dict(chunk)
            enriched_chunk["embedding"] = embeddings[i].tolist()
            enriched_chunk["embedding_dim"] = int(embeddings.shape[1])
            enriched_chunk["embedding_model"] = self.model_name
            enriched.append(enriched_chunk)

        return enriched

    def _sbert_embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embedding batch menggunakan Sentence-BERT."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings

    def _indobert_embed(self, texts: List[str]) -> np.ndarray:
        """
        Embedding batch menggunakan IndoBERT dengan mean pooling.
        Memproses dalam batch untuk efisiensi memori.
        """
        import torch

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(torch.device(self.device))

            with torch.no_grad():
                output = self.model(**encoded)

            # Mean pooling (average token embeddings, ignore padding)
            attention_mask = encoded["attention_mask"]
            token_embeddings = output.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            if self.normalize:
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / np.maximum(norms, 1e-9)

            all_embeddings.append(batch_embeddings)
            logger.debug(f"Batch {i // self.batch_size + 1} selesai")

        return np.vstack(all_embeddings)

    def save_embeddings(
        self,
        enriched_chunks: List[Dict],
        output_path: str,
    ) -> None:
        """
        Menyimpan embedding dalam format .npz (numpy compressed) dan
        metadata chunk dalam JSON terpisah.

        Args:
            enriched_chunks: List chunk dengan field 'embedding'
            output_path    : Path output (tanpa ekstensi, akan dibuat .npz dan .json)
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Pisahkan embedding dan metadata
        embeddings = np.array([c["embedding"] for c in enriched_chunks])
        chunk_ids = [c["chunk_id"] for c in enriched_chunks]

        metadata = []
        for c in enriched_chunks:
            meta = {k: v for k, v in c.items() if k != "embedding"}
            metadata.append(meta)

        # Simpan embedding array
        npz_path = output.with_suffix(".npz")
        np.savez_compressed(npz_path, embeddings=embeddings, chunk_ids=chunk_ids)
        logger.info(f"Embeddings disimpan: {npz_path} | shape={embeddings.shape}")

        # Simpan metadata
        meta_path = output.with_suffix(".json").with_name(
            output.stem + "_metadata.json"
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_chunks": len(metadata),
                    "embedding_dim": int(embeddings.shape[1]),
                    "model": self.model_name,
                    "chunks": metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Metadata disimpan: {meta_path}")

    def load_embeddings(self, path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Memuat embedding dari file .npz.

        Returns:
            Tuple (embeddings array, list chunk_ids)
        """
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"]
        chunk_ids = list(data["chunk_ids"])
        logger.info(f"Embeddings dimuat: shape={embeddings.shape}")
        return embeddings, chunk_ids
    
    # --- Tambahkan method ini di dalam class EmbeddingCreator ---

    def _smart_chunking(self, text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
        """
        Optimalisasi: Memecah teks berdasarkan semantik (paragraf/kalimat) 
        agar embedding memiliki konteks yang padat.
        """
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Overlap untuk menjaga kontinuitas informasi
                current_chunk = current_chunk[-overlap:] + p + " " if overlap < len(current_chunk) else p + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Import load_chunks_from_json — coba dari package dulu, fallback ke modul lokal
    try:
        from data.preprocessing import load_chunks_from_json
    except ImportError:
        from preprocessing import load_chunks_from_json

    BASE_DIR = Path(__file__).resolve().parent.parent
    CHUNKS_PATH = BASE_DIR / "data" / "chunks.json"
    OUTPUT_PATH = BASE_DIR / "embedding" / "stunting_embeddings"

    if not CHUNKS_PATH.exists():
        print(f"ERROR: File chunks tidak ditemukan di {CHUNKS_PATH}")
        print("Jalankan preprocessing terlebih dahulu: python data/preprocessing.py")
        sys.exit(1)

    # Muat chunks
    chunks = load_chunks_from_json(str(CHUNKS_PATH))
    print(f"Chunk dimuat: {len(chunks)}")

    # Buat embedding — gunakan SBERT_INDONESIAN (all-indo-e5-small-v4) sebagai default
    # karena dilatih khusus pada korpus Indonesia sehingga skor cosine similarity lebih tinggi
    creator = EmbeddingCreator(
        model_type="sbert",
        model_name=SBERT_INDONESIAN,
        device="cpu",
        batch_size=32,
        normalize=True,
    )
    creator.load_model()

    enriched = creator.embed_chunks(chunks)
    creator.save_embeddings(enriched, str(OUTPUT_PATH))

    print(f"\nEmbedding selesai!")
    print(f"Dimensi: {enriched[0]['embedding_dim']}")
    print(f"Total chunk: {len(enriched)}")