"""
evaluate_embedding.py
=====================
Evaluasi Lengkap Embedding Vector untuk Pipeline RAG Stunting
Kota Kendari

Metrik yang dievaluasi:
  1. Intrinsic Quality     : normalisasi, distribusi, isotropy, uniformity
  2. Semantic Coherence    : intra vs inter-section cosine similarity
  3. Retrieval Simulation  : Precision@K, Recall@K, MRR, NDCG
  4. Near-Duplicate Check  : deteksi chunk redundan
  5. Coverage Analysis     : distribusi chunk per seksi

Cara pakai:
    python evaluate_embedding.py
    python evaluate_embedding.py --npz path/ke/embeddings.npz --meta path/ke/metadata.json
    python evaluate_embedding.py --report  # simpan laporan ke file
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────
# DEFAULT PATH (sesuaikan jika file di lokasi berbeda)
# ─────────────────────────────────────────────────────────────
DEFAULT_NPZ  = "D:\\Documents\\Stunting-chatbot-fix\\embedding\\stunting_embeddings.npz"
DEFAULT_META = "D:\\Documents\\Stunting-chatbot-fix\\embedding\\stunting_embeddings_metadata.json"

# ─────────────────────────────────────────────────────────────
# GROUND TRUTH QUERIES (untuk evaluasi retrieval)
# Tambah / edit sesuai domain chatbot Anda
# Format: {"query": ..., "relevant_sections": [...], "relevant_chunk_ids": [...]}
# relevant_chunk_ids bisa dikosongkan [] jika tidak tahu persis
# ─────────────────────────────────────────────────────────────
GROUND_TRUTH = [
    {
        "query": "apa itu stunting dan bagaimana definisinya",
        "relevant_sections": ["SEKSI 1: DEFINISI DAN KONSEP DASAR STUNTING"],
        "relevant_chunk_ids": ["chunk_0001", "chunk_0002"],
    },
    {
        "query": "berapa prevalensi stunting di Kota Kendari",
        "relevant_sections": ["SEKSI 2: PREVALENSI STUNTING DI INDONESIA DAN KOTA KENDARI",
                               "SEKSI 7: PENANGANAN STUNTING DI KOTA KENDARI"],
        "relevant_chunk_ids": ["chunk_0003"],
    },
    {
        "query": "apa faktor risiko yang menyebabkan stunting pada anak",
        "relevant_sections": ["SEKSI 3: FAKTOR RISIKO STUNTING"],
        "relevant_chunk_ids": [],
    },
    {
        "query": "dampak stunting terhadap perkembangan kognitif dan kecerdasan anak",
        "relevant_sections": ["SEKSI 4: DAMPAK STUNTING"],
        "relevant_chunk_ids": [],
    },
    {
        "query": "intervensi gizi spesifik untuk mencegah stunting",
        "relevant_sections": ["SEKSI 5: PENCEGAHAN STUNTING - INTERVENSI SPESIFIK GIZI"],
        "relevant_chunk_ids": [],
    },
    {
        "query": "program pencegahan stunting sensitif di tingkat komunitas",
        "relevant_sections": ["SEKSI 6: PENCEGAHAN STUNTING - INTERVENSI SENSITIF"],
        "relevant_chunk_ids": [],
    },
    {
        "query": "layanan penanganan stunting di puskesmas Kendari",
        "relevant_sections": ["SEKSI 7: PENANGANAN STUNTING DI KOTA KENDARI"],
        "relevant_chunk_ids": [],
    },
    {
        "query": "kebutuhan gizi ibu hamil untuk mencegah stunting",
        "relevant_sections": ["SEKSI 8: GIZI IBU HAMIL DAN MENYUSUI"],
        "relevant_chunk_ids": [],
    },
    {
        "query": "kapan waktu pemberian MPASI dan makanan pendamping ASI",
        "relevant_sections": ["SEKSI 9: Makanan Pendamping ASI (MPASI) DAN POLA MAKAN BALITA"],
        "relevant_chunk_ids": [],
    },
    {
        "query": "cara memantau pertumbuhan tinggi badan anak balita",
        "relevant_sections": ["SEKSI 10: PEMANTAUAN PERTUMBUHAN ANAK"],
        "relevant_chunk_ids": [],
    },
    {
        "query": "manfaat ASI eksklusif untuk mencegah stunting",
        "relevant_sections": ["SEKSI 11: Air Susu Ibu (ASI) EKSKLUSIF DAN MENYUSUI"],
        "relevant_chunk_ids": [],
    },
    {
        "query": "tanda bahaya stunting yang harus segera dibawa ke dokter",
        "relevant_sections": ["SEKSI 12: TANDA BAHAYA DAN KAPAN HARUS KE DOKTER"],
        "relevant_chunk_ids": [],
    },
    {
        "query": "mitos dan fakta tentang gizi anak dan stunting",
        "relevant_sections": ["SEKSI 13: INFORMASI TAMBAHAN DAN MITOS GIZI"],
        "relevant_chunk_ids": [],
    },
]


# ═════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════

def separator(title: str = "", width: int = 65) -> str:
    if title:
        pad = (width - len(title) - 2) // 2
        return f"\n{'─' * pad} {title} {'─' * pad}"
    return "─" * width


def badge(val: float, good: float, warn: float, higher_is_better: bool = True) -> str:
    """Kembalikan label BAIK / CUKUP / PERLU PERHATIAN berdasarkan threshold."""
    if higher_is_better:
        if val >= good:
            return "✅ BAIK"
        elif val >= warn:
            return "⚠️  CUKUP"
        else:
            return "❌ PERLU PERHATIAN"
    else:  # lower is better
        if val <= good:
            return "✅ BAIK"
        elif val <= warn:
            return "⚠️  CUKUP"
        else:
            return "❌ PERLU PERHATIAN"


# ═════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════

def load_data(npz_path: str, meta_path: str) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """Muat embedding dan metadata dari file."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float64)
    chunk_ids  = [str(c) for c in data["chunk_ids"]]

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    chunks = meta["chunks"]

    print(f"  Embedding dimuat  : shape={embeddings.shape}, dtype={embeddings.dtype}")
    print(f"  Total chunk       : {len(chunks)}")
    print(f"  Model             : {meta.get('model', 'N/A')}")
    print(f"  Embedding dim     : {meta.get('embedding_dim', embeddings.shape[1])}")
    return embeddings, chunk_ids, chunks


# ═════════════════════════════════════════════════════════════
# 2. INTRINSIC QUALITY METRICS
# ═════════════════════════════════════════════════════════════

def eval_intrinsic(embeddings: np.ndarray, chunks: List[Dict]) -> Dict:
    """Metrik kualitas intrinsik embedding."""
    results = {}

    # --- Normalisasi ---
    norms = np.linalg.norm(embeddings, axis=1)
    results["norm_mean"]   = float(norms.mean())
    results["norm_std"]    = float(norms.std())
    results["norm_min"]    = float(norms.min())
    results["norm_max"]    = float(norms.max())
    results["is_normalized"] = bool(np.allclose(norms, 1.0, atol=1e-4))

    # --- Statistik nilai embedding ---
    results["emb_mean"]    = float(embeddings.mean())
    results["emb_std"]     = float(embeddings.std())
    results["emb_min"]     = float(embeddings.min())
    results["emb_max"]     = float(embeddings.max())

    # --- Uniformity (Wang & Isola 2020) ---
    # log E[exp(-2||u-v||^2)]  → lebih negatif = lebih seragam = lebih baik
    n = len(embeddings)
    sq_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((embeddings[i] - embeddings[j]) ** 2)
            sq_dists.append(math.exp(-2 * d))
    results["uniformity_loss"] = float(math.log(sum(sq_dists) / len(sq_dists)))

    # --- Token count stats ---
    toks = [c.get("token_count", 0) for c in chunks]
    results["token_min"]   = int(min(toks))
    results["token_max"]   = int(max(toks))
    results["token_mean"]  = float(np.mean(toks))
    results["token_std"]   = float(np.std(toks))
    results["short_chunks"] = int(sum(1 for t in toks if t < 30))

    return results


def print_intrinsic(r: Dict):
    print(separator("1. INTRINSIC QUALITY"))

    # Normalisasi
    norm_status = "✅ SEMPURNA (L2 = 1.0)" if r["is_normalized"] else f"❌ TIDAK NORMAL (mean={r['norm_mean']:.4f})"
    print(f"\n  Normalisasi L2    : {norm_status}")
    print(f"    norm mean/std   : {r['norm_mean']:.6f} ± {r['norm_std']:.6f}")

    # Nilai embedding
    print(f"\n  Distribusi Nilai Embedding:")
    print(f"    mean            : {r['emb_mean']:.6f}")
    print(f"    std             : {r['emb_std']:.6f}")
    print(f"    range           : [{r['emb_min']:.4f}, {r['emb_max']:.4f}]")

    # Uniformity
    unif = r["uniformity_loss"]
    print(f"\n  Uniformity Loss   : {unif:.4f}  {badge(unif, -1.5, -1.0, higher_is_better=False)}")
    print(f"    (makin negatif makin seragam; target < -1.5)")

    # Token
    print(f"\n  Distribusi Token per Chunk:")
    print(f"    min / max / mean: {r['token_min']} / {r['token_max']} / {r['token_mean']:.1f} ± {r['token_std']:.1f}")
    if r["short_chunks"] > 0:
        print(f"    ⚠️  Chunk < 30 token: {r['short_chunks']} chunk — embedding kurang informatif, pertimbangkan merge")
    else:
        print(f"    ✅ Tidak ada chunk sangat pendek")


# ═════════════════════════════════════════════════════════════
# 3. SEMANTIC COHERENCE
# ═════════════════════════════════════════════════════════════

def eval_semantic_coherence(embeddings: np.ndarray, chunks: List[Dict]) -> Dict:
    """Intra-section vs inter-section cosine similarity."""
    sections = [c.get("section", "") for c in chunks]
    sim_matrix = cosine_similarity(embeddings)

    intra, inter = [], []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            s = sim_matrix[i, j]
            if sections[i] == sections[j]:
                intra.append(s)
            else:
                inter.append(s)

    # Per-section stats
    section_stats = {}
    unique_sections = list(set(sections))
    for sec in unique_sections:
        idx = [i for i, c in enumerate(chunks) if c.get("section", "") == sec]
        if len(idx) > 1:
            sub = sim_matrix[np.ix_(idx, idx)]
            np.fill_diagonal(sub, 0)
            vals = sub[sub != 0]
            section_stats[sec] = {
                "n_chunks": len(idx),
                "avg_sim": float(vals.mean()),
                "std_sim": float(vals.std()),
            }
        else:
            section_stats[sec] = {
                "n_chunks": len(idx),
                "avg_sim": None,
                "std_sim": None,
            }

    return {
        "intra_mean": float(np.mean(intra)) if intra else 0.0,
        "intra_std":  float(np.std(intra))  if intra else 0.0,
        "inter_mean": float(np.mean(inter)) if inter else 0.0,
        "inter_std":  float(np.std(inter))  if inter else 0.0,
        "separation_ratio": float(np.mean(intra) / np.mean(inter)) if inter and intra else 0.0,
        "cosine_max":  float(np.max(sim_matrix - np.eye(len(embeddings)))),
        "cosine_mean": float((sim_matrix.sum() - len(embeddings)) / (len(embeddings) * (len(embeddings) - 1))),
        "section_stats": section_stats,
    }


def print_semantic_coherence(r: Dict):
    print(separator("2. SEMANTIC COHERENCE"))

    sep = r["separation_ratio"]
    print(f"\n  Intra-section sim : {r['intra_mean']:.4f} ± {r['intra_std']:.4f}")
    print(f"  Inter-section sim : {r['inter_mean']:.4f} ± {r['inter_std']:.4f}")
    print(f"  Separation Ratio  : {sep:.3f}  {badge(sep, 1.5, 1.2)}")
    print(f"    (target > 1.5; Anda: {sep:.3f})")
    print(f"\n  Cosine Sim global :")
    print(f"    mean (off-diag) : {r['cosine_mean']:.4f}")
    print(f"    max pair        : {r['cosine_max']:.4f}")

    print(f"\n  Per-Section Coherence:")
    print(f"  {'Seksi':<48} {'N':>3}  {'Avg Sim':>8}  Status")
    print(f"  {'─'*48} {'─'*3}  {'─'*8}  {'─'*20}")
    for sec, stat in sorted(r["section_stats"].items(), key=lambda x: -(x[1]["n_chunks"])):
        n = stat["n_chunks"]
        avg = stat["avg_sim"]
        if avg is None:
            print(f"  {sec[:48]:<48} {n:>3}  {'  —':>8}  (hanya 1 chunk)")
        else:
            status = badge(avg, 0.65, 0.45)
            print(f"  {sec[:48]:<48} {n:>3}  {avg:>8.4f}  {status}")


# ═════════════════════════════════════════════════════════════
# 4. GEOMETRY / PCA
# ═════════════════════════════════════════════════════════════

def eval_geometry(embeddings: np.ndarray) -> Dict:
    """Analisis geometri ruang embedding dengan PCA."""
    pca = PCA()
    pca.fit(embeddings)
    var_ratio = pca.explained_variance_ratio_
    cumsum    = np.cumsum(var_ratio)

    dims_80 = int(np.searchsorted(cumsum, 0.80)) + 1
    dims_90 = int(np.searchsorted(cumsum, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumsum, 0.95)) + 1

    eigvals   = pca.explained_variance_
    isotropy  = float(eigvals.min() / eigvals.max()) if eigvals.max() > 0 else 0.0

    return {
        "dims_total":  embeddings.shape[1],
        "dims_80pct":  dims_80,
        "dims_90pct":  dims_90,
        "dims_95pct":  dims_95,
        "isotropy":    isotropy,
        "pc_variance_top5": [float(v) for v in var_ratio[:5]],
        "compression_ratio_90pct": dims_90 / embeddings.shape[1],
    }


def print_geometry(r: Dict):
    print(separator("3. GEOMETRY / PCA ANALYSIS"))

    print(f"\n  Total dimensi     : {r['dims_total']}")
    print(f"  Dims 80% variance : {r['dims_80pct']}  ({r['dims_80pct']/r['dims_total']*100:.1f}% dari total)")
    print(f"  Dims 90% variance : {r['dims_90pct']}  ({r['dims_90pct']/r['dims_total']*100:.1f}% dari total)")
    print(f"  Dims 95% variance : {r['dims_95pct']}  ({r['dims_95pct']/r['dims_total']*100:.1f}% dari total)")
    print(f"\n  Top-5 PC variance : {[f'{v:.3f}' for v in r['pc_variance_top5']]}")
    print(f"\n  Isotropy Score    : {r['isotropy']:.6f}")
    print(f"    (1.0 = sempurna isotropik; mendekati 0 = anisotropik — normal untuk domain khusus)")

    ratio = r["compression_ratio_90pct"]
    print(f"\n  Compression Ratio : {ratio:.3f}  — {ratio*100:.1f}% dimensi diperlukan untuk 90% informasi")
    if ratio < 0.15:
        print(f"    → Embedding memiliki redundansi tinggi; PCA compression bisa menghemat ruang")
    else:
        print(f"    → Distribusi informasi cukup merata di banyak dimensi")


# ═════════════════════════════════════════════════════════════
# 5. NEAR-DUPLICATE DETECTION
# ═════════════════════════════════════════════════════════════

def eval_near_duplicates(embeddings: np.ndarray, chunks: List[Dict],
                          threshold: float = 0.95) -> Dict:
    """Deteksi pasangan chunk yang terlalu mirip."""
    sim = cosine_similarity(embeddings)
    duplicates = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if sim[i, j] >= threshold:
                duplicates.append({
                    "chunk_a": chunks[i]["chunk_id"],
                    "chunk_b": chunks[j]["chunk_id"],
                    "section_a": chunks[i].get("section", ""),
                    "section_b": chunks[j].get("section", ""),
                    "similarity": float(sim[i, j]),
                })
    return {
        "threshold": threshold,
        "total_pairs": int(len(embeddings) * (len(embeddings) - 1) / 2),
        "duplicate_pairs": duplicates,
        "n_duplicates": len(duplicates),
    }


def print_near_duplicates(r: Dict):
    print(separator("4. NEAR-DUPLICATE DETECTION"))
    print(f"\n  Threshold         : sim ≥ {r['threshold']}")
    print(f"  Total pasangan    : {r['total_pairs']}")
    print(f"  Duplikat terdeteksi: {r['n_duplicates']}")
    if r["n_duplicates"] == 0:
        print(f"  ✅ Corpus bersih — tidak ada chunk near-duplicate")
    else:
        print(f"  ⚠️  Pasangan duplikat:")
        for d in r["duplicate_pairs"]:
            print(f"    {d['chunk_a']} ↔ {d['chunk_b']}  sim={d['similarity']:.4f}")
            print(f"      [{d['section_a'][:40]}]")
            print(f"      [{d['section_b'][:40]}]")


# ═════════════════════════════════════════════════════════════
# 6. RETRIEVAL EVALUATION
# ═════════════════════════════════════════════════════════════

def _precision_at_k(retrieved_sections: List[str], relevant_sections: List[str], k: int) -> float:
    top_k = retrieved_sections[:k]
    hits  = sum(1 for s in top_k if s in relevant_sections)
    return hits / k


def _recall_at_k(retrieved_sections: List[str], relevant_sections: List[str], k: int) -> float:
    top_k = retrieved_sections[:k]
    hits  = sum(1 for s in top_k if s in relevant_sections)
    return hits / len(relevant_sections) if relevant_sections else 0.0


def _reciprocal_rank(retrieved_sections: List[str], relevant_sections: List[str]) -> float:
    for i, s in enumerate(retrieved_sections, start=1):
        if s in relevant_sections:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(retrieved_sections: List[str], relevant_sections: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    def dcg(rels):
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))

    gains = [1.0 if s in relevant_sections else 0.0 for s in retrieved_sections[:k]]
    ideal = sorted(gains, reverse=True)
    idcg  = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def embed_query_tfidf(query: str, embeddings: np.ndarray, chunks: List[Dict]) -> np.ndarray:
    """
    Simulasi embedding query dengan keyword scoring.
    Di produksi, ganti dengan model.encode(query).
    """
    query_words = set(query.lower().split())
    scores = np.zeros(len(chunks))
    for i, chunk in enumerate(chunks):
        text_words = set(chunk.get("text", "").lower().split())
        keywords   = set(k.lower() for k in chunk.get("keywords", []))
        section    = chunk.get("section", "").lower()
        # TF-like: kata query yang muncul di teks + keyword + section
        match_text = len(query_words & text_words)
        match_kw   = len(query_words & keywords) * 2     # bobot lebih untuk keyword
        match_sec  = len(query_words & set(section.split())) * 1.5
        scores[i]  = match_text + match_kw + match_sec
    # Normalisasi
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores


def eval_retrieval(embeddings: np.ndarray, chunks: List[Dict],
                   ground_truth: List[Dict], k_values: List[int] = [1, 3, 5]) -> Dict:
    """Evaluasi retrieval menggunakan ground truth query."""
    sections_list = [c.get("section", "") for c in chunks]
    results_per_query = []

    for gt in ground_truth:
        query       = gt["query"]
        rel_sections = gt["relevant_sections"]
        rel_ids      = gt.get("relevant_chunk_ids", [])

        # Hitung score tiap chunk untuk query ini
        scores = embed_query_tfidf(query, embeddings, chunks)

        # Urutkan berdasarkan score (tertinggi = paling relevan)
        ranked_idx     = np.argsort(-scores)
        ranked_sections = [sections_list[i] for i in ranked_idx]
        ranked_ids      = [chunks[i]["chunk_id"] for i in ranked_idx]

        row = {
            "query": query,
            "relevant_sections": rel_sections,
            "ranked_sections":   ranked_sections[:max(k_values)],
            "ranked_chunk_ids":  ranked_ids[:max(k_values)],
        }

        for k in k_values:
            row[f"P@{k}"]    = _precision_at_k(ranked_sections, rel_sections, k)
            row[f"R@{k}"]    = _recall_at_k(ranked_sections, rel_sections, k)
            row[f"NDCG@{k}"] = _ndcg_at_k(ranked_sections, rel_sections, k)

        row["MRR"] = _reciprocal_rank(ranked_sections, rel_sections)

        # Chunk-level hit@3 (jika ada chunk_ids di ground truth)
        if rel_ids:
            row["hit@3_chunk"] = any(cid in rel_ids for cid in ranked_ids[:3])
        else:
            row["hit@3_chunk"] = None

        results_per_query.append(row)

    # Agregat
    agg = {}
    for k in k_values:
        agg[f"avg_P@{k}"]    = float(np.mean([r[f"P@{k}"]    for r in results_per_query]))
        agg[f"avg_R@{k}"]    = float(np.mean([r[f"R@{k}"]    for r in results_per_query]))
        agg[f"avg_NDCG@{k}"] = float(np.mean([r[f"NDCG@{k}"] for r in results_per_query]))
    agg["MRR"] = float(np.mean([r["MRR"] for r in results_per_query]))

    chunk_hits = [r["hit@3_chunk"] for r in results_per_query if r["hit@3_chunk"] is not None]
    agg["chunk_hit@3"] = float(np.mean(chunk_hits)) if chunk_hits else None

    return {
        "k_values": k_values,
        "n_queries": len(ground_truth),
        "per_query": results_per_query,
        "aggregate": agg,
    }


def print_retrieval(r: Dict):
    print(separator("5. RETRIEVAL EVALUATION"))
    print(f"\n  Jumlah query GT   : {r['n_queries']}")
    print(f"\n  {'Metrik':<20} {'Nilai':>8}  Status")
    print(f"  {'─'*20} {'─'*8}  {'─'*22}")

    agg = r["aggregate"]
    metrics = [
        (f"Precision@1",  agg.get("avg_P@1", 0),    0.80, 0.60),
        (f"Precision@3",  agg.get("avg_P@3", 0),    0.65, 0.45),
        (f"Precision@5",  agg.get("avg_P@5", 0),    0.55, 0.35),
        (f"Recall@3",     agg.get("avg_R@3", 0),    0.70, 0.50),
        (f"Recall@5",     agg.get("avg_R@5", 0),    0.80, 0.60),
        (f"NDCG@3",       agg.get("avg_NDCG@3", 0), 0.70, 0.50),
        (f"NDCG@5",       agg.get("avg_NDCG@5", 0), 0.75, 0.55),
        (f"MRR",          agg.get("MRR", 0),         0.80, 0.60),
    ]
    for name, val, good, warn in metrics:
        print(f"  {name:<20} {val:>8.4f}  {badge(val, good, warn)}")

    if agg.get("chunk_hit@3") is not None:
        print(f"  {'Chunk Hit@3':<20} {agg['chunk_hit@3']:>8.4f}  {badge(agg['chunk_hit@3'], 0.70, 0.50)}")

    print(f"\n  Per-Query Detail:")
    print(f"  {'Query':<45} {'P@1':>5} {'P@3':>5} {'R@3':>5} {'MRR':>5}  Top-1 Retrieved Section")
    print(f"  {'─'*45} {'─'*5} {'─'*5} {'─'*5} {'─'*5}  {'─'*30}")
    for q in r["per_query"]:
        top1 = q["ranked_sections"][0] if q["ranked_sections"] else "-"
        hit  = "✅" if q["ranked_sections"] and q["ranked_sections"][0] in q["relevant_sections"] else "❌"
        print(f"  {q['query'][:45]:<45} {q['P@1']:>5.2f} {q['P@3']:>5.2f} {q['R@3']:>5.2f} {q['MRR']:>5.2f}  {hit} {top1[:28]}")


# ═════════════════════════════════════════════════════════════
# 7. COVERAGE ANALYSIS
# ═════════════════════════════════════════════════════════════

def eval_coverage(chunks: List[Dict]) -> Dict:
    from collections import Counter
    sections = Counter(c.get("section", "?") for c in chunks)
    total    = len(chunks)
    main_chunks = sum(v for k, v in sections.items() if not k.startswith("Referensi"))
    ref_chunks  = total - main_chunks
    return {
        "total_chunks":  total,
        "main_chunks":   main_chunks,
        "ref_chunks":    ref_chunks,
        "section_dist":  dict(sections.most_common()),
        "n_sections":    len(sections),
    }


def print_coverage(r: Dict):
    print(separator("6. COVERAGE ANALYSIS"))
    print(f"\n  Total chunk       : {r['total_chunks']}")
    print(f"  Chunk konten utama: {r['main_chunks']}")
    print(f"  Chunk referensi   : {r['ref_chunks']}")
    print(f"  Jumlah seksi      : {r['n_sections']}")
    print(f"\n  Distribusi Chunk:")
    for sec, cnt in r["section_dist"].items():
        bar = "█" * cnt + "░" * max(0, 5 - cnt)
        print(f"    {bar} {cnt:>2}  {sec[:55]}")


# ═════════════════════════════════════════════════════════════
# 8. RINGKASAN & REKOMENDASI
# ═════════════════════════════════════════════════════════════

def print_summary(intrinsic: Dict, coherence: Dict, geometry: Dict,
                  dup: Dict, retrieval: Dict):
    print(separator("RINGKASAN & REKOMENDASI"))

    scores = {
        "Normalisasi L2":     (1.0 if intrinsic["is_normalized"] else 0.0, 1.0, 0.9),
        "Uniformity Loss":    (-intrinsic["uniformity_loss"], 1.5, 1.0),
        "Separation Ratio":   (coherence["separation_ratio"], 1.5, 1.2),
        "Precision@3":        (retrieval["aggregate"].get("avg_P@3", 0), 0.65, 0.45),
        "MRR":                (retrieval["aggregate"]["MRR"], 0.80, 0.60),
        "NDCG@3":             (retrieval["aggregate"].get("avg_NDCG@3", 0), 0.70, 0.50),
    }

    print(f"\n  {'Metrik':<25} {'Nilai':>8}  Status")
    print(f"  {'─'*25} {'─'*8}  {'─'*22}")
    for name, (val, good, warn) in scores.items():
        status = badge(val, good, warn)
        print(f"  {name:<25} {val:>8.4f}  {status}")

    print(f"\n  REKOMENDASI PERBAIKAN:")

    recs = []

    if intrinsic["short_chunks"] > 0:
        recs.append((
            "TINGGI",
            f"Merge {intrinsic['short_chunks']} chunk pendek (< 30 token)",
            "Chunk sangat pendek menghasilkan embedding kurang informatif. "
            "Merge ke chunk terdekat sebelum re-embed."
        ))

    if coherence["separation_ratio"] < 1.5:
        recs.append((
            "TINGGI",
            "Gunakan prefix E5-style saat embedding",
            "Tambahkan 'query: ' untuk query dan 'passage: ' untuk dokumen. "
            "Atau coba model LazarusNLP/all-indo-e5-small-v4 yang sudah ada di kode."
        ))

    if retrieval["aggregate"].get("avg_P@1", 0) < 0.8:
        recs.append((
            "SEDANG",
            "Buat ground truth dataset nyata (30–50 pasang query-dokumen)",
            "Evaluasi retrieval saat ini menggunakan keyword matching. "
            "Gunakan model.encode() asli untuk evaluasi lebih akurat."
        ))

    if dup["n_duplicates"] > 0:
        recs.append((
            "SEDANG",
            f"Hapus {dup['n_duplicates']} chunk near-duplicate",
            "Chunk duplikat membuang slot retrieval dan menambah noise."
        ))

    if coherence["intra_mean"] < 0.6:
        recs.append((
            "RENDAH",
            "Pertimbangkan chunking berbasis paragraf tematik",
            "Intra-section similarity masih di bawah 0.6. Chunking lebih kohesif "
            "akan meningkatkan semantic alignment."
        ))

    if not recs:
        print(f"  ✅ Tidak ada masalah kritis. Embedding siap digunakan.")
    else:
        for priority, title, detail in recs:
            icon = "🔴" if priority == "TINGGI" else ("🟡" if priority == "SEDANG" else "🟢")
            print(f"\n  {icon} [{priority}] {title}")
            print(f"     → {detail}")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def run_evaluation(npz_path: str, meta_path: str, save_report: bool = False):
    print("\n" + "═" * 65)
    print("  EVALUASI EMBEDDING — CHATBOT STUNTING KOTA KENDARI")
    print("═" * 65)

    print(separator("LOADING DATA"))
    print()
    embeddings, chunk_ids, chunks = load_data(npz_path, meta_path)

    print()
    r_intrinsic  = eval_intrinsic(embeddings, chunks)
    r_coherence  = eval_semantic_coherence(embeddings, chunks)
    r_geometry   = eval_geometry(embeddings)
    r_dup        = eval_near_duplicates(embeddings, chunks, threshold=0.95)
    r_retrieval  = eval_retrieval(embeddings, chunks, GROUND_TRUTH, k_values=[1, 3, 5])
    r_coverage   = eval_coverage(chunks)

    print_intrinsic(r_intrinsic)
    print_semantic_coherence(r_coherence)
    print_geometry(r_geometry)
    print_near_duplicates(r_dup)
    print_retrieval(r_retrieval)
    print_coverage(r_coverage)
    print_summary(r_intrinsic, r_coherence, r_geometry, r_dup, r_retrieval)

    print("\n" + "═" * 65)

    if save_report:
        report = {
            "intrinsic":  r_intrinsic,
            "coherence":  {k: v for k, v in r_coherence.items() if k != "section_stats"},
            "section_coherence": r_coherence["section_stats"],
            "geometry":   r_geometry,
            "duplicates": {k: v for k, v in r_dup.items() if k != "duplicate_pairs"},
            "duplicate_pairs": r_dup["duplicate_pairs"],
            "retrieval_aggregate": r_retrieval["aggregate"],
            "retrieval_per_query": [
                {k: v for k, v in q.items() if k not in ("ranked_sections", "ranked_chunk_ids")}
                for q in r_retrieval["per_query"]
            ],
            "coverage": {k: v for k, v in r_coverage.items() if k != "section_dist"},
        }
        out_path = Path("embedding_evaluation_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  📄 Laporan JSON disimpan ke: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi Embedding Stunting RAG")
    parser.add_argument("--npz",    default=DEFAULT_NPZ,  help="Path ke file .npz embedding")
    parser.add_argument("--meta",   default=DEFAULT_META, help="Path ke file metadata .json")
    parser.add_argument("--report", action="store_true",  help="Simpan laporan ke JSON")
    args = parser.parse_args()

    run_evaluation(args.npz, args.meta, save_report=args.report)