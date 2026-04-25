"""
preprocessing.py
================
Modul Preprocessing Data untuk Chatbot Stunting RAG
Meliputi: Chunking, Cleaning, dan Normalisasi Teks

Proyek  : Chatbot Konsultasi Risiko Stunting Kota Kendari
Model   : IndoBERT + Sentence-BERT + LLM (RAG Pipeline)
Author  : Sistem Chatbot Stunting
Version : 1.0
"""

import re
import json
import os
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ─────────────────────────────────────────────
# SETUP LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stunting.preprocessing")


# ─────────────────────────────────────────────
# DATACLASS: CHUNK
# ─────────────────────────────────────────────
@dataclass
class Chunk:
    """Representasi satu potongan teks (chunk) dari korpus."""

    chunk_id: str
    text: str
    section: str
    subsection: str
    source_file: str
    char_start: int
    char_end: int
    token_count: int
    keywords: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────
# KELAS UTAMA: StuntingPreprocessor
# ─────────────────────────────────────────────
class StuntingPreprocessor:
    """
    Kelas utama untuk preprocessing korpus pengetahuan stunting.

    Pipeline:
        1. load_corpus()   → baca raw text
        2. clean_text()    → bersihkan teks
        3. normalize_text()→ normalisasi
        4. chunk_text()    → potong menjadi chunk
        5. enrich_chunks() → tambah keywords & metadata
        6. save_chunks()   → simpan ke JSON

    Contoh Penggunaan:
        preprocessor = StuntingPreprocessor()
        chunks = preprocessor.run_pipeline(
            input_path="data/stunting_docs.txt",
            output_path="data/chunks.json"
        )
    """

    # Pola seksi dalam dokumen
    SECTION_PATTERN = re.compile(
        r"\[SEKSI\s+(\d+)\]\s+(.*?)(?=\[SEKSI|\Z)", re.DOTALL | re.IGNORECASE
    )
    SUBSECTION_PATTERN = re.compile(r"^([A-Z]\.\s+.+)$", re.MULTILINE)

    # Kamus normalisasi singkatan medis dan istilah gizi
    ABBREVIATION_MAP = {
        r"\bTTD\b": "Tablet Tambah Darah",
        r"\bBBLR\b": "Berat Badan Lahir Rendah",
        r"\bASI\b": "Air Susu Ibu",
        r"\bIMD\b": "Inisiasi Menyusu Dini",
        r"\bMPASI\b": "Makanan Pendamping ASI",
        r"\bANC\b": "Antenatal Care",
        r"\bKEK\b": "Kurang Energi Kronis",
        r"\bLiLA\b": "Lingkar Lengan Atas",
        r"\bHPK\b": "Hari Pertama Kehidupan",
        r"\bIMT\b": "Indeks Massa Tubuh",
        r"\bSD\b": "Standar Deviasi",
        r"\bHb\b": "Hemoglobin",
        r"\bKMS\b": "Kartu Menuju Sehat",
        r"\bPMT\b": "Pemberian Makanan Tambahan",
        r"\bBB\b": "Berat Badan",
        r"\bTB\b": "Tinggi Badan",
        r"\bPB\b": "Panjang Badan",
        r"\bBAB\b": "Buang Air Besar",
        r"\bBAK\b": "Buang Air Kecil",
        r"\bISPA\b": "Infeksi Saluran Pernapasan Akut",
        r"\bSTBM\b": "Sanitasi Total Berbasis Masyarakat",
        r"\bCTPS\b": "Cuci Tangan Pakai Sabun",
        r"\bJKN\b": "Jaminan Kesehatan Nasional",
        r"\bPKH\b": "Program Keluarga Harapan",
        r"\bBPNT\b": "Bantuan Pangan Non-Tunai",
        r"\bTPPS\b": "Tim Percepatan Penurunan Stunting",
        r"\bDHA\b": "Docosahexaenoic Acid",
        r"\bARA\b": "Arachidonic Acid",
        r"\bIgA\b": "Imunoglobulin A",
        r"\bUHC\b": "Universal Health Coverage",
        r"\bRUTF\b": "Ready to Use Therapeutic Food",
        r"\bNTD\b": "Neural Tube Defect",
        r"\bLAM\b": "Lactational Amenorrhea Method",
        r"\bTFC\b": "Therapeutic Feeding Center",
    }

    # Kata kunci domain stunting untuk ekstraksi otomatis
    DOMAIN_KEYWORDS = {
        "stunting": ["stunting", "pendek", "sangat pendek", "gagal tumbuh"],
        "gizi": ["gizi", "nutrisi", "malnutrisi", "defisiensi", "kekurangan gizi"],
        "asi": ["asi", "menyusui", "laktasi", "kolostrum", "air susu ibu"],
        "mpasi": ["mpasi", "makanan pendamping", "bubur bayi", "finger food"],
        "ibu_hamil": ["ibu hamil", "kehamilan", "hamil", "trimester", "janin"],
        "balita": ["balita", "bayi", "anak", "batita", "toddler"],
        "gizi_buruk": ["gizi buruk", "marasmus", "kwashiorkor", "kurus"],
        "anemia": ["anemia", "hemoglobin", "hb rendah", "kurang darah"],
        "posyandu": ["posyandu", "kader", "penimbangan", "kms"],
        "puskesmas": ["puskesmas", "tenaga kesehatan", "dokter", "bidan", "ahli gizi"],
        "kendari": ["kendari", "sulawesi tenggara", "kota kendari"],
        # Tambahan berdasarkan evaluasi — topik yang sering miss-retrieved
        "dampak": ["dampak", "akibat", "konsekuensi", "kognitif", "kecerdasan",
                   "produktivitas", "penyakit kronis", "jangka panjang"],
        "intervensi_spesifik": ["intervensi spesifik", "suplementasi", "tablet tambah darah",
                                "vitamin", "mineral", "tatalaksana", "terapi gizi"],
        "intervensi_sensitif": ["intervensi sensitif", "sanitasi", "air bersih", "ctps",
                                "ketahanan pangan", "perlindungan sosial", "pkh", "bpnt"],
        "gizi_ibu_hamil": ["gizi ibu hamil", "nutrisi kehamilan", "asam folat", "zat besi",
                           "kalsium", "protein ibu", "suplemen kehamilan"],
    }

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        min_chunk_length: int = 50,
        expand_abbreviations: bool = True,
        preserve_numbers: bool = True,
    ):
        """
        Inisialisasi preprocessor.

        Args:
            chunk_size        : Ukuran target chunk dalam token (kata)
            chunk_overlap     : Jumlah token overlap antar chunk
            min_chunk_length  : Panjang minimum chunk (karakter)
            expand_abbreviations: Apakah singkatan di-expand
            preserve_numbers  : Pertahankan angka dan satuan gizi

        PERBAIKAN (dari evaluasi):
        - chunk_size diturunkan 400 → 200: evaluasi menunjukkan token mean=138
          per chunk, artinya chunk 400 token sering menggabungkan beberapa sub-topik
          berbeda dalam satu chunk. Chunk yang terlalu lebar menyebabkan cross-topic
          contamination: SEKSI 13 (Mitos/Info) yang panjang dan membahas banyak hal
          selalu menang cosine similarity meski bukan sumber yang tepat.
        - chunk_overlap diturunkan 80 → 50: overlap proporsional dengan chunk_size baru.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.expand_abbreviations = expand_abbreviations
        self.preserve_numbers = preserve_numbers

        logger.info(
            f"StuntingPreprocessor diinisialisasi | chunk_size={chunk_size} | "
            f"overlap={chunk_overlap} | expand_abbrev={expand_abbreviations}"
        )

    # ─────────────────────────────────────────
    # 1. LOAD CORPUS
    # ─────────────────────────────────────────
    def load_corpus(self, filepath: str) -> str:
        """
        Memuat file teks korpus stunting.

        Args:
            filepath: Path ke file teks.

        Returns:
            String isi teks mentah.

        Raises:
            FileNotFoundError: Jika file tidak ditemukan.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File korpus tidak ditemukan: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            corpus = f.read()

        logger.info(f"Korpus berhasil dimuat: {filepath} ({len(corpus):,} karakter)")
        return corpus

    # ─────────────────────────────────────────
    # 2. CLEAN TEXT
    # ─────────────────────────────────────────
    def clean_text(self, text: str) -> str:
        """
        Membersihkan teks korpus dari noise dan karakter tidak diinginkan.

        Tahapan:
            - Normalisasi encoding Unicode
            - Hapus karakter kontrol
            - Standardisasi tanda baca
            - Bersihkan spasi berlebih
            - Hapus baris pembatas (=====)
            - Normalisasi baris baru

        Args:
            text: Teks mentah.

        Returns:
            Teks yang sudah dibersihkan.
        """
        logger.debug("Memulai proses cleaning teks...")

        # Normalisasi Unicode (NFC: Canonical Decomposition + Composition)
        text = unicodedata.normalize("NFC", text)

        # Hapus BOM (Byte Order Mark) jika ada
        text = text.lstrip("\ufeff")

        # Hapus karakter kontrol kecuali newline dan tab
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Hapus baris pembatas dekoratif (===, ---, *** dst.)
        text = re.sub(r"^[=\-*#~]{3,}\s*$", "", text, flags=re.MULTILINE)

        # Hapus spasi di awal/akhir setiap baris
        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines)

        # Standardisasi tanda kutip
        text = re.sub(r"[''‛]", "'", text)
        text = re.sub(r'[""‟„]', '"', text)

        # Standardisasi strip/dash
        text = re.sub(r"[–—]", "-", text)

        # Standardisasi bullet points menjadi tanda strip
        text = re.sub(r"^[\•\●\▪\▸\→\►\✓\✔\-]\s+", "- ", text, flags=re.MULTILINE)

        # Hapus spasi berlebih (lebih dari 1 spasi menjadi 1)
        text = re.sub(r" {2,}", " ", text)

        # Reduce lebih dari 2 newline menjadi 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Hapus tab dan ganti dengan spasi
        text = re.sub(r"\t", " ", text)

        text = text.strip()

        logger.info(f"Cleaning selesai: {len(text):,} karakter tersisa")
        return text

    # ─────────────────────────────────────────
    # 3. NORMALIZE TEXT
    # ─────────────────────────────────────────
    def normalize_text(self, text: str) -> str:
        """
        Normalisasi teks untuk meningkatkan kualitas retrieval.

        Tahapan:
            - Expand singkatan domain stunting (opsional)
            - Normalisasi angka dan satuan gizi
            - Standardisasi ejaan umum bahasa Indonesia
            - Lowercase untuk konsistensi (dengan pengecualian)

        Args:
            text: Teks yang sudah dibersihkan.

        Returns:
            Teks yang sudah dinormalisasi.
        """
        logger.debug("Memulai normalisasi teks...")

        # Expand singkatan jika diaktifkan
        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)

        # Normalisasi satuan gizi umum
        text = self._normalize_units(text)

        # Perbaiki kesalahan umum ejaan
        text = self._fix_common_typos(text)

        logger.info("Normalisasi teks selesai.")
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """
        Memperluas singkatan medis dan gizi agar lebih mudah di-match
        oleh model embedding.
        
        Contoh: "TTD" → "Tablet Tambah Darah (TTD)"

        PERBAIKAN: Regex sebelumnya menggunakan literal string r"\\b(\w+)\\b"
        yang tidak pernah match karena mencari karakter backslash, bukan
        word boundary. Sekarang ekstraksi singkatan dilakukan dari pola
        langsung dengan strip \b.
        """
        for pattern, expansion in self.ABBREVIATION_MAP.items():
            # Ekstrak nama singkatan dari pattern regex "\bXXX\b" → "XXX"
            short = pattern.replace(r"\b", "").strip()
            replacement = f"{expansion} ({short})"
            text = re.sub(pattern, replacement, text)
        return text

    def _normalize_units(self, text: str) -> str:
        """
        Normalisasi satuan gizi, medis, dan antropometri.
        Contoh: "2500g" → "2500 g", "180cm" → "180 cm"
        """
        # Satuan berat tanpa spasi → tambah spasi
        text = re.sub(r"(\d+)(mg|mcg|g|kg|IU|kkal|kcal)\b", r"\1 \2", text)
        # Satuan panjang/tinggi
        text = re.sub(r"(\d+)(cm|mm|m)\b", r"\1 \2", text)
        # Satuan volume
        text = re.sub(r"(\d+)(ml|liter|L)\b", r"\1 \2", text)
        # Persentase tanpa spasi
        text = re.sub(r"(\d+)(%)", r"\1 \2", text)
        return text

    def _fix_common_typos(self, text: str) -> str:
        """
        Memperbaiki kesalahan ejaan umum dalam teks kesehatan Indonesia.
        """
        corrections = {
            r"\bstuntig\b": "stunting",
            r"\bstuntiing\b": "stunting",
            r"\bgisi\b": "gizi",
            r"\bnutirisi\b": "nutrisi",
            r"\bkehamlan\b": "kehamilan",
            r"\bposayandu\b": "posyandu",
            r"\bpusekesmas\b": "puskesmas",
            r"\bbalits\b": "balita",
            r"\bimunisas\b": "imunisasi",
        }
        for pattern, correction in corrections.items():
            text = re.sub(pattern, correction, text, flags=re.IGNORECASE)
        return text

    # ─────────────────────────────────────────
    # 4. CHUNK TEXT
    # ─────────────────────────────────────────
    def chunk_text(self, text: str, source_file: str = "stunting_docs.txt") -> List[Chunk]:
        """
        Memotong teks menjadi chunk yang dapat diindeks untuk retrieval.

        Strategi chunking:
            1. Chunking berbasis seksi (header [SEKSI N])
            2. Di dalam setiap seksi, chunking berbasis paragraf
            3. Chunk yang terlalu panjang dipotong ulang dengan sliding window
            4. Chunk yang terlalu pendek digabungkan dengan sebelumnya

        Args:
            text       : Teks yang sudah dinormalisasi.
            source_file: Nama file sumber untuk metadata.

        Returns:
            List objek Chunk.
        """
        logger.info("Memulai proses chunking...")
        chunks: List[Chunk] = []
        chunk_counter = 0

        # Pisahkan berdasarkan seksi
        sections = self._split_by_sections(text)

        for section_num, section_title, section_text in sections:
            section_label = f"SEKSI {section_num}: {section_title}"
            subsections = self._split_by_subsections(section_text)

            for sub_label, sub_text in subsections:
                # Pisahkan menjadi paragraf
                paragraphs = self._split_into_paragraphs(sub_text)
                buffer = ""
                buffer_start = 0

                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue

                    # Hitung token sederhana (jumlah kata)
                    combined = (buffer + "\n\n" + para).strip() if buffer else para
                    token_count = len(combined.split())

                    if token_count <= self.chunk_size:
                        buffer = combined
                    else:
                        # Simpan buffer saat ini sebagai chunk
                        if buffer and len(buffer) >= self.min_chunk_length:
                            chunk_counter += 1
                            char_start = text.find(buffer[:50])
                            chunk = Chunk(
                                chunk_id=f"chunk_{chunk_counter:04d}",
                                text=buffer.strip(),
                                section=section_label,
                                subsection=sub_label,
                                source_file=source_file,
                                char_start=max(0, char_start),
                                char_end=max(0, char_start) + len(buffer),
                                token_count=len(buffer.split()),
                            )
                            chunks.append(chunk)

                        # Mulai buffer baru dengan overlap
                        overlap_text = self._get_overlap_text(buffer)
                        buffer = (overlap_text + "\n\n" + para).strip() if overlap_text else para

                # Simpan sisa buffer
                if buffer and len(buffer) >= self.min_chunk_length:
                    chunk_counter += 1
                    char_start = text.find(buffer[:50])
                    chunk = Chunk(
                        chunk_id=f"chunk_{chunk_counter:04d}",
                        text=buffer.strip(),
                        section=section_label,
                        subsection=sub_label,
                        source_file=source_file,
                        char_start=max(0, char_start),
                        char_end=max(0, char_start) + len(buffer),
                        token_count=len(buffer.split()),
                    )
                    chunks.append(chunk)

        logger.info(f"Chunking selesai: {len(chunks)} chunk dihasilkan")
        return chunks

    def _split_by_sections(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Memisahkan teks berdasarkan header seksi [SEKSI N].
        
        Returns:
            List tuple: (nomor_seksi, judul_seksi, teks_seksi)
        """
        pattern = re.compile(
            r"\[SEKSI\s+(\d+)\]\s+(.*?)(?=\[SEKSI\s+\d+\]|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        sections = []
        for match in pattern.finditer(text):
            num = match.group(1)
            title = match.group(2).split("\n")[0].strip()
            body = match.group(2)
            sections.append((num, title, body))

        if not sections:
            # Fallback: perlakukan seluruh teks sebagai satu seksi
            logger.warning("Tidak ada header [SEKSI] ditemukan. Memproses sebagai satu blok.")
            sections = [("0", "Dokumen Stunting", text)]

        return sections

    def _split_by_subsections(self, text: str) -> List[Tuple[str, str]]:
        """
        Memisahkan teks seksi berdasarkan sub-header (A., B., C., dll.)
        
        Returns:
            List tuple: (label_subseksi, teks_subseksi)
        """
        pattern = re.compile(r"^([A-Z]\.\s+[^\n]+)$", re.MULTILINE)
        parts = pattern.split(text)

        subsections = []
        if len(parts) <= 1:
            subsections.append(("Umum", text.strip()))
        else:
            # parts[0] = teks sebelum subseksi pertama (header seksi)
            if parts[0].strip():
                subsections.append(("Pendahuluan", parts[0].strip()))
            for i in range(1, len(parts), 2):
                label = parts[i].strip() if i < len(parts) else ""
                content = parts[i + 1].strip() if (i + 1) < len(parts) else ""
                if content:
                    subsections.append((label, content))

        return subsections

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Memisahkan teks menjadi paragraf berdasarkan baris kosong.
        Setiap item daftar (bullet) dianggap satu unit.
        """
        # Pisahkan berdasarkan dua newline atau lebih
        paragraphs = re.split(r"\n{2,}", text)
        result = []
        for para in paragraphs:
            para = para.strip()
            if para:
                result.append(para)
        return result

    def _get_overlap_text(self, text: str) -> str:
        """
        Mengambil bagian akhir teks untuk overlap antar chunk.
        Mengambil kalimat-kalimat terakhir hingga overlap_size token.
        """
        words = text.split()
        if len(words) <= self.chunk_overlap:
            return text
        overlap_words = words[-self.chunk_overlap:]
        return " ".join(overlap_words)

    # ─────────────────────────────────────────
    # 5. ENRICH CHUNKS
    # ─────────────────────────────────────────
    def enrich_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Memperkaya setiap chunk dengan keyword domain dan metadata tambahan.

        Args:
            chunks: List chunk hasil pemrosesan.

        Returns:
            List chunk yang sudah diperkaya.
        """
        logger.info(f"Memperkaya {len(chunks)} chunk dengan keyword dan metadata...")

        for chunk in chunks:
            # Ekstrak keyword domain
            chunk.keywords = self._extract_keywords(chunk.text)
            # Tambah metadata
            chunk.metadata = self._build_metadata(chunk)

        logger.info("Enrichment selesai.")
        return chunks

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Mengekstrak keyword domain stunting dari teks chunk.
        """
        text_lower = text.lower()
        found_keywords = []
        for category, keywords in self.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    found_keywords.append(category)
                    break  # Satu kategori cukup satu match
        return list(set(found_keywords))

    def _build_metadata(self, chunk: Chunk) -> Dict:
        """
        Membangun metadata chunk untuk filtering dalam retrieval.
        display_text menyimpan representasi lengkap dengan label sumber
        yang digunakan saat menampilkan jawaban ke pengguna.
        """
        display_text = (
            f"[Sumber: {chunk.section} | {chunk.subsection}]\n{chunk.text}"
        )
        return {
            "has_numbers": bool(re.search(r"\d+", chunk.text)),
            "has_list": bool(re.search(r"^\s*-\s", chunk.text, re.MULTILINE)),
            "word_count": len(chunk.text.split()),
            "char_count": len(chunk.text),
            "language": "id",  # Indonesia
            "domain": "stunting_health",
            "location": "kendari_sultra" if "kendari" in chunk.text.lower() else "nasional",
            "display_text": display_text,
        }

    # ─────────────────────────────────────────
    # 6. LOAD REFERENCES
    # ─────────────────────────────────────────
    def load_references(self, filepath: str) -> Dict:
        """
        Memuat file referensi ilmiah JSON.

        Args:
            filepath: Path ke file JSON referensi.

        Returns:
            Dictionary berisi data referensi.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"File referensi tidak ditemukan: {filepath}")
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(
            f"Referensi berhasil dimuat: {len(data.get('references', []))} referensi"
        )
        return data

    def references_to_chunks(self, references_data: Dict) -> List[Chunk]:
        """
        Mengonversi referensi ilmiah menjadi chunk teks untuk diindeks.
        Setiap referensi menjadi satu chunk dengan format yang kaya.

        Args:
            references_data: Dictionary hasil load_references().

        Returns:
            List Chunk dari referensi ilmiah.
        """
        chunks = []
        refs = references_data.get("references", [])

        for i, ref in enumerate(refs, start=1):
            # Bangun teks representasi referensi
            authors = ", ".join(ref.get("penulis", []))
            tahun = ref.get("tahun", "")
            judul = ref.get("judul", "")
            ringkasan = ref.get("ringkasan", "")
            poin = ref.get("poin_kunci", [])
            poin_str = "\n".join(f"- {p}" for p in poin)

            text = (
                f"Referensi: {judul} ({tahun})\n"
                f"Penulis: {authors}\n"
                f"Kategori: {ref.get('kategori', '')}\n"
                f"Ringkasan: {ringkasan}\n"
                f"Poin Kunci:\n{poin_str}"
            )

            chunk = Chunk(
                chunk_id=f"ref_{ref.get('id', i):>06}",
                text=text.strip(),
                section=f"Referensi Ilmiah - {ref.get('kategori', 'Umum')}",
                subsection=ref.get("kategori", ""),
                source_file="stunting_references.json",
                char_start=0,
                char_end=len(text),
                token_count=len(text.split()),
                keywords=ref.get("relevansi_topik", []),
                metadata={
                    "ref_id": ref.get("id"),
                    "tahun": tahun,
                    "doi": ref.get("doi"),
                    "url": ref.get("url"),
                    "language": "id",
                    "domain": "scientific_reference",
                },
            )
            chunks.append(chunk)

        logger.info(f"Referensi dikonversi: {len(chunks)} chunk referensi")
        return chunks

    # ─────────────────────────────────────────
    # 7. SAVE CHUNKS
    # ─────────────────────────────────────────
    def save_chunks(self, chunks: List[Chunk], output_path: str) -> None:
        """
        Menyimpan chunk ke file JSON terstruktur.

        Args:
            chunks      : List chunk yang akan disimpan.
            output_path : Path file output JSON.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "total_chunks": len(chunks),
                "chunk_size_target": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "expand_abbreviations": self.expand_abbreviations,
            },
            "chunks": [c.to_dict() for c in chunks],
        }

        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Chunks disimpan: {output_path} ({len(chunks)} chunk)")

    # ─────────────────────────────────────────
    # 8. PIPELINE UTAMA
    # ─────────────────────────────────────────
    def run_pipeline(
        self,
        docs_path: str,
        refs_path: str,
        output_path: str,
    ) -> List[Chunk]:
        """
        Menjalankan seluruh pipeline preprocessing dari awal hingga akhir.

        Args:
            docs_path  : Path ke stunting_docs.txt
            refs_path  : Path ke stunting_references.json
            output_path: Path output chunks.json

        Returns:
            List semua chunk yang dihasilkan.
        """
        logger.info("=" * 60)
        logger.info("MEMULAI PIPELINE PREPROCESSING STUNTING")
        logger.info("=" * 60)

        all_chunks: List[Chunk] = []

        # --- Proses Dokumen Utama ---
        logger.info(">>> TAHAP 1: Memuat dokumen utama...")
        raw_text = self.load_corpus(docs_path)

        logger.info(">>> TAHAP 2: Membersihkan teks...")
        clean = self.clean_text(raw_text)

        logger.info(">>> TAHAP 3: Menormalisasi teks...")
        normalized = self.normalize_text(clean)

        logger.info(">>> TAHAP 4: Memotong teks menjadi chunk...")
        doc_chunks = self.chunk_text(normalized, source_file=Path(docs_path).name)

        logger.info(">>> TAHAP 5: Memperkaya chunk dengan metadata...")
        doc_chunks = self.enrich_chunks(doc_chunks)
        all_chunks.extend(doc_chunks)

        # --- Proses Referensi Ilmiah ---
        logger.info(">>> TAHAP 6: Memuat dan mengonversi referensi ilmiah...")
        refs_data = self.load_references(refs_path)
        if refs_data:
            ref_chunks = self.references_to_chunks(refs_data)
            all_chunks.extend(ref_chunks)

        # --- Simpan Hasil ---
        logger.info(">>> TAHAP 7: Menyimpan semua chunk...")
        self.save_chunks(all_chunks, output_path)

        logger.info("=" * 60)
        logger.info(f"PIPELINE SELESAI: {len(all_chunks)} chunk total")
        logger.info(f"  - Chunk dokumen : {len(doc_chunks)}")
        logger.info(f"  - Chunk referensi: {len(all_chunks) - len(doc_chunks)}")
        logger.info(f"  - Output: {output_path}")
        logger.info("=" * 60)

        return all_chunks


# ─────────────────────────────────────────────
# FUNGSI UTILITAS TAMBAHAN
# ─────────────────────────────────────────────
def load_chunks_from_json(filepath: str) -> List[Dict]:
    """
    Memuat chunk dari file JSON yang sudah disimpan.

    Args:
        filepath: Path ke file chunks JSON.

    Returns:
        List dictionary chunk.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chunks", [])


def print_chunk_statistics(chunks: List[Chunk]) -> None:
    """
    Mencetak statistik ringkas dari chunk yang dihasilkan.
    """
    if not chunks:
        print("Tidak ada chunk untuk ditampilkan.")
        return

    token_counts = [c.token_count for c in chunks]
    char_counts = [len(c.text) for c in chunks]

    print("\n" + "=" * 50)
    print("STATISTIK CHUNK")
    print("=" * 50)
    print(f"Total chunk        : {len(chunks)}")
    print(f"Rata-rata token    : {sum(token_counts)/len(token_counts):.1f}")
    print(f"Min token          : {min(token_counts)}")
    print(f"Max token          : {max(token_counts)}")
    print(f"Rata-rata karakter : {sum(char_counts)/len(char_counts):.1f}")
    print(f"Total karakter     : {sum(char_counts):,}")

    # Distribusi per seksi
    sections = {}
    for c in chunks:
        sec = c.section[:40]
        sections[sec] = sections.get(sec, 0) + 1

    print("\nDistribusi per Seksi:")
    for sec, count in sorted(sections.items(), key=lambda x: -x[1]):
        print(f"  {sec:<42} : {count} chunk")
    print("=" * 50)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Tentukan path relatif dari lokasi script ini
    BASE_DIR = Path(__file__).resolve().parent.parent
    DOCS_PATH = BASE_DIR / "data" / "stunting_docs.txt"
    REFS_PATH = BASE_DIR / "data" / "stunting_references.json"
    OUTPUT_PATH = BASE_DIR / "data" / "chunks.json"

    print(f"Base directory : {BASE_DIR}")
    print(f"Dokumen utama  : {DOCS_PATH}")
    print(f"Referensi      : {REFS_PATH}")
    print(f"Output chunks  : {OUTPUT_PATH}")

    # Inisialisasi dan jalankan pipeline
    preprocessor = StuntingPreprocessor(
        chunk_size=200,
        chunk_overlap=50,
        min_chunk_length=50,
        expand_abbreviations=True,
        preserve_numbers=True,
    )

    chunks = preprocessor.run_pipeline(
        docs_path=str(DOCS_PATH),
        refs_path=str(REFS_PATH),
        output_path=str(OUTPUT_PATH),
    )

    # Tampilkan statistik
    print_chunk_statistics(chunks)

    # Tampilkan contoh 3 chunk pertama
    print("\nCONTOH 3 CHUNK PERTAMA:")
    print("-" * 50)
    for chunk in chunks[:3]:
        print(f"\nID      : {chunk.chunk_id}")
        print(f"Seksi   : {chunk.section}")
        print(f"Token   : {chunk.token_count}")
        print(f"Keywords: {chunk.keywords}")
        print(f"Teks    :\n{chunk.text[:300]}...")
        print("-" * 50)