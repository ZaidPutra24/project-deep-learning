"""
cleaning.py
===========
Modul Pembersihan Teks Lanjutan untuk Pipeline RAG Stunting

Berisi fungsi-fungsi cleaning yang dapat digunakan secara modular
di luar kelas StuntingPreprocessor.

Proyek  : Chatbot Konsultasi Risiko Stunting Kota Kendari
"""

import re
import unicodedata
import logging
from typing import List

logger = logging.getLogger("stunting.cleaning")


def remove_html_tags(text: str) -> str:
    """Hapus tag HTML jika ada dalam teks."""
    return re.sub(r"<[^>]+>", "", text)


def remove_urls(text: str) -> str:
    """Hapus URL dari teks."""
    return re.sub(r"https?://\S+|www\.\S+", "[URL]", text)


def remove_emails(text: str) -> str:
    """Hapus alamat email."""
    return re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)


def normalize_whitespace(text: str) -> str:
    """Normalisasi spasi berlebih dan newline."""
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
    """
    Hapus karakter khusus yang tidak diperlukan.
    
    Args:
        text            : Input teks
        keep_punctuation: Pertahankan tanda baca standar jika True
    """
    if keep_punctuation:
        # Hanya hapus karakter non-ASCII yang bukan tanda baca standar
        text = re.sub(r"[^\w\s\.,;:!?()\-\'\"/\n%°]", "", text)
    else:
        text = re.sub(r"[^\w\s]", "", text)
    return text


def clean_number_formatting(text: str) -> str:
    """
    Standardisasi format angka dalam teks medis.
    Contoh: 21,6% → 21.6%, 2.500 gram → 2500 gram
    """
    # Angka dengan titik ribuan Indonesia (2.500) → hapus titik pemisah ribuan
    text = re.sub(r"(\d)\.(\d{3})(?!\d)", r"\1\2", text)
    # Koma desimal Indonesia → titik desimal
    # Hati-hati: hanya angka,angka bukan kata,kata
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    return text


def clean_medical_text(text: str) -> str:
    """
    Pipeline cleaning khusus teks medis dan gizi Indonesia.
    Menggabungkan semua fungsi cleaning di atas.
    
    Args:
        text: Teks medis mentah
        
    Returns:
        Teks yang sudah dibersihkan
    """
    text = unicodedata.normalize("NFC", text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = clean_number_formatting(text)
    text = normalize_whitespace(text)
    logger.debug("clean_medical_text selesai")
    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Memisahkan teks menjadi kalimat-kalimat.
    Memperhatikan singkatan umum bahasa Indonesia.
    
    Args:
        text: Paragraf teks
        
    Returns:
        List kalimat
    """
    # Singkatan yang tidak mengakhiri kalimat
    abbreviations = r"(?<!\bdr)(?<!\bProf)(?<!\bdkk)(?<!\bhlm)(?<!\bvol)"
    sentence_pattern = re.compile(
        abbreviations + r"(?<=[.!?])\s+(?=[A-Z])"
    )
    sentences = sentence_pattern.split(text)
    return [s.strip() for s in sentences if s.strip()]
