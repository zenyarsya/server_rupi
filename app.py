from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing import List
import difflib
import re

app = FastAPI()

# Mengaktifkan CORS agar Construct 3 dari browser tidak diblokir
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memuat model SBERT (Multilingual agar paham Bahasa Indonesia)
print("Sedang membangunkan otak AI Rupi... (Tunggu sebentar ya)")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Otak AI siap menerima jawaban!")

# Format struktur data (JSON) yang akan diterima dari Construct 3
class DataJawaban(BaseModel):
    jawaban_siswa: str
    kunci_jawaban: str
    kata_ajaib: List[str]  # Array untuk fitur Fuzzy Matching

# =====================================================================
# PIPELINE PRE-PROCESSING TEKS (Sesuai Rancangan Skripsi / Slide Bab 3)
# =====================================================================
def preprocess_text(text: str):
    # Tahap 1: Case Folding (Uang RUPIAH -> uang rupiah)
    text = text.lower()
    
    # Tahap 2: Tokenization (Memecah teks menjadi array kata)
    tokens = text.split()
    
    # Tahap 3: Stopword Removal (Custom)
    # HANYA membuang kata basa-basi/filler lisan agar struktur makna SBERT tetap utuh
    custom_stopwords = {"hmm", "eh", "anu", "sepertinya", "kayaknya", "sih", "dong", "ya", "deh", "loh", "tuh", "gitu"}
    tokens_filtered = [kata for kata in tokens if kata not in custom_stopwords]
    
    # Menggabungkan kembali token yang sudah disaring
    text_joined = " ".join(tokens_filtered)
    
    # Tahap 4: Normalisasi (Menghapus tanda baca seperti uang... rupiah!! -> uang rupiah)
    text_normalized = re.sub(r'[^\w\s]', '', text_joined)
    
    return text_normalized.strip()

# =====================================================================
# ENDPOINT UTAMA PENILAIAN AI
# =====================================================================
@app.post("/similarity")
def hitung_similarity(data: DataJawaban):
    # 1. Jalankan Pipeline Pre-processing pada jawaban dan kunci
    jawaban_bersih = preprocess_text(data.jawaban_siswa)
    kunci_bersih = preprocess_text(data.kunci_jawaban)

    # 2. Proses SBERT Embedding (Mengubah teks menjadi vektor ruang)
    embedding_siswa = model.encode(jawaban_bersih, convert_to_tensor=True)
    embedding_kunci = model.encode(kunci_bersih, convert_to_tensor=True)

    # 3. Kalkulasi Cosine Similarity
    kemiripan = float(util.cos_sim(embedding_siswa, embedding_kunci)[0][0])

    # 4. Fuzzy Matching (Penangkap Typo Berdasarkan 'Kata Ajaib')
    # Pecah kembali jawaban bersih untuk mengecek kata per kata
    kata_siswa_list = jawaban_bersih.split()
    ada_typo_atau_kata_pas = False
    
    for kata in kata_siswa_list:
        mirip = difflib.get_close_matches(kata, data.kata_ajaib, n=1, cutoff=0.8)
        if mirip:
            ada_typo_atau_kata_pas = True
            break

    # 5. Evaluasi Threshold & Adaptive Feedback 
    # (Sesuai rentang nilai pada gambar presentasi: >= 0.75, 0.50 - 0.74, < 0.50)
    if kemiripan >= 0.75 or ada_typo_atau_kata_pas == True:
        status_jawaban = "benar"
    elif kemiripan >= 0.50:
        status_jawaban = "parsial"
    else:
        status_jawaban = "belum tepat"

    # Print log ke terminal untuk memantau mesin saat sidang/demo
    print(f"[{status_jawaban.upper()}] Skor: {kemiripan:.2f} | Typo Terselamatkan: {ada_typo_atau_kata_pas} | Teks: '{jawaban_bersih}'")

    # 6. Kirim Response JSON kembali ke Game Construct 3
    return {
        "skor": round(kemiripan, 2),
        "status": status_jawaban
    }