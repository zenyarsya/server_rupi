# Server RUPI (Backend NLP & Text Similarity)

## Deskripsi Singkat
Repositori ini berisi *source code* *backend* untuk sistem pemrosesan teks menggunakan **FastAPI** dan **Sentence Transformers**. Server ini dirancang untuk menghitung tingkat kemiripan teks (*text similarity*) dan mendeteksi/memperbaiki kesalahan ketik (*typo correction*). 

Sistem ini telah dikonfigurasi dengan CORS (Cross-Origin Resource Sharing) agar dapat terintegrasi dan berkomunikasi secara langsung dengan *engine* **Construct 3** melalui *browser*.

## Teknologi & Library Utama
* **Python 3.x**
* **FastAPI**: *Framework* untuk membangun API berkinerja tinggi.
* **Sentence-Transformers**: Model *Machine Learning* untuk pemrosesan bahasa alami (NLP) dan ekstraksi fitur teks.
* **Uvicorn**: Server ASGI untuk menjalankan aplikasi web FastAPI.
* **Pydantic**: Untuk validasi data yang dikirim melalui API.

## Struktur File
* `app.py`: Merupakan file utama (*entry point*) yang memuat konfigurasi *routing* API, *middleware* CORS, dan logika utama perhitungan *similarity*.

## Cara Menjalankan Server secara Lokal
1. Pastikan Python sudah terinstal di sistem Anda.
2. Instal seluruh *library* pendukung dengan menjalankan perintah berikut di terminal:
   ```bash
   pip install fastapi uvicorn sentence-transformers pydantic

## Jalankan server menggunakan Uvicorn: 
uvicorn app:app --reload

Server akan berjalan secara lokal (biasanya pada http://127.0.0.1:8000). Anda bisa mengakses dokumentasi API interaktif pada http://127.0.0.1:8000/docs.
