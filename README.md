# UAS-Penalaran-Komputer
Repositori ini merupakan bagian dari tugas akhir UAS mata kuliah **Penalaran Komputer**. Proyek ini membangun sistem **Case-Based Reasoning (CBR)** untuk membantu retrieval dan prediksi solusi dari kasus hukum berdasarkan teks putusan.

Sistem dibangun menggunakan dua pendekatan representasi vektor:  
- **TF-IDF** (Text-based retrieval)  
- **BERT** (Semantic-based retrieval dengan model `indobenchmark/indobert-base-p1`)

---

## Struktur Direktori

```
/data/                      # Dataset putusan (raw, processed, queries, predictions)
/notebooks/                 # (Opsional) Notebook tambahan per tahap (tidak wajib)
/03_retrieval.py            # Proses vektorisasi & retrieval kasus serupa
/04_predict.py              # Prediksi solusi berdasarkan retrieval
/05_evaluation.py           # Evaluasi hasil retrieval & prediksi
README.md                   # File ini
requirements.txt            # Daftar library yang dibutuhkan
```

---

## Cara Menjalankan (End-to-End)

### 1. Clone repositori
```bash
git clone https://github.com/username/UAS-Penalaran-Komputer.git
cd UAS-Penalaran-Komputer
```

### 2. Install dependency
```bash
pip install -r requirements.txt
```

### 3. Jalankan script tahap demi tahap:
```bash
python 03_retrieval.py     # Membuat vektor dan file queries
python 04_predict.py       # Prediksi solusi dari top-k kasus
python 05_evaluation.py    # Evaluasi retrieval & prediksi
```

> **Catatan**: Pastikan data telah tersedia di folder `/data/` dan file `_clean.txt` sudah ada di `/TXT`.

---

## Deskripsi Tahapan

| Tahap | Nama File | Deskripsi |
|------|-----------|-----------|
| 1 | `03_retrieval.py` | Pembersihan teks, representasi dengan TF-IDF & BERT, pemisahan data, retrieval |
| 2 | `04_predict.py` | Prediksi isi "amar putusan" dari kasus serupa menggunakan voting atau weighted |
| 3 | `05_evaluation.py` | Evaluasi performa retrieval dan prediksi menggunakan metrik Accuracy, Precision, Recall, F1-score, Cosine Similarity |

---

## Requirements

```txt
transformers
torch
scikit-learn
pandas
numpy
tqdm
matplotlib
```

> Untuk BERT, kamu perlu koneksi internet untuk `from_pretrained()` pertama kali.

---

## Lisensi & Atribusi

Proyek ini disusun untuk keperluan akademik.  
Model BERT yang digunakan: [indobenchmark/indobert-base-p1](https://huggingface.co/indobenchmark/indobert-base-p1)
