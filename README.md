# UAS-Penalaran-Komputer
Repositori ini merupakan bagian dari tugas akhir UAS mata kuliah **Penalaran Komputer**. Proyek ini membangun sistem **Case-Based Reasoning (CBR)** untuk membantu retrieval dan prediksi solusi dari kasus hukum berdasarkan teks putusan.

Sistem dibangun menggunakan dua pendekatan representasi vektor:  
- **TF-IDF** (Text-based retrieval)  
- **BERT** (Semantic-based retrieval dengan model `indobenchmark/indobert-base-p1`)

---

## 🔧 Struktur Direktori

```
/data/                      # Dataset putusan (raw, processed, queries, predictions)
/notebooks/                 # (Opsional) Notebook tambahan per tahap (tidak wajib)
/03_retrieval.py            # Proses vektorisasi & retrieval kasus serupa
/04_predict.py              # Prediksi solusi berdasarkan retrieval
/05_evaluation.py           # Evaluasi hasil retrieval & prediksi
README.md                   # File ini
requirements.txt            # Daftar library yang dibutuhkan
```
