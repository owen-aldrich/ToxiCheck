---
title: ToxiCheck
emoji: 🕵️
colorFrom: red
colorTo: orange
sdk: docker
app_port: 8501
tags:
- streamlit
- nlp
- hate-speech
- indonesian
pinned: false
---

# ToxiCheck 🕵️‍♂️
### Indonesian Hate Speech & Abusive Language Detector

Aplikasi web untuk mendeteksi dan mengklasifikasikan tweet berbahasa Indonesia ke dalam tiga kategori: **Hate Speech**, **Abusive**, dan **Neutral** — menggunakan empat algoritma machine learning klasik dengan perbandingan hasil secara langsung.

Proyek ini dibuat untuk tugas NLP (Application of Learning / AoL).

---

## Fitur Aplikasi

- 🔍 **Input teks bebas** — masukkan tweet atau kalimat apapun untuk dianalisis
- 🤖 **4 model sekaligus** — Naive Bayes, K-NN, Logistic Regression, SVM (Linear)
- 🗳️ **Majority Vote** — kesimpulan akhir berdasarkan suara terbanyak dari keempat model
- ⚠️ **Tiebreaker otomatis** — jika seri 2-2, prediksi SVM digunakan sebagai penentu (akurasi tertinggi), disertai notifikasi ke pengguna
- 📖 **Panduan Kategori** — kotak deskripsi permanen yang menjelaskan arti tiap label
- 📊 **Rincian voting** — distribusi suara tiap model ditampilkan secara visual

---

## Struktur Proyek

```
ToxiCheck/
├── dataset/
│   ├── data.csv              # Dataset tweet dengan label HS & Abusive
│   ├── new_kamusalay.csv     # Kamus normalisasi kata slang/alay
│   ├── abusive.csv           # Leksikon kata-kata kasar
│   └── README.md             # Dokumentasi sumber dataset
├── models/
│   ├── count_vec.pkl         # CountVectorizer (untuk Naive Bayes)
│   ├── tfidf_vec.pkl         # TfidfVectorizer (untuk KNN/LogReg/SVM)
│   ├── nb_model.pkl          # Naive Bayes
│   ├── knn_model.pkl         # K-Nearest Neighbors
│   ├── logreg_model.pkl      # Logistic Regression
│   └── svm_model.pkl         # SVM Linear
├── nlp_aol.ipynb             # Notebook training & evaluasi model
├── streamlit_app.py          # Aplikasi web Streamlit
├── Dockerfile                # Konfigurasi container Docker
└── requirements.txt          # Dependensi Python
```

---

## Dataset

Bersumber dari paper akademik: **"Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter"** — Ibrohim & Budi, ACL Workshop ALW3, 2019. Lisensi: CC BY-NC-SA 4.0.

| Kategori      | Jumlah     | Aturan Pelabelan           |
|---------------|------------|----------------------------|
| Neutral       | 5.860      | `HS = 0` dan `Abusive = 0` |
| Hate Speech   | 5.561      | `HS = 1`                   |
| Abusive       | 1.748      | `Abusive = 1` dan `HS = 0` |
| **Total**     | **13.169** |                            |

Split data: **80:20** stratified (`random_state=42`) → 10.535 train / 2.634 test.

---

## Pipeline NLP

Fungsi `text_cleaner()` diterapkan pada setiap teks sebelum prediksi:

1. **Lowercase** — semua huruf dijadikan kecil
2. **Hapus karakter escape** — menghapus representasi hex emoji (`\xNN`)
3. **Hapus token Twitter** — `rt`, `user`, `url`
4. **Hapus non-alfabet** — hanya menyisakan huruf a–z
5. **Normalisasi slang** — lookup di `new_kamusalay.csv` (contoh: `loe → kamu`)
6. **Stopword removal** — NLTK Indonesian stopwords
7. **Stemming** — PySastrawi Indonesian stemmer

---

## Model & Hasil Evaluasi

| Model               | Vectorizer  | Accuracy   | F1 Macro   |
|---------------------|-------------|------------|------------|
| Naive Bayes         | CountVec    | 0.7912     | 0.7451     |
| K-NN                | TF-IDF      | 0.7024     | 0.6730     |
| Logistic Regression | TF-IDF      | 0.8151     | 0.7825     |
| **SVM (Linear)**    | **TF-IDF**  | **0.8166** | **0.7865** |

SVM (Linear) adalah model terbaik secara keseluruhan dan digunakan sebagai tiebreaker saat terjadi seri 2-2.

---

## Cara Menjalankan

### Opsi 1 — Docker (Direkomendasikan)

Tidak perlu install Python atau dependensi apapun.

```bash
# 1. Clone repo
git clone <repo-url>
cd ToxiCheck

# 2. Build image
docker build -t toxicheck .

# 3. Jalankan container
docker run -p 8501:8501 toxicheck

# 4. Buka browser
# → http://localhost:8501
```

### Opsi 2 — Lokal (Manual)

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Jalankan aplikasi
streamlit run streamlit_app.py
```

### Opsi 3 — Jalankan Notebook Training

Untuk melatih ulang model dari awal:

```bash
jupyter notebook nlp_aol.ipynb
```

Jalankan semua cell secara berurutan. Proses stemming Sastrawi memakan beberapa menit. Model hasil training tersimpan otomatis ke folder `models/`.

---

## Tech Stack

Python · Streamlit · scikit-learn · NLTK · PySastrawi · pandas · NumPy · joblib · Docker

---

## Catatan

- Dataset mengandung konten kasar dan ujaran kebencian — hanya digunakan untuk keperluan akademik.
- Model dilatih pada tweet berbahasa Indonesia, performa bisa menurun pada domain atau gaya bahasa lain.
