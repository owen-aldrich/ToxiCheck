---
title: ToxiCheck
emoji: 🕵️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: 3.10
app_file: app.py
tags:
- gradio
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
- ⚠️ **Tiebreaker otomatis** — jika seri 2-2, prediksi SVM digunakan sebagai penentu karena memperoleh CV Macro F1 tertinggi
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
├── app.py                    # Aplikasi web Gradio
├── requirements-local.txt    # Dependensi tambahan untuk penggunaan lokal
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

Hyperparameter setiap model dipilih menggunakan `GridSearchCV` dengan
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Proses tuning hanya
menggunakan data train dan memilih kandidat berdasarkan **Macro F1**. Test set
20% tetap dipisahkan untuk evaluasi akhir.

| Model               | Vectorizer | CV Macro F1 | Accuracy   | Macro Precision | HS Recall  | Test Macro F1 |
|---------------------|------------|-------------|------------|-----------------|------------|---------------|
| Naive Bayes         | CountVec   | 0.7483      | 0.7809     | 0.7494          | 0.8129     | 0.7548        |
| K-NN                | TF-IDF     | 0.6967      | 0.7369     | 0.7223          | 0.8085     | 0.7060        |
| **Logistic Regression** | **TF-IDF** | 0.7845  | **0.8193** | **0.7998**      | **0.8273** | **0.7909**    |
| SVM (Linear)        | TF-IDF     | **0.7864**  | 0.8090     | 0.7765          | 0.8192     | 0.7787        |

Hyperparameter terbaik:

| Model               | Parameter terbaik                       |
|---------------------|------------------------------------------|
| Naive Bayes         | `alpha=0.1`                              |
| K-NN                | `metric='cosine'`, `n_neighbors=11`      |
| Logistic Regression | `C=10.0`, `class_weight=None`            |
| SVM (Linear)        | `C=2.0`, `class_weight='balanced'`       |

Logistic Regression memberikan hasil terbaik pada test set, dengan accuracy
0.8193 dan Macro F1 0.7909. SVM memperoleh CV Macro F1 tertinggi (0.7864) dan
tetap digunakan sebagai tiebreaker ketika voting seri 2-2.

---

## Cara Menjalankan

### Opsi 1 — Hugging Face Spaces

```bash
# Push langsung ke remote Hugging Face Space
git push space main
```

Space dikonfigurasi menggunakan Gradio SDK sehingga tidak memerlukan Docker.

### Opsi 2 — Lokal (Manual)

```bash
# 1. Install dependensi
pip install -r requirements-local.txt

# 2. Jalankan aplikasi Gradio
python app.py
```

### Opsi 3 — Jalankan Notebook Training

Untuk melatih ulang model dari awal:

```bash
jupyter notebook nlp_aol.ipynb
```

Jalankan semua cell secara berurutan. Proses stemming Sastrawi memakan beberapa menit. Model hasil training tersimpan otomatis ke folder `models/`.

---

## Tech Stack

Python · Gradio · scikit-learn · NLTK · PySastrawi · pandas · NumPy · joblib

---

## Catatan

- Dataset mengandung konten kasar dan ujaran kebencian — hanya digunakan untuk keperluan akademik.
- Model dilatih pada tweet berbahasa Indonesia, performa bisa menurun pada domain atau gaya bahasa lain.
