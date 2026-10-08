# Indonesian Hate Speech & Abusive Language Detection

Klasifikasi tweet berbahasa Indonesia ke dalam tiga kategori (**Neutral**, **Abusive**, **Hate Speech**) menggunakan empat algoritma machine learning klasik dan representasi teks TF-IDF / Bag-of-Words.

Proyek ini dibuat untuk tugas NLP (Application of Learning / AoL).

## Fitur Utama

- Pipeline preprocessing teks Indonesia: case folding, pembersihan entitas Twitter, normalisasi kata slang, penghapusan stopword, dan stemming (Sastrawi)
- Perbandingan 4 model: Naive Bayes, K-NN, Logistic Regression, dan SVM (Linear)
- Evaluasi: accuracy, precision, recall, F1-score, dan confusion matrix
- Model dan vectorizer disimpan dengan `joblib` untuk dipakai ulang
- Mode interaktif untuk menguji kalimat sendiri pada semua model sekaligus

## Struktur Proyek

```
.
├── dataset/
│   ├── data.csv              # Dataset tweet beserta label HS & Abusive
│   └── new_kamusalay.csv     # Kamus normalisasi kata slang (alay)
├── models/                   # Dihasilkan setelah notebook dijalankan
│   ├── count_vec.pkl
│   ├── tfidf_vec.pkl
│   ├── nb_model.pkl
│   ├── knn_model.pkl
│   ├── logreg_model.pkl
│   └── svm_model.pkl
├── nlp_aol.ipynb             # Notebook utama
└── README.md
```

## Dataset

| Kategori    | Jumlah | Aturan Pelabelan                  |
|-------------|--------|-----------------------------------|
| Neutral     | 5.860  | `HS = 0` dan `Abusive = 0`        |
| Hate Speech | 5.561  | `HS = 1`                          |
| Abusive     | 1.748  | `Abusive = 1` dan `HS = 0`        |
| **Total**   | **13.169** |                               |

Data dibagi **80:20** (stratified, `random_state=42`): 10.535 data latih dan 2.634 data uji.

## Metodologi

1. **Data loading & labeling**: kolom `HS` dan `Abusive` digabung menjadi satu kolom `Category`.
2. **Text preprocessing** (`text_cleaner`):
   - lowercase
   - hapus karakter escape (`\xNN`) serta token `rt`, `user`, `url`
   - hapus semua karakter non-alfabet
   - normalisasi slang memakai `new_kamusalay.csv`
   - hapus stopword (NLTK Indonesian)
   - stemming memakai Sastrawi
3. **Feature extraction**:
   - `CountVectorizer` (unigram + bigram) untuk Naive Bayes
   - `TfidfVectorizer` (unigram + bigram, `sublinear_tf=True`) untuk K-NN, Logistic Regression, dan SVM
4. **Training & evaluasi** keempat model pada data uji.

### Konfigurasi Model

| Model               | Fitur  | Parameter                                       |
|---------------------|--------|-------------------------------------------------|
| Naive Bayes         | Count  | `MultinomialNB(alpha=1.0)`                      |
| K-NN                | TF-IDF | `n_neighbors=5`, `metric='cosine'`, `brute`     |
| Logistic Regression | TF-IDF | `C=1.0`, `max_iter=1000`                        |
| SVM (Linear)        | TF-IDF | `kernel='linear'`, `C=1.0`                      |

## Hasil

| Algoritma           | Accuracy | Precision (macro) | Recall (Hate Speech) | F1-Score (macro) |
|---------------------|----------|-------------------|----------------------|------------------|
| Naive Bayes         | 0.7912   | 0.8189            | 0.8723               | 0.7451           |
| K-NN                | 0.7024   | 0.6724            | 0.7545               | 0.6730           |
| Logistic Regression | 0.8151   | 0.8091            | 0.8354               | 0.7825           |
| **SVM (Linear)**    | **0.8166** | 0.7960          | 0.8372               | **0.7865**       |

- **SVM (Linear)** memiliki accuracy dan macro F1 tertinggi, disusul sangat dekat oleh Logistic Regression.
- **Naive Bayes** memiliki precision dan recall Hate Speech tertinggi, tetapi macro F1-nya lebih rendah.
- **K-NN** paling lemah pada semua metrik.

## Cara Menjalankan

### 1. Prasyarat

- Python 3.9+
- Jupyter Notebook / JupyterLab / VS Code

### 2. Install dependensi

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk PySastrawi joblib jupyter
```

### 3. Siapkan dataset

Letakkan `data.csv` dan `new_kamusalay.csv` di folder `dataset/`.

### 4. Jalankan notebook

```bash
jupyter notebook nlp_aol.ipynb
```

Jalankan semua cell secara berurutan. Proses stemming Sastrawi memakan waktu beberapa menit.

### 5. Coba prediksi interaktif

Cell terakhir membuka loop input. Ketik kalimat apa saja untuk melihat prediksi dari keempat model, dan ketik `exit` atau `quit` untuk berhenti.

### Memakai model yang sudah disimpan

```python
import joblib

tfidf_vec = joblib.load('models/tfidf_vec.pkl')
svm_model = joblib.load('models/svm_model.pkl')

# `text_cleaner` adalah fungsi preprocessing yang sama seperti di notebook
teks = text_cleaner("contoh kalimat yang ingin diuji")
print(svm_model.predict(tfidf_vec.transform([teks]))[0])
```

> Model Naive Bayes memakai `count_vec.pkl`, sedangkan K-NN, Logistic Regression, dan SVM memakai `tfidf_vec.pkl`.

## Tech Stack

Python, pandas, NumPy, scikit-learn, NLTK, PySastrawi, matplotlib, seaborn, joblib

## Catatan

- Dataset berisi konten kasar dan ujaran kebencian, dan hanya digunakan untuk keperluan akademik.
- Model dilatih pada tweet, sehingga performanya bisa turun pada gaya bahasa atau domain lain.
