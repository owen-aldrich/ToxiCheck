import streamlit as st
import joblib
import pandas as pd
import re
import nltk
import os
from collections import Counter
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(page_title="ToxiCheck", page_icon="🕵️‍♂️", layout="wide")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Grid panduan kategori */
.desc-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
    align-items: stretch;
    margin-bottom: 8px;
}

/* Deskripsi kategori */
.desc-card {
    border-radius: 12px;
    padding: 18px 20px;
    box-sizing: border-box;
}
.desc-hate   { background-color: #fde8e8; border-left: 5px solid #e53e3e; }
.desc-abusive{ background-color: #fef3e2; border-left: 5px solid #dd6b20; }
.desc-neutral{ background-color: #e8f5e9; border-left: 5px solid #38a169; }
.desc-card h4 { margin: 0 0 8px 0; font-size: 1rem; }
.desc-card p  { margin: 0; font-size: 0.88rem; color: #444; line-height: 1.5; }

/* Kartu hasil model */
.result-card {
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid #e2e8f0;
    background: #f8fafc;
    margin-bottom: 6px;
}
.result-card .model-name { font-size: 0.8rem; color: #718096; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }
.result-card .result-label { font-size: 1.15rem; font-weight: 700; }
.label-hate    { color: #e53e3e; }
.label-abusive { color: #dd6b20; }
.label-neutral { color: #38a169; }

/* Tiebreaker warning banner */
.tie-banner {
    border-radius: 14px;
    padding: 18px 24px;
    text-align: center;
    margin-bottom: 16px;
    background: linear-gradient(135deg, #fefcbf 0%, #f6e05e 100%);
    border: 2px solid #d69e2e;
    color: #744210;
}
.tie-banner .tie-title { font-size: 1.1rem; font-weight: 800; margin-bottom: 6px; }
.tie-banner .tie-body  { font-size: 0.88rem; line-height: 1.6; margin: 0; }
.mv-banner {
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
    margin-bottom: 20px;
}
.mv-hate    { background: linear-gradient(135deg, #fc8181 0%, #e53e3e 100%); color: white; }
.mv-abusive { background: linear-gradient(135deg, #f6ad55 0%, #dd6b20 100%); color: white; }
.mv-neutral { background: linear-gradient(135deg, #68d391 0%, #38a169 100%); color: white; }
.mv-banner .mv-title { font-size: 0.9rem; font-weight: 600; opacity: 0.9; margin-bottom: 4px; }
.mv-banner .mv-label { font-size: 2rem; font-weight: 800; margin: 0; }
.mv-banner .mv-note  { font-size: 0.82rem; opacity: 0.85; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Load model & NLP ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    count_vec    = joblib.load(os.path.join(current_dir, 'models', 'count_vec.pkl'))
    tfidf_vec    = joblib.load(os.path.join(current_dir, 'models', 'tfidf_vec.pkl'))
    nb_model     = joblib.load(os.path.join(current_dir, 'models', 'nb_model.pkl'))
    knn_model    = joblib.load(os.path.join(current_dir, 'models', 'knn_model.pkl'))
    logreg_model = joblib.load(os.path.join(current_dir, 'models', 'logreg_model.pkl'))
    svm_model    = joblib.load(os.path.join(current_dir, 'models', 'svm_model.pkl'))

    if not hasattr(svm_model, '_effective_probability'):
        svm_model._effective_probability = False

    return count_vec, tfidf_vec, nb_model, knn_model, logreg_model, svm_model


@st.cache_data
def load_nlp_components():
    nltk.download('stopwords', quiet=True)
    id_stopwords = set(stopwords.words('indonesian'))
    id_stemmer   = StemmerFactory().create_stemmer()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    slang_data  = pd.read_csv(os.path.join(current_dir, 'dataset', 'new_kamusalay.csv'),
                              encoding='latin-1', header=None)
    slang_dict  = pd.Series(slang_data[1].values, index=slang_data[0]).to_dict()

    return id_stopwords, id_stemmer, slang_dict


count_vec, tfidf_vec, nb_model, knn_model, logreg_model, svm_model = load_models()
id_stopwords, id_stemmer, slang_dict = load_nlp_components()


# ── Text preprocessor ─────────────────────────────────────────────────────────
def text_cleaner(sentence):
    text = str(sentence).lower()
    text = re.sub(r'\\x[a-f0-9]{2}', ' ', text)
    text = re.sub(r'\b(rt|user|url)\b', ' ', text)
    text = re.sub(r'[^a-z]+', ' ', text).strip()

    words = text.split()
    final_tokens = []
    for word in words:
        formal_word = slang_dict.get(word, word)
        if formal_word not in id_stopwords:
            stemmed_word = id_stemmer.stem(formal_word)
            if stemmed_word:
                final_tokens.append(stemmed_word)

    return ' '.join(final_tokens)


# ── Helper: render label dengan CSS class ─────────────────────────────────────
def label_class(prediction):
    if prediction == "Hate Speech":
        return "label-hate", "⚠️ Hate Speech"
    elif prediction == "Abusive":
        return "label-abusive", "🤬 Abusive"
    else:
        return "label-neutral", "✅ Neutral"


def mv_class(prediction):
    if prediction == "Hate Speech":
        return "mv-hate"
    elif prediction == "Abusive":
        return "mv-abusive"
    else:
        return "mv-neutral"


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("ToxiCheck 🕵️‍♂️")
st.markdown("""Aplikasi ini membandingkan prediksi dari **4 algoritma Machine Learning**:
Naive Bayes, K-Nearest Neighbors, Logistic Regression, dan Support Vector Machine (SVM).""")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INPUT
# ═══════════════════════════════════════════════════════════════════════════════
user_input = st.text_area("Masukkan teks atau kalimat (tweet) yang ingin dianalisis:", height=150)
detect_btn = st.button("🔍 Deteksi Teks", type="primary", use_container_width=False)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — KOTAK DESKRIPSI KATEGORI (selalu tampil)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📖 Panduan Kategori")

st.markdown("""
<div class="desc-grid">
    <div class="desc-card desc-hate">
        <h4>⚠️ Hate Speech</h4>
        <p>Teks yang mengandung <strong>ujaran kebencian</strong> dan secara eksplisit menargetkan
        individu atau kelompok tertentu berdasarkan ras, agama, etnis, gender, atau atribut
        identitas lainnya. Konten ini berpotensi memicu diskriminasi atau kekerasan.</p>
    </div>
    <div class="desc-card desc-abusive">
        <h4>🤬 Abusive</h4>
        <p>Teks yang mengandung <strong>kata-kata kasar, makian, atau bahasa tidak sopan</strong>,
        namun tidak secara spesifik menargetkan kelompok tertentu. Biasanya berupa ekspresi
        emosional negatif atau penghinaan yang bersifat umum.</p>
    </div>
    <div class="desc-card desc-neutral">
        <h4>✅ Neutral</h4>
        <p>Teks yang <strong>tidak mengandung unsur kebencian maupun bahasa kasar</strong>.
        Konten bersifat normal dan tidak mengarah pada diskriminasi, kekerasan,
        atau penghinaan terhadap siapapun.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HASIL PREDIKSI (muncul setelah klik tombol)
# ═══════════════════════════════════════════════════════════════════════════════
if detect_btn:
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Sedang memproses dan membersihkan teks..."):
            cleaned_input = text_cleaner(user_input)

            input_count = count_vec.transform([cleaned_input])
            input_tfidf = tfidf_vec.transform([cleaned_input])

            pred_nb     = nb_model.predict(input_count)[0]
            pred_knn    = knn_model.predict(input_tfidf)[0]
            pred_logreg = logreg_model.predict(input_tfidf)[0]
            pred_svm    = svm_model.predict(input_tfidf)[0]

            # Majority vote — dengan tiebreaker SVM jika seri 2-2
            all_preds   = [pred_nb, pred_knn, pred_logreg, pred_svm]
            vote_counts = Counter(all_preds)
            top_count   = vote_counts.most_common(1)[0][0]
            is_tie      = len(vote_counts) >= 2 and list(vote_counts.values()).count(vote_counts.most_common(1)[0][1]) > 1

            if is_tie:
                majority = pred_svm  # tiebreaker: ikut SVM (model terbaik)
            else:
                majority = vote_counts.most_common(1)[0][0]
            top_count = vote_counts[majority]

        st.success("✅ Analisis selesai!")
        st.write(f"**Teks yang sudah dibersihkan:** `{cleaned_input}`")

        st.markdown("---")

        # ── Majority Vote Banner ──────────────────────────────────────────────
        st.markdown("### 🗳️ Kesimpulan — Majority Vote")

        mv_css        = mv_class(majority)
        _, mv_display = label_class(majority)

        if is_tie:
            # Tampilkan warning seri dulu, lalu banner hasil SVM
            tied_labels = [lbl for lbl, cnt in vote_counts.items()
                           if cnt == vote_counts.most_common(1)[0][1]]
            tied_str = " vs ".join(tied_labels)
            st.markdown(f"""
            <div class="tie-banner">
                <div class="tie-title">⚠️ Hasil Seri 2-2!</div>
                <p class="tie-body">
                    Model tidak sepakat — terjadi seri antara <strong>{tied_str}</strong>.<br>
                    Menggunakan prediksi <strong>SVM (Linear)</strong> sebagai tiebreaker
                    karena memiliki akurasi tertinggi di antara keempat model.
                </p>
            </div>
            """, unsafe_allow_html=True)
            agree_str = f"Tiebreaker: SVM → {mv_display}"
        else:
            agree_str = f"{top_count} dari 4 model sepakat"

        st.markdown(f"""
        <div class="mv-banner {mv_css}">
            <div class="mv-title">HASIL KEPUTUSAN MAYORITAS</div>
            <div class="mv-label">{mv_display}</div>
            <div class="mv-note">{agree_str}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Hasil 4 Model ─────────────────────────────────────────────────────
        st.markdown("### 🤖 Prediksi per Model")

        models_info = [
            ("Naive Bayes",         pred_nb),
            ("K-NN",                pred_knn),
            ("Logistic Regression", pred_logreg),
            ("SVM (Linear)",        pred_svm),
        ]

        cols = st.columns(4)
        for col, (model_name, pred) in zip(cols, models_info):
            css_class, display_label = label_class(pred)
            with col:
                st.markdown(f"""
                <div class="result-card">
                    <div class="model-name">{model_name}</div>
                    <div class="result-label {css_class}">{display_label}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Rincian vote ──────────────────────────────────────────────────────
        with st.expander("📊 Lihat rincian voting"):
            for label, count in vote_counts.most_common():
                bar = "█" * count + "░" * (4 - count)
                st.markdown(f"**{label}** &nbsp; `{bar}` &nbsp; {count}/4 model")
