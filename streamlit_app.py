import streamlit as st
import joblib
import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(page_title="Deteksi Hate Speech", page_icon="🕵️‍♂️", layout="wide")

@st.cache_resource
def load_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    count_vec = joblib.load(os.path.join(current_dir, 'count_vec.pkl'))
    tfidf_vec = joblib.load(os.path.join(current_dir, 'tfidf_vec.pkl'))
    nb_model = joblib.load(os.path.join(current_dir, 'nb_model.pkl'))
    knn_model = joblib.load(os.path.join(current_dir, 'knn_model.pkl'))
    logreg_model = joblib.load(os.path.join(current_dir, 'logreg_model.pkl'))
    svm_model = joblib.load(os.path.join(current_dir, 'svm_model.pkl'))
    
    # Perbaikan untuk error AttributeError di Hugging Face
    if not hasattr(svm_model, '_effective_probability'):
        svm_model._effective_probability = False
        
    return count_vec, tfidf_vec, nb_model, knn_model, logreg_model, svm_model

@st.cache_data
def load_nlp_components():
    nltk.download('stopwords', quiet=True)
    id_stopwords = set(stopwords.words('indonesian'))
    id_stemmer = StemmerFactory().create_stemmer()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    slang_data = pd.read_csv(os.path.join(current_dir, 'new_kamusalay.csv'), encoding='latin-1', header=None)
    slang_dict = pd.Series(slang_data[1].values, index=slang_data[0]).to_dict()
    
    return id_stopwords, id_stemmer, slang_dict

# Load semua model dan komponen NLP
count_vec, tfidf_vec, nb_model, knn_model, logreg_model, svm_model = load_models()
id_stopwords, id_stemmer, slang_dict = load_nlp_components()

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

st.title("🕵️‍♂️ Hate Speech & Abusive Text Detector")
st.markdown("""Aplikasi ini membandingkan prediksi dari 4 algoritma *Machine Learning*: **Naive Bayes**, **K-Nearest Neighbors (K-NN)**, **Logistic Regression**, dan **Support Vector Machine (SVM)**.""")

user_input = st.text_area("Masukkan teks atau kalimat (tweet) yang ingin dianalisis:", height=150)

if st.button("Deteksi Teks", type="primary"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Sedang memproses dan membersihkan teks..."):
            cleaned_input = text_cleaner(user_input)
            
            input_count = count_vec.transform([cleaned_input])
            input_tfidf = tfidf_vec.transform([cleaned_input])
            
            pred_nb = nb_model.predict(input_count)[0]
            pred_knn = knn_model.predict(input_tfidf)[0]
            pred_logreg = logreg_model.predict(input_tfidf)[0]
            pred_svm = svm_model.predict(input_tfidf)[0]
            
            st.success("Analisis Selesai!")
            
        st.write(f"**Teks yang sudah dibersihkan:** `{cleaned_input}`")
        st.markdown("### Hasil Prediksi 4 Model")
        
        col1, col2, col3, col4 = st.columns(4)
        
        def format_result(prediction):
            if prediction == "Hate Speech":
                return "🔴 Hate Speech"
            elif prediction == "Abusive":
                return "🟠 Abusive"
            else:
                return "🟢 Neutral"
                
        with col1:
            st.info("Naive Bayes")
            st.subheader(format_result(pred_nb))
            
        with col2:
            st.info("K-NN")
            st.subheader(format_result(pred_knn))
            
        with col3:
            st.info("Log Reg")
            st.subheader(format_result(pred_logreg))
            
        with col4:
            st.info("SVM (Linear)")
            st.subheader(format_result(pred_svm))