import os
import re
from collections import Counter

import gradio as gr
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

try:
    import spaces
except ImportError:
    # Memungkinkan aplikasi tetap dijalankan secara lokal tanpa paket ZeroGPU.
    class _LocalSpaces:
        @staticmethod
        def GPU(*decorator_args, **decorator_kwargs):
            if decorator_args and callable(decorator_args[0]):
                return decorator_args[0]
            return lambda function: function

    spaces = _LocalSpaces()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_models():
    model_dir = os.path.join(BASE_DIR, "models")
    count_vec = joblib.load(os.path.join(model_dir, "count_vec.pkl"))
    tfidf_vec = joblib.load(os.path.join(model_dir, "tfidf_vec.pkl"))
    nb_model = joblib.load(os.path.join(model_dir, "nb_model.pkl"))
    knn_model = joblib.load(os.path.join(model_dir, "knn_model.pkl"))
    logreg_model = joblib.load(os.path.join(model_dir, "logreg_model.pkl"))
    svm_model = joblib.load(os.path.join(model_dir, "svm_model.pkl"))

    # Menjaga kompatibilitas model SVC pada beberapa versi scikit-learn.
    if not hasattr(svm_model, "_effective_probability"):
        svm_model._effective_probability = False

    return count_vec, tfidf_vec, nb_model, knn_model, logreg_model, svm_model


def load_nlp_components():
    try:
        id_stopwords = set(stopwords.words("indonesian"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        id_stopwords = set(stopwords.words("indonesian"))

    slang_path = os.path.join(BASE_DIR, "dataset", "new_kamusalay.csv")
    slang_data = pd.read_csv(slang_path, encoding="latin-1", header=None)
    slang_dict = pd.Series(slang_data[1].values, index=slang_data[0]).to_dict()
    id_stemmer = StemmerFactory().create_stemmer()
    return id_stopwords, id_stemmer, slang_dict


(
    count_vec,
    tfidf_vec,
    nb_model,
    knn_model,
    logreg_model,
    svm_model,
) = load_models()
id_stopwords, id_stemmer, slang_dict = load_nlp_components()


def text_cleaner(sentence):
    text = str(sentence).lower()
    text = re.sub(r"\\x[a-f0-9]{2}", " ", text)
    text = re.sub(r"\b(rt|user|url)\b", " ", text)
    text = re.sub(r"[^a-z]+", " ", text).strip()

    final_tokens = []
    for word in text.split():
        formal_word = slang_dict.get(word, word)
        if formal_word not in id_stopwords:
            stemmed_word = id_stemmer.stem(formal_word)
            if stemmed_word:
                final_tokens.append(stemmed_word)

    return " ".join(final_tokens)


def label_style(prediction):
    styles = {
        "Hate Speech": ("label-hate", "⚠️ Hate Speech"),
        "Abusive": ("label-abusive", "🤬 Abusive"),
        "Neutral": ("label-neutral", "✅ Neutral"),
    }
    return styles.get(prediction, ("label-neutral", str(prediction)))


def majority_style(prediction):
    return {
        "Hate Speech": "mv-hate",
        "Abusive": "mv-abusive",
        "Neutral": "mv-neutral",
    }.get(prediction, "mv-neutral")


@spaces.GPU(duration=10)
def predict_text(user_input):
    if not user_input or not user_input.strip():
        raise gr.Error("Silakan masukkan teks terlebih dahulu!")

    cleaned_input = text_cleaner(user_input)
    if not cleaned_input:
        raise gr.Error("Teks tidak memiliki kata yang dapat dianalisis setelah preprocessing.")

    input_count = count_vec.transform([cleaned_input])
    input_tfidf = tfidf_vec.transform([cleaned_input])

    predictions = {
        "Naive Bayes": nb_model.predict(input_count)[0],
        "K-NN": knn_model.predict(input_tfidf)[0],
        "Logistic Regression": logreg_model.predict(input_tfidf)[0],
        "SVM (Linear)": svm_model.predict(input_tfidf)[0],
    }

    vote_counts = Counter(predictions.values())
    max_votes = max(vote_counts.values())
    tied_labels = [label for label, count in vote_counts.items() if count == max_votes]
    is_tie = len(tied_labels) > 1
    majority = predictions["SVM (Linear)"] if is_tie else tied_labels[0]

    mv_css = majority_style(majority)
    _, mv_display = label_style(majority)
    if is_tie:
        tied_str = " vs ".join(tied_labels)
        tie_html = f"""
        <div class="tie-banner">
            <div class="tie-title">⚠️ Hasil Seri 2-2!</div>
            <p>Terjadi seri antara <strong>{tied_str}</strong>.<br>
            Prediksi <strong>SVM (Linear)</strong> digunakan sebagai tiebreaker
            karena memperoleh CV Macro F1 tertinggi.</p>
        </div>
        """
        vote_note = f"Tiebreaker: SVM → {mv_display}"
    else:
        tie_html = ""
        vote_note = f"{max_votes} dari 4 model sepakat"

    majority_html = f"""
    {tie_html}
    <div class="mv-banner {mv_css}">
        <div class="mv-title">HASIL KEPUTUSAN MAYORITAS</div>
        <div class="mv-label">{mv_display}</div>
        <div class="mv-note">{vote_note}</div>
    </div>
    """

    cards = []
    for model_name, prediction in predictions.items():
        css_class, display_label = label_style(prediction)
        cards.append(
            f"""
            <div class="result-card">
                <div class="model-name">{model_name}</div>
                <div class="result-label {css_class}">{display_label}</div>
            </div>
            """
        )
    models_html = f'<div class="result-grid">{"".join(cards)}</div>'

    vote_lines = ["### 📊 Rincian Voting"]
    for label, count in vote_counts.most_common():
        bar = "█" * count + "░" * (4 - count)
        vote_lines.append(f"- **{label}:** `{bar}` {count}/4 model")

    return (
        f"**Teks setelah preprocessing:** `{cleaned_input}`",
        majority_html,
        models_html,
        "\n".join(vote_lines),
    )


CSS = """
.gradio-container { max-width: 1100px !important; margin: auto !important; }
.hero { text-align: center; margin-bottom: 1rem; }
.hero h1 { margin-bottom: 0.25rem; }
.desc-grid, .result-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
}
.result-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.desc-card, .result-card { border-radius: 12px; padding: 16px; box-sizing: border-box; }
.desc-card h4 { margin: 0 0 8px; }
.desc-card p { margin: 0; line-height: 1.5; }
.desc-hate { background: #fde8e8; border-left: 5px solid #e53e3e; }
.desc-abusive { background: #fef3e2; border-left: 5px solid #dd6b20; }
.desc-neutral { background: #e8f5e9; border-left: 5px solid #38a169; }
.result-card { text-align: center; border: 1px solid #e2e8f0; background: #f8fafc; }
.model-name { color: #718096; font-size: .8rem; font-weight: 700; margin-bottom: 8px; }
.result-label { font-size: 1.05rem; font-weight: 800; }
.label-hate { color: #e53e3e; }
.label-abusive { color: #dd6b20; }
.label-neutral { color: #38a169; }
.tie-banner { border-radius: 14px; padding: 16px; text-align: center; margin-bottom: 12px;
    background: #fefcbf; border: 2px solid #d69e2e; color: #744210; }
.tie-title { font-size: 1.05rem; font-weight: 800; }
.tie-banner p { margin-bottom: 0; }
.mv-banner { border-radius: 14px; padding: 22px; text-align: center; color: white; }
.mv-hate { background: linear-gradient(135deg, #fc8181, #e53e3e); }
.mv-abusive { background: linear-gradient(135deg, #f6ad55, #dd6b20); }
.mv-neutral { background: linear-gradient(135deg, #68d391, #38a169); }
.mv-title { font-size: .85rem; font-weight: 700; opacity: .9; }
.mv-label { font-size: 2rem; font-weight: 800; }
.mv-note { font-size: .85rem; opacity: .9; }
@media (max-width: 800px) {
    .desc-grid, .result-grid { grid-template-columns: 1fr; }
}
"""


CATEGORY_GUIDE = """
<div class="desc-grid">
    <div class="desc-card desc-hate">
        <h4>⚠️ Hate Speech</h4>
        <p>Ujaran kebencian yang menargetkan individu atau kelompok berdasarkan
        identitas tertentu dan berpotensi memicu diskriminasi atau kekerasan.</p>
    </div>
    <div class="desc-card desc-abusive">
        <h4>🤬 Abusive</h4>
        <p>Kata kasar, makian, atau bahasa tidak sopan yang tidak secara spesifik
        menargetkan kelompok berdasarkan identitas tertentu.</p>
    </div>
    <div class="desc-card desc-neutral">
        <h4>✅ Neutral</h4>
        <p>Teks yang tidak mengandung ujaran kebencian maupun bahasa kasar.</p>
    </div>
</div>
"""


with gr.Blocks(title="ToxiCheck") as demo:
    gr.HTML(
        """
        <div class="hero">
            <h1>ToxiCheck 🕵️</h1>
            <p>Deteksi Hate Speech dan Abusive Language dalam teks berbahasa Indonesia</p>
        </div>
        """
    )
    gr.Markdown(
        "Aplikasi membandingkan prediksi **Naive Bayes, K-NN, Logistic Regression,** "
        "dan **SVM (Linear)**, kemudian menentukan hasil melalui majority vote."
    )

    user_input = gr.Textbox(
        label="Teks yang ingin dianalisis",
        placeholder="Masukkan tweet atau kalimat berbahasa Indonesia...",
        lines=5,
    )
    detect_button = gr.Button("🔍 Deteksi Teks", variant="primary")

    gr.Markdown("## 📖 Panduan Kategori")
    gr.HTML(CATEGORY_GUIDE)

    gr.Markdown("## 🗳️ Hasil Analisis")
    cleaned_output = gr.Markdown()
    majority_output = gr.HTML()
    gr.Markdown("### 🤖 Prediksi per Model")
    models_output = gr.HTML()
    voting_output = gr.Markdown()

    detect_button.click(
        fn=predict_text,
        inputs=user_input,
        outputs=[cleaned_output, majority_output, models_output, voting_output],
    )
    user_input.submit(
        fn=predict_text,
        inputs=user_input,
        outputs=[cleaned_output, majority_output, models_output, voting_output],
    )


if __name__ == "__main__":
    demo.launch(css=CSS, ssr_mode=False)
