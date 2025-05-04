import streamlit as st
from rouge_score import rouge_scorer
from bert_score import score as bert_scorer_func
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

st.set_page_config(layout="wide")


@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)


download_nltk_data()


@st.cache_data
def calculate_metrics(generated_text, original_text):
    results = {}

    # Basic Stats
    results['Basic Stats'] = {
        'Generated Words': len(word_tokenize(generated_text)),
        'Original Words': len(word_tokenize(original_text)),
        'Generated Chars': len(generated_text),
        'Original Chars': len(original_text)
    }

    # ROUGE
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    rouge_scores = scorer.score(original_text, generated_text)
    results['ROUGE'] = {
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure,
        'ROUGE-Lsum': rouge_scores['rougeLsum'].fmeasure
    }

    # BERTScore
    try:
        P, R, F1 = bert_scorer_func([generated_text], [
                                    original_text], lang="en", rescale_with_baseline=True, verbose=False)
        results['BERTScore'] = {
            'Precision': P.mean().item(),
            'Recall': R.mean().item(),
            'F1': F1.mean().item()
        }
    except Exception as e:
        results['BERTScore'] = {'Error': f"Could not calculate BERTScore: {e}"}

    # NLTK-based Scores
    original_tokens = word_tokenize(original_text)
    generated_tokens = word_tokenize(generated_text)

    if original_tokens and generated_tokens:
        # BLEU (using sentence_bleu, expects list of references)
        results['BLEU'] = {
            # BLEU-4
            'BLEU-4': sentence_bleu([original_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        }
        # METEOR
        results['METEOR'] = {
            'METEOR': single_meteor_score(original_tokens, generated_tokens)
        }
    else:
        results['BLEU'] = {'BLEU-4': 0.0}
        results['METEOR'] = {'METEOR': 0.0}

    # TF-IDF Cosine Similarity
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(
            [original_text, generated_text])
        results['Cosine Similarity'] = {
            'TF-IDF Cosine': cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        }
    except ValueError:  # Handle case with empty strings after tokenization/preprocessing
        results['Cosine Similarity'] = {'TF-IDF Cosine': 0.0}

    return results


st.title("📝 Text Generation Benchmark")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Text (Reference)")
    original_text = st.text_area(
        "Original Text", height=300, key="original", label_visibility="collapsed")

with col2:
    st.subheader("Generated Text (Candidate)")
    generated_text = st.text_area(
        "Generated Text", height=300, key="generated", label_visibility="collapsed")

if st.button("📊 Calculate Scores", use_container_width=True):
    if generated_text and original_text:
        with st.spinner("Calculating scores... This might take a moment, especially for BERTScore."):
            scores = calculate_metrics(generated_text, original_text)

        st.subheader("📊 Benchmark Results")

        # Display results in a more structured way
        df_data = []
        for category, metrics in scores.items():
            for metric, value in metrics.items():
                if isinstance(value, float):
                    value = round(value, 4)
                df_data.append(
                    {'Category': category, 'Metric': metric, 'Score': value})

        df = pd.DataFrame(df_data)

        st.dataframe(df, use_container_width=True, hide_index=True)


    elif not original_text:
        st.warning("Please enter the Original Text.")
    elif not generated_text:
        st.warning("Please enter the Generated Text.")
