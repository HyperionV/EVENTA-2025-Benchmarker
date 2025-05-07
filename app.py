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

# Metric explanations for tooltips
metric_explanations = {
    "BLEU-4": "BLEU (Bilingual Evaluation Understudy) measures how similar the generated text is to the reference by counting matching n-grams. Scores range from 0 to 1, where higher values indicate better matches. Commonly used in machine translation evaluation.",
    "ROUGE-1": "ROUGE-1 measures the overlap of unigrams (single words) between generated and reference texts. Higher scores (0-1) indicate better quality.",
    "ROUGE-2": "ROUGE-2 measures the overlap of bigrams (word pairs) between generated and reference texts. Higher scores (0-1) indicate better quality.",
    "ROUGE-L": "ROUGE-L measures the longest common subsequence between generated and reference texts. Higher scores (0-1) indicate better quality.",
    "METEOR": "METEOR (Metric for Evaluation of Translation with Explicit Ordering) evaluates text by aligning and scoring words between generated and reference texts, considering synonyms and stemming. Scores range from 0 to 1, with higher values indicating better quality.",
    "BERTScore F1": "BERTScore leverages BERT embeddings to measure semantic similarity between generated and reference texts. Scores typically range from 0 to 1, with higher values indicating better semantic match. More robust than n-gram based metrics for capturing meaning."
}


@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab/english/')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
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

    original_tokens = word_tokenize(original_text)
    generated_tokens = word_tokenize(generated_text)

    if original_tokens and generated_tokens:
        results['BLEU'] = {
            'BLEU-4': sentence_bleu([original_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        }
        results['METEOR'] = {
            'METEOR': single_meteor_score(original_tokens, generated_tokens)
        }
    else:
        results['BLEU'] = {'BLEU-4': 0.0}
        results['METEOR'] = {'METEOR': 0.0}

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(
            [original_text, generated_text])
        results['Cosine Similarity'] = {
            'TF-IDF Cosine': cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        }
    except ValueError:  
        results['Cosine Similarity'] = {'TF-IDF Cosine': 0.0}

    focused_results = {
        'BLEU': {'BLEU-4': results['BLEU']['BLEU-4']},
        'ROUGE': {
            'ROUGE-1': results['ROUGE']['ROUGE-1'],
            'ROUGE-2': results['ROUGE']['ROUGE-2'],
            'ROUGE-L': results['ROUGE']['ROUGE-L']
        },
        'METEOR': {'METEOR': results['METEOR']['METEOR']},
        'BERTScore': {'F1': results['BERTScore'].get('F1', 0.0)}
    }

    return results, focused_results


st.title("📝 Text Generation Benchmark")
st.caption("Focused on BLEU, ROUGE, METEOR, and BERTScore metrics")

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
            results, focused_results = calculate_metrics(
                generated_text, original_text)

        st.subheader("📊 Benchmark Results")

        df_data = []
        for category, metrics in results.items():
            for metric, value in metrics.items():
                if isinstance(value, float):
                    value = round(value, 4)
                df_data.append(
                    {'Category': category, 'Metric': metric, 'Score': value})

        df = pd.DataFrame(df_data)

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("🔍 Focused Metrics")

        df_focused_data = []
        for category, metrics in focused_results.items():
            for metric, value in metrics.items():
                if isinstance(value, float):
                    value = round(value, 4)
                tooltip = ""
                if metric == "BLEU-4":
                    tooltip = metric_explanations["BLEU-4"]
                elif metric.startswith("ROUGE"):
                    tooltip = metric_explanations.get(metric, "")
                elif metric == "METEOR":
                    tooltip = metric_explanations["METEOR"]
                elif metric == "F1":
                    tooltip = metric_explanations["BERTScore F1"]
                df_focused_data.append(
                    {'Category': category, 'Metric': metric, 'Score': value, 'Description': tooltip})

        df_focused = pd.DataFrame(df_focused_data)

        st.dataframe(
            df_focused,
            column_config={
                "Category": st.column_config.TextColumn(
                    "Category",
                    help="The category of the metric",
                ),
                "Metric": st.column_config.TextColumn(
                    "Metric",
                    help="The specific metric name",
                ),
                "Score": st.column_config.NumberColumn(
                    "Score",
                    help="Score value (usually between 0-1, higher is better)",
                    format="%.4f",
                ),
                "Description": st.column_config.TextColumn(
                    "Description",
                    help="Explanation of what the metric means",
                ),
            },
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("📈 Key Metrics Visualization")
        st.caption("Hover over each metric name for a detailed explanation")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="BLEU-4",
                value=f"{focused_results['BLEU']['BLEU-4']:.4f}",
                help=metric_explanations["BLEU-4"]
            )

        with col2:
            st.metric(
                label="ROUGE-L",
                value=f"{focused_results['ROUGE']['ROUGE-L']:.4f}",
                help=metric_explanations["ROUGE-L"]
            )

        with col3:
            st.metric(
                label="METEOR",
                value=f"{focused_results['METEOR']['METEOR']:.4f}",
                help=metric_explanations["METEOR"]
            )

        with col4:
            st.metric(
                label="BERTScore F1",
                value=f"{focused_results['BERTScore']['F1']:.4f}",
                help=metric_explanations["BERTScore F1"]
            )

    elif not original_text:
        st.warning("Please enter the Original Text.")
    elif not generated_text:
        st.warning("Please enter the Generated Text.")
