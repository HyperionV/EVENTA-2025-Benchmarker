import streamlit as st
import json
from rouge_score import rouge_scorer
from bert_score import score as bert_scorer_func
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from io import BytesIO
import plotly.express as px

st.set_page_config(layout="wide")

metric_explanations = {
    "BLEU-4": "BLEU measures how similar the generated text is to the reference by counting matching n-grams. Scores range from 0 to 1, higher is better.",
    "ROUGE-1": "ROUGE-1 measures unigram overlap between generated and reference texts. Higher scores (0-1) indicate better quality.",
    "ROUGE-2": "ROUGE-2 measures bigram overlap between generated and reference texts. Higher scores (0-1) indicate better quality.",
    "ROUGE-L": "ROUGE-L measures the longest common subsequence between generated and reference texts. Higher scores (0-1) indicate better quality.",
    "METEOR": "METEOR evaluates text by aligning words, considering synonyms and stemming. Scores range from 0 to 1, higher is better.",
    "BERTScore F1": "BERTScore uses BERT embeddings for semantic similarity. Scores range from 0 to 1, higher indicates better semantic match."
}

available_metrics = ["BLEU-4", "ROUGE-1",
                     "ROUGE-2", "ROUGE-L", "METEOR", "BERTScore F1"]

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
    results['Basic Stats'] = {
        'Generated Words': len(word_tokenize(generated_text)),
        'Original Words': len(word_tokenize(original_text)),
        'Generated Chars': len(generated_text),
        'Original Chars': len(original_text)
    }
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    rouge_scores = scorer.score(original_text, generated_text)
    results['ROUGE'] = {
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure,
        'ROUGE-Lsum': rouge_scores['rougeLsum'].fmeasure
    }
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


def get_all_metrics(focused_results):
    return {
        'BLEU-4': focused_results['BLEU'].get('BLEU-4', 0.0),
        'ROUGE-1': focused_results['ROUGE'].get('ROUGE-1', 0.0),
        'ROUGE-2': focused_results['ROUGE'].get('ROUGE-2', 0.0),
        'ROUGE-L': focused_results['ROUGE'].get('ROUGE-L', 0.0),
        'METEOR': focused_results['METEOR'].get('METEOR', 0.0),
        'BERTScore F1': focused_results['BERTScore'].get('F1', 0.0)
    }


def compute_evaluations(tc):
    tc_title = tc['title']
    original_text = tc['content']
    if 'evaluations' not in st.session_state:
        st.session_state.evaluations = {}
    st.session_state.evaluations[tc_title] = {}
    for model in st.session_state.models:
        if model in tc:
            generated_text = tc[model]['content']
            _, focused_results = calculate_metrics(
                generated_text, original_text)
            model_metrics = get_all_metrics(focused_results)
            st.session_state.evaluations[tc_title][model] = model_metrics


def display_evaluations(tc_title):
    if tc_title in st.session_state.get('evaluations', {}):
        st.divider()
        st.subheader("Metrics")
        evaluations = st.session_state.evaluations[tc_title]
        models = list(evaluations.keys())

        if len(models) < 2:
            st.warning(
                "At least two models are required to display the evaluation comparison.")
            st.divider()
            return
        if len(models) > 2:
            st.warning(
                f"More than two models detected ({len(models)}). Displaying only the first two: {models[0]} and {models[1]}.")

        col1, col2 = st.columns(2)

        with col1:
            model1 = models[0]
            st.write(f"#### {model1}")
            metrics1 = evaluations[model1]
            for metric in st.session_state.selected_metrics:
                if metric in metrics1:
                    st.metric(
                        label=metric, value=f"{metrics1[metric]:.4f}", help=metric_explanations.get(metric, ""))

        with col2:
            model2 = models[1]
            st.write(f"#### {model2}")
            metrics2 = evaluations[model2]
            for metric in st.session_state.selected_metrics:
                if metric in metrics2:
                    st.metric(
                        label=metric, value=f"{metrics2[metric]:.4f}", help=metric_explanations.get(metric, ""))
        st.divider()


def compute_overall_evaluations():
    if 'evaluations' not in st.session_state or not st.session_state.evaluations:
        return {}
    overall = {model: {metric: [] for metric in available_metrics}
               for model in st.session_state.models}
    for tc_title in st.session_state.evaluations:
        for model in st.session_state.evaluations[tc_title]:
            if model in overall:
                for metric in available_metrics:
                    if metric in st.session_state.evaluations[tc_title][model]:
                        overall[model][metric].append(
                            st.session_state.evaluations[tc_title][model][metric])
    for model in overall:
        for metric in overall[model]:
            if overall[model][metric]:
                overall[model][metric] = sum(
                    overall[model][metric]) / len(overall[model][metric])
            else:
                overall[model][metric] = 0.0
    return overall


if 'models' not in st.session_state:
    st.session_state.models = []
if 'test_cases' not in st.session_state:
    st.session_state.test_cases = []
if 'selected_metrics' not in st.session_state:
    st.session_state.selected_metrics = available_metrics

with st.sidebar:
    st.subheader("Configuration")
    model_input = st.text_input("Enter model name")
    if st.button("Add Model"):
        if model_input and model_input not in st.session_state.models:
            st.session_state.models.append(model_input)
            st.success(f"Added model: {model_input}")
        else:
            st.warning("Model name already exists or is empty")
    st.write("Current models:", ", ".join(st.session_state.models))

    st.session_state.selected_metrics = st.multiselect(
        "Select metrics to display", available_metrics, default=available_metrics
    )

    st.subheader("Add New Test Case")
    new_title = st.text_input("Title")
    new_content = st.text_area("Original Content", height=200)
    new_summaries = {}
    for model in st.session_state.models:
        summary_content = st.text_area(
            f"{model} Summary", height=100, key=f"summary_{model}")
        time_input = st.text_input(
            f"{model} Generation Time", key=f"time_{model}")
        new_summaries[model] = {"content": summary_content, "time": time_input}
    if st.button("Add Test Case"):
        if new_title and new_content:
            new_test_case = {"title": new_title, "content": new_content}
            new_test_case.update(new_summaries)
            st.session_state.test_cases.append(new_test_case)
            st.success("Test case added")
        else:
            st.warning("Please enter title and original content")

st.title("Summarization Benchmark")

st.subheader("Import Data")
uploaded_file = st.file_uploader("Upload JSON data", type="json")
auto_evaluate = st.checkbox("Auto Evaluate", value=False)

if uploaded_file is not None:
    data = json.load(uploaded_file)
    st.session_state.models = data['models']
    st.session_state.test_cases = data['content']
    st.success("Data imported successfully")
    if auto_evaluate:
        total_cases = len(st.session_state.test_cases)
        if total_cases > 0:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Starting evaluation...")
            for i, tc in enumerate(st.session_state.test_cases):
                compute_evaluations(tc)
                progress = (i + 1) / total_cases
                progress_bar.progress(progress)
                status_text.text(f"Evaluating test case {i + 1}/{total_cases}")
            status_text.text("Evaluation completed!")
            evaluation_results = []
            for tc in st.session_state.test_cases:
                tc_title = tc['title']
                for model in st.session_state.models:
                    if model in tc and tc_title in st.session_state.evaluations and model in st.session_state.evaluations[tc_title]:
                        metrics = st.session_state.evaluations[tc_title][model]
                        row = {
                            'Title': tc_title,
                            'Model': model,
                            'Summary': tc[model]['content'],
                            'Generation Time': tc[model]['time']
                        }
                        row.update(metrics)
                        evaluation_results.append(row)
            df = pd.DataFrame(evaluation_results)
            output = BytesIO()
            df.to_excel(output, index=False, engine='xlsxwriter')
            output.seek(0)
            btn = st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name="evaluation_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No test cases available to evaluate.")

# Article Selection and Display
if st.session_state.test_cases:
    st.subheader("Select Article to View")
    article_titles = [tc['title'] for tc in st.session_state.test_cases]
    selected_article = st.selectbox("Select an article", article_titles)
    for tc in st.session_state.test_cases:
        if tc['title'] == selected_article:
            st.write(f"## {tc['title']}")
            with st.expander("Original Content"):
                st.write(tc['content'])
            st.write("### Summaries")
            for model in st.session_state.models:
                if model in tc:
                    with st.expander(f"{model} Summary"):
                        st.write(tc[model]['content'])
                        st.write(f"Generation Time: {tc[model]['time']}")
                else:
                    st.write(f"{model}: No summary available")
            if selected_article in st.session_state.get('evaluations', {}):
                display_evaluations(selected_article)
            else:
                if st.button("Evaluate Summaries"):
                    with st.spinner("Evaluating..."):
                        compute_evaluations(tc)
                    display_evaluations(selected_article)
            break
    if 'evaluations' in st.session_state and st.session_state.evaluations:
        overall_evaluations = compute_overall_evaluations()
        st.subheader("Overall Evaluation")
        models = list(overall_evaluations.keys())
        if len(models) == 2:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"#### {models[0]}")
                for metric in st.session_state.selected_metrics:
                    score = overall_evaluations[models[0]].get(metric, 0.0)
                    st.metric(
                        label=metric, value=f"{score:.4f}", help=metric_explanations.get(metric, ""))
            with col2:
                st.write(f"#### {models[1]}")
                for metric in st.session_state.selected_metrics:
                    score = overall_evaluations[models[1]].get(metric, 0.0)
                    st.metric(
                        label=metric, value=f"{score:.4f}", help=metric_explanations.get(metric, ""))
        else:
            st.write("Overall average metrics:")
            for model in models:
                st.write(f"#### {model}")
                for metric in st.session_state.selected_metrics:
                    score = overall_evaluations[model].get(metric, 0.0)
                    st.metric(
                        label=metric, value=f"{score:.4f}", help=metric_explanations.get(metric, ""))
        st.subheader("Visualization")
        if models and st.session_state.selected_metrics:
            vis_data = {model: [overall_evaluations[model].get(
                metric, 0.0) for metric in st.session_state.selected_metrics] for model in models}
            df = pd.DataFrame(
                vis_data, index=st.session_state.selected_metrics)
            df_long = df.reset_index().melt(
                id_vars='index', var_name='Model', value_name='Score')
            fig = px.bar(df_long, x='index', y='Score', color='Model',
                         barmode='group', title="Average Metrics Comparison")
            fig.update_layout(xaxis_title="Metrics", yaxis_title="Score")
            st.plotly_chart(fig)
        else:
            st.write("No data to visualize.")
else:
    st.warning("No data available. Please import data or add test cases.")
