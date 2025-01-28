# Install necessary dependencies
!pip install sentence-transformers transformers scikit-learn matplotlib networkx pandas

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import kruskal
from statsmodels.multivariate.manova import MANOVA
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_pipeline = pipeline("sentiment-analysis")
zero_shot_pipeline = pipeline("zero-shot-classification")

# Categories for Inductive Reasoning (Fine-tuned and Granularized)
geopolitical_labels = ["pro-West", "anti-globalization", "pro-sovereignty", "pro-globalization", "neutral"]
ideological_labels = ["liberalism", "realism", "socialism", "conservatism", "anarchism", "fascism"]
philosophical_labels = ["utilitarianism", "existentialism", "deontology", "pragmatism", "nihilism", "virtue ethics"]

# Step 1: Load and Parse JSON Files
def load_responses(file_paths):
    data = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                responses = json.load(file)
                for llm_id, response in responses.items():
                    if isinstance(response, str):  # Ensure response is a string
                        data.append({"LLM_ID": llm_id, "Response": response})
                    else:
                        print(f"Invalid response format for {llm_id} in {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return pd.DataFrame(data)

# Step 2: Inductive Reasoning Analysis
def analyze_inductive_reasoning(response):
    # Geopolitical Classification
    geo_result = zero_shot_pipeline(response, candidate_labels=geopolitical_labels, multi_label=True)
    geo_scores = {label: score for label, score in zip(geo_result['labels'], geo_result['scores'])}

    # Ideological Classification
    ideology_result = zero_shot_pipeline(response, candidate_labels=ideological_labels, multi_label=True)
    ideology_scores = {label: score for label, score in zip(ideology_result['labels'], ideology_result['scores'])}

    # Philosophical Classification
    philosophy_result = zero_shot_pipeline(response, candidate_labels=philosophical_labels, multi_label=True)
    philosophy_scores = {label: score for label, score in zip(philosophy_result['labels'], philosophy_result['scores'])}

    return geo_scores, ideology_scores, philosophy_scores

# Step 3: Statistical Analysis
def perform_statistical_tests(df, metric_columns):
    results = {}
    for metric in metric_columns:
        grouped_data = [group[metric].values for _, group in df.groupby("LLM_ID")]
        h_stat, p_val = kruskal(*grouped_data)
        results[f"{metric}_Kruskal_H"] = h_stat
        results[f"{metric}_Kruskal_p"] = p_val

    # MANOVA
    try:
        formula = ' + '.join(metric_columns) + ' ~ C(LLM_ID)'
        manova = MANOVA.from_formula(formula, data=df)
        results["MANOVA"] = str(manova.mv_test())
    except Exception as e:
        results["MANOVA_ERROR"] = str(e)

    return results

# Step 4: Visualization
def plot_heatmap(df, metric_columns, output_path="heatmap.png"):
    corr_matrix = df[metric_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Metrics")
    plt.savefig(output_path)
    plt.close()

# Step 5: Main Execution
def main():
    # File paths for JSON files
    file_paths = ["ResponsesToSet#1.json", "ResponsesToSet#2.json", "ResponsesToSet#3.json", "ResponsesToSet#4.json"]

    # Load responses
    df = load_responses(file_paths)

    # Add inductive reasoning metrics
    geo_scores_list, ideology_scores_list, philosophy_scores_list = [], [], []
    for response in df["Response"]:
        geo_scores, ideology_scores, philosophy_scores = analyze_inductive_reasoning(response)
        geo_scores_list.append(geo_scores)
        ideology_scores_list.append(ideology_scores)
        philosophy_scores_list.append(philosophy_scores)

    df["Geopolitical_Scores"] = geo_scores_list
    df["Ideological_Scores"] = ideology_scores_list
    df["Philosophical_Scores"] = philosophy_scores_list

    # Convert scores to numerical metrics for analysis
    for label in geopolitical_labels:
        df[label] = df["Geopolitical_Scores"].apply(lambda x: x.get(label, 0))
    for label in ideological_labels:
        df[label] = df["Ideological_Scores"].apply(lambda x: x.get(label, 0))
    for label in philosophical_labels:
        df[label] = df["Philosophical_Scores"].apply(lambda x: x.get(label, 0))

    # Statistical Analysis
    metric_columns = geopolitical_labels + ideological_labels + philosophical_labels
    stats_results = perform_statistical_tests(df, metric_columns)

    # Save results
    df.to_csv("WorldView_Analysis.csv", index=False)
    with open("Statistical_Results.txt", "w") as f:
        for key, value in stats_results.items():
            f.write(f"{key}: {value}\n")

    # Visualization
    plot_heatmap(df, metric_columns, output_path="WorldView_Heatmap.png")

    print("Analysis complete. Results saved as 'WorldView_Analysis.csv', 'Statistical_Results.txt', and 'WorldView_Heatmap.png'.")

if __name__ == "__main__":
    main()

