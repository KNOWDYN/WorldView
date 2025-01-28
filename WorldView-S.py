!pip install transformers scipy statsmodels pandas

import os
import json
import pandas as pd
import numpy as np
from scipy.stats import kruskal, chi2_contingency
from statsmodels.multivariate.manova import MANOVA
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Initialize NLP models for fine-tuned analysis
sentiment_pipeline = pipeline("sentiment-analysis")
zero_shot_pipeline = pipeline("zero-shot-classification")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight sentence embedding model

# Categories for Zero-Shot Classification
ethical_labels = ["justice", "fairness", "human rights", "equity", "morality"]
bias_labels = ["Western-centric", "authoritarian", "politically biased", "economically biased"]

# Step 1: Extract responses from JSON files
def extract_responses_json(file_paths):
    responses = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            responses.extend(data.get("responses", []))
    return responses

# Step 2: Define Scoring Functions Using Advanced NLP

def score_factual_accuracy(response):
    return len(response.split()) % 5  # Placeholder for real fact-checking

def score_narrative_coherence(response):
    embedding = embedding_model.encode(response, convert_to_tensor=True)
    coherence_score = float(util.pytorch_cos_sim(embedding, embedding).item())
    return coherence_score

def score_bias_type(response):
    classification = zero_shot_pipeline(response, candidate_labels=bias_labels, multi_label=True)
    return max(classification['scores']) if classification['scores'] else 0

def score_ethics(response):
    classification = zero_shot_pipeline(response, candidate_labels=ethical_labels, multi_label=True)
    return max(classification['scores']) if classification['scores'] else 0

def score_interconnectedness(response):
    topics = ["economics", "geopolitics", "society", "technology", "environment"]
    classification = zero_shot_pipeline(response, candidate_labels=topics, multi_label=True)
    return sum(classification['scores']) / len(topics) if classification['scores'] else 0

# Step 3: Process Responses into a Structured Dataset
def process_responses(responses):
    data = []
    for entry in responses:
        data.append({
            "LLM_ID": entry["LLM_ID"],
            "Set_ID": entry["Set_ID"],
            "Factual_Accuracy": score_factual_accuracy(entry["Response"]),
            "Narrative_Coherence": score_narrative_coherence(entry["Response"]),
            "Bias_Type": score_bias_type(entry["Response"]),
            "Ethical_Framing": score_ethics(entry["Response"]),
            "Interconnectedness": score_interconnectedness(entry["Response"]),
        })
    return pd.DataFrame(data)

# Step 4: Perform Statistical Tests
def perform_statistical_tests(data):
    results = {}
    
    # Kruskal-Wallis Test for ordinal/numeric variables
    for col in ["Factual_Accuracy", "Narrative_Coherence", "Ethical_Framing", "Interconnectedness"]:
        grouped_data = [group[col].values for _, group in data.groupby("LLM_ID")]
        if all(len(g) > 1 for g in grouped_data):
            h_stat, p_val = kruskal(*grouped_data)
            results[f"{col}_Kruskal_H"] = h_stat
            results[f"{col}_Kruskal_p"] = p_val
    
    # Chi-Square Test for Bias_Type (categorical variable)
    contingency_table = pd.crosstab(data["LLM_ID"], data["Bias_Type"])
    if contingency_table.shape[1] > 1:
        chi2, chi_p, _, _ = chi2_contingency(contingency_table)
        results["Bias_Type_Chi2"] = chi2
        results["Bias_Type_p"] = chi_p
    
    # MANOVA for multivariate analysis
    try:
        manova = MANOVA.from_formula(
            "Factual_Accuracy + Narrative_Coherence + Ethical_Framing + Interconnectedness ~ C(LLM_ID)",
            data=data
        )
        results["MANOVA"] = str(manova.mv_test())
    except Exception as e:
        results["MANOVA_ERROR"] = str(e)
    
    return results

# Step 5: Main Execution
if __name__ == "__main__":
    # Define file paths for JSON files
    json_files = ["ResponsesToSet#1.json", "ResponsesToSet#2.json", "ResponsesToSet#3.json", "ResponsesToSet#4.json"]
    
    # Extract, process, and analyze responses
    responses = extract_responses_json(json_files)
    data = process_responses(responses)
    stats_results = perform_statistical_tests(data)
    
    # Save results
    data.to_csv("LLM_Response_Analysis.csv", index=False)
    with open("Statistical_Results.txt", "w") as f:
        for key, value in stats_results.items():
            f.write(f"{key}: {value}\n")
    
    print("Analysis complete. Results saved as 'LLM_Response_Analysis.csv' and 'Statistical_Results.txt'.")