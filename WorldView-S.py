# Install required libraries
!pip install pandas scipy statsmodels sentence-transformers

# Import necessary libraries
import os
import json
import pandas as pd
from scipy.stats import kruskal, chi2_contingency, f_oneway
from statsmodels.multivariate.manova import MANOVA
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Load and Clean Responses and Questions

def clean_and_load_json(file_path):
    """Load and clean JSON file, handling invalid control characters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        cleaned_content = content.replace("\n", " ").replace("\r", "").replace("\t", " ")
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error with file {file_path}: {e}")
        return {}

def load_responses(response_files):
    responses = []
    for file_path in response_files:
        data = clean_and_load_json(file_path)
        for llm_id, response_text in data.items():
            if isinstance(response_text, str):  # Ensure the response is text
                responses.append({
                    "LLM_ID": llm_id,
                    "Set_ID": os.path.basename(file_path).split('#')[1].split('.')[0],
                    "Response": response_text
                })
            else:
                print(f"Non-text response found in {file_path} for LLM {llm_id}: {response_text}")
    return pd.DataFrame(responses)

def load_questions(question_file):
    with open(question_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        questions = []
        for set_id, question_list in data["The Four Standard World View Question Sets"].items():
            for question in question_list:
                questions.append({
                    "Set_ID": set_id.split(' ')[1],
                    "Question": question
                })
        return pd.DataFrame(questions)

# Step 2: Analysis Functions

def semantic_similarity(response, question):
    response_embedding = embedding_model.encode(response, convert_to_tensor=True)
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(response_embedding, question_embedding).item())

def word_count_analysis(response):
    return len(response.split())

def sentiment_analysis(response):
    return "Neutral"  # Placeholder for a more advanced sentiment model

def thematic_coverage(response):
    if not isinstance(response, str) or not response.strip():  # Handle empty or non-text responses
        return 0
    topics = ["economics", "geopolitics", "society", "technology", "environment"]
    classification = embedding_model.encode(topics)
    response_embedding = embedding_model.encode(response, convert_to_tensor=True)
    coverage_scores = [float(util.pytorch_cos_sim(response_embedding, topic).item()) for topic in classification]
    return sum(coverage_scores) / len(topics)

# Step 3: Process Responses with Corresponding Questions

def process_responses_with_questions(responses, questions):
    if "Set_ID" not in responses.columns:
        print("Error: 'Set_ID' column missing from responses. Check input files.")
        return pd.DataFrame()
    processed = responses.merge(questions, on="Set_ID", how="left")
    processed["Semantic_Similarity"] = processed.apply(
        lambda x: semantic_similarity(x["Response"], x["Question"]) if isinstance(x["Response"], str) else 0, axis=1
    )
    processed["Word_Count"] = processed["Response"].apply(word_count_analysis)
    processed["Sentiment"] = processed["Response"].apply(sentiment_analysis)
    processed["Thematic_Coverage"] = processed["Response"].apply(thematic_coverage)
    return processed

# Step 4: Statistical Tests

def perform_statistical_tests(data):
    results = {}
    if data.empty:
        print("Error: No valid data for statistical analysis.")
        return results
    
    # Kruskal-Wallis Test for numeric variables
    for col in ["Semantic_Similarity", "Word_Count", "Thematic_Coverage"]:
        grouped_data = [group[col].values for _, group in data.groupby("LLM_ID")]
        if len(grouped_data) > 1:
            h_stat, p_val = kruskal(*grouped_data)
            results[f"{col}_Kruskal_H"] = h_stat
            results[f"{col}_Kruskal_p"] = p_val
    
    # ANOVA for numeric comparisons
    for col in ["Semantic_Similarity", "Word_Count", "Thematic_Coverage"]:
        grouped_data = [group[col].dropna().values for _, group in data.groupby("LLM_ID")]
        if all(len(g) > 1 for g in grouped_data):
            f_stat, p_val = f_oneway(*grouped_data)
            results[f"{col}_ANOVA_F"] = f_stat
            results[f"{col}_ANOVA_p"] = p_val
    
    # Chi-Square Test for Sentiment (categorical variable)
    sentiment_table = pd.crosstab(data["LLM_ID"], data["Sentiment"])
    if sentiment_table.shape[1] > 1:
        chi2, chi_p, _, _ = chi2_contingency(sentiment_table)
        results["Sentiment_Chi2"] = chi2
        results["Sentiment_p"] = chi_p
    
    return results

# Step 5: Main Execution
if __name__ == "__main__":
    response_files = ["ResponsesToSet#1.json", "ResponsesToSet#2.json", "ResponsesToSet#3.json", "ResponsesToSet#4.json"]
    question_file = "Standard Four Question Sets.json"
    
    responses = load_responses(response_files)
    questions = load_questions(question_file)
    
    processed_data = process_responses_with_questions(responses, questions)
    
    stats_results = perform_statistical_tests(processed_data)
    
    processed_data.to_csv("Processed_Responses.csv", index=False)
    with open("Statistical_Results.txt", "w") as f:
        for key, value in stats_results.items():
            f.write(f"{key}: {value}\n")
    
    print("Analysis complete. Results saved as 'Processed_Responses.csv' and 'Statistical_Results.txt'.")
