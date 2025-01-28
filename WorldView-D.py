# Install necessary dependencies
!pip install sentence-transformers transformers scikit-learn matplotlib networkx pandas

import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from scipy.stats import kruskal, chi2_contingency
from statsmodels.multivariate.manova import MANOVA
import networkx as nx
import matplotlib.pyplot as plt

# Initialize advanced NLP models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Semantic embeddings
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Step 1: Extract responses from JSON files
def extract_responses_json(file_paths):
    responses = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            responses.extend(data.get("responses", []))
    return responses

# Step 2: Advanced Semantic Analysis
def semantic_similarity(response, reference):
    response_embedding = embedding_model.encode(response, convert_to_tensor=True)
    reference_embedding = embedding_model.encode(reference, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(response_embedding, reference_embedding).item())

def sentiment_analysis(response):
    sentiment = sentiment_pipeline(response[:512])  # Analyze only first 512 tokens for efficiency
    return sentiment[0]['label'], sentiment[0]['score']

# Step 3: Thematic Modeling
def extract_themes(responses):
    vectorizer = CountVectorizer(max_df=1.0, min_df=1, stop_words='english')  # Adjusted thresholds
    dtm = vectorizer.fit_transform(responses)
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)  # 5 topics
    lda_model.fit(dtm)
    topics = lda_model.transform(dtm)
    return topics, vectorizer.get_feature_names_out()

# Step 4: Inter-Topic Connectivity Analysis
def calculate_interconnections(responses):
    embeddings = embedding_model.encode(responses, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()
    graph = nx.Graph()
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            if similarity_matrix[i, j] > 0.2:  # Lowered threshold
                graph.add_edge(i, j, weight=similarity_matrix[i, j])
    if len(graph) == 0:  # Handle empty graph
        return None
    return graph

# Step 5: Process Responses into Metrics
def process_responses(responses):
    data = []
    reference_prompt = "Describe the causes and consequences of global challenges."
    for entry in responses:
        response = entry["Response"]
        themes, _ = extract_themes([response])
        sentiment_label, sentiment_score = sentiment_analysis(response)
        graph = calculate_interconnections([response])
        depth = nx.average_shortest_path_length(graph) if graph and nx.is_connected(graph) else 0

        data.append({
            "LLM_ID": entry["LLM_ID"],
            "Set_ID": entry["Set_ID"],
            "Semantic_Similarity": semantic_similarity(response, reference_prompt),
            "Sentiment_Label": sentiment_label,
            "Sentiment_Score": sentiment_score,
            "Thematic_Diversity": len(themes[0]),
            "Interconnections": depth
        })
    return pd.DataFrame(data)

# Step 6: Statistical Analysis
def perform_statistical_tests(data):
    results = {}

    # Kruskal-Wallis Test for numeric variables
    for col in ["Semantic_Similarity", "Sentiment_Score", "Thematic_Diversity", "Interconnections"]:
        grouped_data = [group[col].values for _, group in data.groupby("LLM_ID")]
        if all(np.allclose(group, grouped_data[0]) for group in grouped_data):
            print(f"Skipping Kruskal-Wallis test for '{col}' as all values are identical.")
            results[f"{col}_Kruskal_H"] = "Skipped (identical values)"
            results[f"{col}_Kruskal_p"] = "Skipped (identical values)"
        else:
            h_stat, p_val = kruskal(*grouped_data)
            results[f"{col}_Kruskal_H"] = h_stat
            results[f"{col}_Kruskal_p"] = p_val

    # Chi-Square Test for categorical variables
    contingency_table = pd.crosstab(data["LLM_ID"], data["Sentiment_Label"])
    if contingency_table.shape[1] > 1:
        chi2, chi_p, _, _ = chi2_contingency(contingency_table)
        results["Sentiment_Label_Chi2"] = chi2
        results["Sentiment_Label_p"] = chi_p
    else:
        print("Skipping Chi-Square test for 'Sentiment_Label' as not enough categories exist.")
        results["Sentiment_Label_Chi2"] = "Skipped (insufficient categories)"
        results["Sentiment_Label_p"] = "Skipped (insufficient categories)"

    # MANOVA for multivariate analysis
    try:
        manova = MANOVA.from_formula(
            "Semantic_Similarity + Sentiment_Score + Thematic_Diversity + Interconnections ~ C(LLM_ID)",
            data=data
        )
        results["MANOVA"] = str(manova.mv_test())
    except Exception as e:
        results["MANOVA_ERROR"] = str(e)

    return results

# Step 7: Visualization
def visualize_interconnections(graph, output_path="interconnections_graph.png"):
    if graph is None or len(graph.nodes) == 0:  # Handle empty graph case
        print("Graph is empty. No visualization created.")
        return
    plt.figure(figsize=(10, 8))
    nx.draw(graph, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10)
    plt.savefig(output_path)

# Step 8: Main Execution
if __name__ == "__main__":
    # Define file paths for JSON files
    json_files = ["ResponsesToSet#1.json", "ResponsesToSet#2.json", "ResponsesToSet#3.json", "ResponsesToSet#4.json"]

    # Extract, process, and analyze responses
    responses = extract_responses_json(json_files)
    data = process_responses(responses)
    stats_results = perform_statistical_tests(data)

    # Save results
    data.to_csv("WorldView-D_Analysis.csv", index=False)
    with open("WorldView-D_Statistical_Results.txt", "w") as f:
        for key, value in stats_results.items():
            f.write(f"{key}: {value}\n")

    # Generate interconnection graph for visualization
    all_responses = [entry["Response"] for entry in responses]
    graph = calculate_interconnections(all_responses)
    visualize_interconnections(graph)

    print("Deep analysis complete. Results saved as 'WorldView-D_Analysis.csv' and 'WorldView-D_Statistical_Results.txt'.")
