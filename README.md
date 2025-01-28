# WorldView Project

## Introduction

WorldView is a research project designed to study bias and reasoning patterns across large language models (LLMs) in describing a coherent and holistic world view. By analyzing responses to carefully crafted question sets, the project aims to uncover latent biases, alignments, and worldview tendencies of LLMs. The findings will provide insights into how LLMs impact the knowledge economy and decision-making processes.

---

## Table of Contents

- [Introduction](#introduction)
- [Purpose and Context](#purpose-and-context)
- [Standard Question Sets](#standard-question-sets)
- [WorldView-S: Surface-Level Analysis](#worldview-s-surface-level-analysis)
  - [Features](#features)
  - [Code Implementation](#code-implementation)
- [WorldView-D: Deep-Level Analysis](#worldview-d-deep-level-analysis)
  - [Features](#features-1)
  - [Code Implementation](#code-implementation-1)
- [Getting Started](#getting-started)
- [License](#license)

---

## Purpose and Context

The rapid growth of LLMs raises concerns about how these models perceive, reason, and represent the world. The WorldView project addresses these questions by developing structured analysis pipelines:
- **WorldView-S**: For surface-level statistical and thematic analysis.
- **WorldView-D**: For deep-level inductive reasoning and classification.

---

## Standard Question Sets

The project uses four carefully designed question sets to assess LLMs' responses. Each set focuses on critical themes, including colonialism, global governance, climate change, and technology.

- **Set 1**: Focuses on colonial reparations, democracy, and cultural restitution.
- **Set 2**: Discusses climate reparations, NATO, and sustainability.
- **Set 3**: Analyzes global digital governance, AI risks, and sovereignty.
- **Set 4**: Explores strategies for low-income nations, cultural preservation, and universal income.

For the full question sets, refer to the [Standard Four Question Sets](./Standard%20Four%20Question%20Sets.json) file.

---

## WorldView-S: Surface-Level Analysis

### Features
WorldView-S provides statistical insights into LLM behavior. Key functionalities include:
1. **Semantic Similarity**: Measures how closely an LLM's response aligns with the question using sentence embeddings.
2. **Word Count Analysis**: Evaluates verbosity across LLMs.
3. **Sentiment Analysis**: Analyzes response sentiment using pre-trained models.
4. **Thematic Coverage**: Determines how well responses address predefined themes like economics, geopolitics, and environment.
5. **Statistical Testing**: Uses ANOVA, Kruskal-Wallis, and chi-square tests to identify significant differences.

### Code Implementation

#### 1. **Semantic Similarity**
Using cosine similarity:

```python
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(response, question):
    response_embedding = embedding_model.encode(response, convert_to_tensor=True)
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(response_embedding, question_embedding).item())
```

#### 2. **Kruskal-Wallis Test**
Non-parametric statistical test:

```python
from scipy.stats import kruskal

def kruskal_wallis_test(groups):
    """Perform Kruskal-Wallis test on multiple groups of responses"""
    return kruskal(*groups)
```

#### 3. **Thematic Coverage**
Computes the average similarity of a response with predefined themes:

```python
def thematic_coverage(response, topics=["economics", "geopolitics", "society", "technology", "environment"]):
    topic_embeddings = embedding_model.encode(topics)
    response_embedding = embedding_model.encode(response, convert_to_tensor=True)
    scores = [float(util.pytorch_cos_sim(response_embedding, topic).item()) for topic in topic_embeddings]
    return sum(scores) / len(topics)
```

---

## WorldView-D: Deep-Level Analysis

### Features
WorldView-D explores deeper inductive reasoning patterns using:
1. **Zero-Shot Classification**: Classifies responses into geopolitical, ideological, and philosophical categories.
2. **Latent Dirichlet Allocation (LDA)**: Identifies hidden topics within responses.
3. **Heatmap Visualization**: Displays correlations between metrics.
4. **MANOVA Testing**: Examines multivariate differences between LLMs.

### Code Implementation

#### 1. **Zero-Shot Classification**
Classifying responses into predefined categories:

```python
from transformers import pipeline

zero_shot_pipeline = pipeline("zero-shot-classification")

geopolitical_labels = ["pro-West", "anti-globalization", "pro-sovereignty", "pro-globalization", "neutral"]

def classify_geopolitical(response):
    result = zero_shot_pipeline(response, candidate_labels=geopolitical_labels, multi_label=True)
    return {label: score for label, score in zip(result['labels'], result['scores'])}
```

#### 2. **Topic Modeling (LDA)**
Extracting topics from responses:

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def extract_topics(responses, num_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(responses)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    return lda.components_
```

#### 3. **MANOVA for Statistical Testing**
Multivariate analysis of variance:

```python
from statsmodels.multivariate.manova import MANOVA
import pandas as pd

def perform_manova(df, metric_columns):
    formula = ' + '.join(metric_columns) + ' ~ C(LLM_ID)'
    manova = MANOVA.from_formula(formula, data=df)
    return manova.mv_test()
```

---

## Getting Started

1. Clone the repository:
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run `WorldView-S`:
   ```bash
   python WorldView-S.py
   ```
4. Run `WorldView-D`:
   ```bash
   python WorldView-D.py
   ```

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
```
