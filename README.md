# CineMatch: A Hybrid Movie Recommendation System with XAI

**CineMatch** is a prototype hybrid movie recommendation system developed as a capstone project for Data Science certification.  
This project combines **Collaborative Filtering** and **Content-Based Filtering** approaches to deliver personalized and accurate movie recommendations, enhanced with **Explainable AI (XAI)** features for transparency.

---

## Table of Contents

1. [Project Background](#1-project-background)  
2. [Key Features](#2-key-features)  
3. [Methodology (CRISP-DM)](#3-methodology-crisp-dm)  
4. [Technical Architecture](#4-technical-architecture)  
5. [Dataset](#5-dataset)  
6. [Project Structure](#6-project-structure)  
7. [Installation](#7-installation)  
8. [How to Run](#8-how-to-run)  
9. [Evaluation Results](#9-evaluation-results)  
10. [Limitations and Future Work](#10-limitations-and-future-work)  
11. [License](#11-license)  

---

## 1. Project Background

In the era of streaming and modern ticketing, users face the **Paradox of Choice** — too many options lead to confusion and frustration.  
For businesses, this can increase the risk of **user churn**.  

This project aims to address these challenges by:

- Delivering highly personalized recommendations  
- Tackling **cold-start** (new user) and **long-tail** (niche movie) problems  
- Building user trust through **transparency (XAI)**  

---

## 2. Key Features

- **Hybrid Model:**  
  Combines the strength of **SVD (Collaborative Filtering)** for personalized accuracy and **TF-IDF/Cosine Similarity (Content-Based Filtering)** to handle cold-start and feature relevance.

- **Semantic Feature Engineering:**  
  Creates a unique *“Content Soup”* feature that merges **Plot**, **Genre**, **Main Actors**, and **Director** for rich movie representation.

- **Explainable AI (XAI):**  
  Each recommendation includes on-the-fly explanations (e.g., *“Similar Actor: …”*, *“Same Director: …”*).

- **Interactive Prototype:**  
  Built with **Streamlit** for an end-to-end demonstration of the system workflow.

---

## 3. Methodology (CRISP-DM)

This project follows the **CRISP-DM** industry-standard framework to ensure a structured and validated process.

| Phase | Description |
|-------|--------------|
| **Business Understanding** | Define the churn problem caused by the *paradox of choice*. |
| **Data Understanding** | Use EDA to identify long-tail issues in movie ratings. |
| **Data Preparation** | Clean, integrate (metadata + credits), and engineer the *Content Soup* feature. |
| **Modeling** | Implement two engines (SVD and TF-IDF). |
| **Evaluation** | Validate SVD using Cross-Validation (RMSE < 0.9). |
| **Deployment** | Build a Streamlit prototype and export models using *pickle*. |

---

## 4. Technical Architecture

The system consists of two main engines integrated at the frontend.

### Engine 1: Collaborative Filtering (CF)
- **Algorithm:** SVD (Singular Value Decomposition) — `scikit-surprise`  
- **Purpose:** Predict the rating a user would give to a movie based on similar users’ behavior.  
- **Input:** `userId`  
- **Output:** List of movies with the highest predicted ratings  

### Engine 2: Content-Based (CB)
- **Algorithm:** TF-IDF Vectorizer & Cosine Similarity — `scikit-learn`  
- **Purpose:** Find semantically similar movies (plot, genre, actors, director) to the ones a user likes.  
- **Input:** List of 3–5 favorite movies  
- **Output:** List of movies with the highest content similarity scores  

### Hybrid Logic (in `app.py`)
Scores from both engines are normalized (Min-Max Scaling for CF) and combined with a **50/50 weighting** to generate the final hybrid score.

---

## 5. Dataset

This project uses **The Movies Dataset** from [Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset).  
The specific files used are:

- `movies_metadata.csv` — Movie information (plot, genre, ID)  
- `credits.csv` — Cast and crew details  
- `ratings_small.csv` — 100,000 ratings from 610 users  

---

## 6. Project Structure
/CineMatch <br>
├── data/ <br>
│ ├── movies_metadata.csv <br>
│ ├── credits.csv <br>
│ └── ratings_small.csv <br>
│ <br>
├── models/ <br>
│ └── (This folder will contain 5 .pkl files after the pipeline runs) <br>
│ <br>
├── hybrid_recommender.ipynb # Main notebook for ETL & training pipeline <br>
├── app.py # Streamlit application file <br>
├── requirements.txt # Python dependency <br>

## 7. Installation

[Download the dataset from Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv) <br>
Clone this repository:

```bash
git clone (https://github.com/nadhifaah/202510-hybrid-movie-recommendation.git)
cd 202510-hybrid-movie-recommendation
source venv/bin/activate
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py


