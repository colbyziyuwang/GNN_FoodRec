from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.abspath("../graphsage"))
from llm import generate_recommendations
from thompson_sampling import ThompsonSamplingBandit
from UCB import LinUCB, load_embedding_from_zip
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score, accuracy_score, precision_score, recall_score
import zipfile

app = Flask(__name__)

# Load dataset and embeddings
MERGED_FILE_PATH = "../graphsage/food-data/merged_recipes_interactions.csv"
EMBEDDINGS_ZIP_PATH = "../graphsage/graphsage_embeddings.zip"
data = pd.read_csv(MERGED_FILE_PATH)
zipf = zipfile.ZipFile(EMBEDDINGS_ZIP_PATH, "r")

# Mapping of IDs to embeddings
unique_user_ids = data["user_id"].unique()
unique_recipe_ids = data["recipe_id"].unique()
all_ids = list(unique_user_ids) + list(unique_recipe_ids)
node_id_mapping = {id_val: idx for idx, id_val in enumerate(all_ids)}

# Initialize LinUCB and Thompson Sampling
num_actions = 2  # binary: recommend or not recommend
feature_dim = 128 * 2  # user + recipe embedding size
alpha = 1.0  # Exploration parameter for LinUCB
linucb = LinUCB(num_actions, feature_dim, alpha)
thompson_bandit = ThompsonSamplingBandit(num_actions)

def load_embedding(node_id):
    embedding_filename = f"embedding_{node_id}.npy"
    try:
        with zipf.open(embedding_filename) as file:
            return np.load(file)
    except KeyError:
        return None
    
def preload_embeddings(unique_ids, zip_file):
    embedding_cache = {}
    for node_id in unique_ids:
        try:
            embedding_cache[node_id] = load_embedding(node_id)
        except KeyError:
            embedding_cache[node_id] = None  # Handle missing embeddings
    return embedding_cache

all_embeddings = preload_embeddings(all_ids, zipf)
top_k = 10

def load_cached_embedding(node_id):
    return all_embeddings.get(node_id)

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    API Endpoint: Recommend recipes based on user preferences.
    Input: JSON { "user_id": Optional[int], "preferences": str, "method": str }
    Output: JSON { "recommendations": [str], "description": str, "metrics": {...} }
    """
    user_req = request.get_json()
    user_id = user_req.get("user_id")
    preferences = user_req.get("preferences", "")
    method = user_req.get("method", "similarity") #default to llm if no other model provided

    if user_id:
        if user_id not in node_id_mapping:
            return jsonify({"error": "User not found"}), 404
        user_embedding = load_cached_embedding(node_id_mapping[user_id])
        if user_embedding is None:
            return jsonify({"error": "User embedding not found"}), 404
    else:
        user_embedding = np.random.rand(128)  #random embedding since it's necessary for calculation on non-llm cases

    if method == "similarity":
        recipe_embeddings = [load_cached_embedding(node_id_mapping[recipe_id]) for recipe_id in unique_recipe_ids]
        recipe_embeddings = np.array([emb for emb in recipe_embeddings if emb is not None])
        valid_recipe_ids = [recipe_id for recipe_id, emb in zip(unique_recipe_ids, recipe_embeddings) if emb is not None]

        user_embedding = user_embedding.reshape(1, -1)
        similarities = np.dot(recipe_embeddings, user_embedding.T).flatten() 

        top_k = 10
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_recipe_ids = [valid_recipe_ids[i] for i in top_indices]
        top_recipe_scores = [similarities[i] for i in top_indices]

        recommended_recipes = data[data["recipe_id"].isin(top_recipe_ids)]
        top_recipe_names = list(set(recommended_recipes["name"].tolist()))

        ground_truth_ratings = [
            recommended_recipes.loc[recommended_recipes["recipe_id"] == recipe_id, "rating"].values[0]
            for recipe_id in top_recipe_ids
            if recipe_id in recommended_recipes["recipe_id"].values
        ]
        relevance = [1 if rating >= 4 else 0 for rating in ground_truth_ratings]
        predicted_scores = top_recipe_scores[:len(relevance)]  

        if len(relevance) != len(predicted_scores):
            print("Debugging Mismatch:")
            print("Relevance Length:", len(relevance))
            print("Predicted Scores Length:", len(predicted_scores))

        metrics = {
            "ndcg": ndcg_score([relevance], [predicted_scores]) if relevance and predicted_scores else 0,
            "precision": precision_score(relevance, [1] * len(relevance), zero_division=0) if relevance else 0,
            "recall": recall_score(relevance, [1] * len(relevance), zero_division=0) if relevance else 0,
            "accuracy": accuracy_score(relevance, [1] * len(relevance)) if relevance else 0,
        }

        description = (
            generate_recommendations(preferences, top_recipe_names)
            if top_recipe_names
            else "We couldn't find any suitable recommendations based on similarity."
        )

        return jsonify({"recommendations": top_recipe_names, "description": description, "metrics": metrics})

    elif method == "thompson":
        top_k = 10
        recipe_embeddings = [
            load_cached_embedding(node_id_mapping[recipe_id]) for recipe_id in unique_recipe_ids[:top_k * 10]
        ]
        recipe_embeddings = np.array([emb for emb in recipe_embeddings if emb is not None])
        valid_recipe_ids = [recipe_id for recipe_id, emb in zip(unique_recipe_ids, recipe_embeddings) if emb is not None]

        predictions, ground_truth = [], []

        for idx, recipe_id in enumerate(valid_recipe_ids):
            predicted_action = thompson_bandit.select_action()
            target_action = 1 if data.loc[data["recipe_id"] == recipe_id, "rating"].mean() >= 4 else 0
            predictions.append(predicted_action)
            ground_truth.append(target_action)
            reward = 1 if predicted_action == target_action else 0
            thompson_bandit.update(predicted_action, reward)

        recommended_recipe_ids = [recipe_id for idx, recipe_id in enumerate(valid_recipe_ids) if predictions[idx] == 1]
        recommended_recipes = data[data["recipe_id"].isin(recommended_recipe_ids)]
        top_recipe_names = list(set(recommended_recipes["name"].tolist()))

        relevance = ground_truth[:len(recommended_recipe_ids)]
        predicted_scores = predictions[:len(recommended_recipe_ids)]

        if len(relevance) != len(predicted_scores):
            print("Debugging Mismatch (Thompson):")
            print("Relevance Length:", len(relevance))
            print("Predicted Scores Length:", len(predicted_scores))

        metrics = {
            "ndcg": ndcg_score([relevance], [predicted_scores]) if relevance and predicted_scores else 0,
            "precision": precision_score(relevance, [1] * len(relevance), zero_division=0) if relevance else 0,
            "recall": recall_score(relevance, [1] * len(relevance), zero_division=0) if relevance else 0,
            "accuracy": accuracy_score(relevance, [1] * len(relevance)) if relevance else 0,
        }

        description = (
            f"Thompson Sampling recommends: {', '.join(top_recipe_names[:5])}."
            if top_recipe_names
            else "Thompson Sampling couldn't find any suitable recommendations."
        )

        return jsonify({"recommendations": top_recipe_names, "description": description, "metrics": metrics})


    elif method == "linucb":
        top_k = 10
        recipe_embeddings = [
            load_cached_embedding(node_id_mapping[recipe_id]) for recipe_id in unique_recipe_ids[:top_k * 10]
        ]
        recipe_embeddings = np.array([emb for emb in recipe_embeddings if emb is not None])
        valid_recipe_ids = [recipe_id for recipe_id, emb in zip(unique_recipe_ids, recipe_embeddings) if emb is not None]

        predictions, ground_truth = [], []

        for idx, recipe_id in enumerate(valid_recipe_ids):
            state_vector = np.concatenate([user_embedding, recipe_embeddings[idx]])
            predicted_action = linucb.select_action(state_vector)
            target_action = 1 if data.loc[data["recipe_id"] == recipe_id, "rating"].mean() >= 4 else 0
            predictions.append(predicted_action)
            ground_truth.append(target_action)
            reward = 1 if predicted_action == target_action else 0
            linucb.update(predicted_action, state_vector, reward)

        recommended_recipe_ids = [recipe_id for idx, recipe_id in enumerate(valid_recipe_ids) if predictions[idx] == 1]
        recommended_recipes = data[data["recipe_id"].isin(recommended_recipe_ids)]
        top_recipe_names = list(set(recommended_recipes["name"].tolist()))

        relevance = ground_truth[:len(recommended_recipe_ids)]
        predicted_scores = predictions[:len(recommended_recipe_ids)]

        if len(relevance) != len(predicted_scores):
            print("Debugging Mismatch (LinUCB):")
            print("Relevance Length:", len(relevance))
            print("Predicted Scores Length:", len(predicted_scores))

        metrics = {
            "ndcg": ndcg_score([relevance], [predicted_scores]) if relevance and predicted_scores else 0,
            "precision": precision_score(relevance, [1] * len(relevance), zero_division=0) if relevance else 0,
            "recall": recall_score(relevance, [1] * len(relevance), zero_division=0) if relevance else 0,
            "accuracy": accuracy_score(relevance, [1] * len(relevance)) if relevance else 0,
        }

        description = (
            f"LinUCB recommends: {', '.join(top_recipe_names[:5])}."
            if top_recipe_names
            else "LinUCB couldn't find any suitable recommendations."
        )

        return jsonify({"recommendations": top_recipe_names, "description": description, "metrics": metrics})


    else:
        return jsonify({"error": "Invalid method specified"}), 400


if __name__ == "__main__":
    app.run(debug=True)
