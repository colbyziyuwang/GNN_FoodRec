from flask import Flask, request, jsonify, render_template_string
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

@app.route("/")
def home():
    """
    Render a simple HTML form for user input.
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recipe Recommendation</title>
    </head>
    <body>
        <h1>Recipe Recommendation System</h1>
        <form action="/recommend" method="post">
            <label for="user_id">User ID (optional):</label><br>
            <input type="text" id="user_id" name="user_id"><br><br>
            
            <label for="preferences">Preferences:</label><br>
            <textarea id="preferences" name="preferences" rows="4" cols="50" placeholder="E.g., Vegan, likes spicy food"></textarea><br><br>
            
            <label for="method">Recommendation Method:</label><br>
            <select id="method" name="method">
                <option value="similarity">Similarity (LLM)</option>
                <option value="thompson">Thompson Sampling</option>
                <option value="linucb">LinUCB</option>
            </select><br><br>
            
            <button type="submit">Get Recommendations</button>
        </form>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Handle recommendation logic and return HTML with results.
    """
    user_id = request.form.get("user_id", type=int)
    preferences = request.form.get("preferences", "")
    method = request.form.get("method", "similarity")

    # Fetch past recipes and ratings for the user
    past_recipes_html = ""
    if user_id:
        if user_id not in node_id_mapping:
            return "<p>Error: User not found.</p>", 404

        user_embedding = load_embedding(node_id_mapping[user_id])
        if user_embedding is None:
            return "<p>Error: User embedding not found.</p>", 404

        # Fetch the user's past recipes and ratings
        user_history = data[data["user_id"] == user_id][["name", "rating"]]
        if not user_history.empty:
            past_recipes_html = "<h3>Past Recipes and Ratings:</h3><ul>"
            for _, row in user_history.iterrows():
                past_recipes_html += f"<li>{row['name']} - Rating: {row['rating']}</li>"
            past_recipes_html += "</ul>"
        else:
            past_recipes_html = "<p>No past recipes found for this user.</p>"
    else:
        user_embedding = np.random.rand(128)  # Random embedding for non-LLM methods
        past_recipes_html = "<p>User ID not provided. No past recipes to show.</p>"

    if method == "similarity":
        # Step 1: Retrieve top 10 recipes by similarity
        recipe_embeddings = [
            load_embedding(node_id_mapping[recipe_id]) for recipe_id in unique_recipe_ids
        ]
        recipe_embeddings = np.array([emb for emb in recipe_embeddings if emb is not None])
        valid_recipe_ids = [recipe_id for recipe_id, emb in zip(unique_recipe_ids, recipe_embeddings) if emb is not None]

        user_embedding = user_embedding.reshape(1, -1)
        similarities = np.dot(recipe_embeddings, user_embedding.T).flatten()

        # Sort by similarity and take the top 10
        top_10_indices = np.argsort(similarities)[-10:][::-1]
        top_10_recipe_ids = [valid_recipe_ids[i] for i in top_10_indices]

        # Ensure distinct recipe names
        recommended_recipes = data[data["recipe_id"].isin(top_10_recipe_ids)]
        top_10_recipes = list(set(recommended_recipes["name"].tolist()))

        # Step 2: Use LLM to select the best recipe with justification
        if top_10_recipes:
            llm_response = generate_recommendations(preferences, top_10_recipes)
        else:
            llm_response = "No suitable recipes found."

        # Step 3: Metrics (optional, calculated for top-10 similarity-based recommendations)
        ground_truth_ratings = [
            recommended_recipes.loc[recommended_recipes["recipe_id"] == recipe_id, "rating"].values[0]
            for recipe_id in top_10_recipe_ids
            if recipe_id in recommended_recipes["recipe_id"].values
        ]
        relevance = [1 if rating >= 4 else 0 for rating in ground_truth_ratings]
        predicted_scores = [similarities[i] for i in top_10_indices[:len(relevance)]]  # Match lengths

        metrics = {
            "ndcg": ndcg_score([relevance], [predicted_scores]) if relevance and predicted_scores else 0,
            "precision": precision_score(relevance, [1] * len(relevance), zero_division=0) if relevance else 0,
            "recall": recall_score(relevance, [1] * len(relevance), zero_division=0) if relevance else 0,
            "accuracy": accuracy_score(relevance, [1] * len(relevance)) if relevance else 0,
        }

        return f"""
        <!DOCTYPE html>
        <html>
        <body>
            <h2>Similarity (LLM) Recommendations</h2>
            <p><strong>LLM Recommendation:</strong> {llm_response}</p>
            <p><strong>Top 10 Recipes:</strong> {", ".join(top_10_recipes)}</p>
            {past_recipes_html}
        </body>
        </html>
        """

    elif method == "thompson":
        # Thompson Sampling-based recommendation
        top_k = 10
        predictions, ground_truth = [], []

        for recipe_id in unique_recipe_ids[:top_k * 10]:
            recipe_embedding = load_embedding(node_id_mapping[recipe_id])
            if recipe_embedding is not None:
                predicted_action = thompson_bandit.select_action()
                target_action = 1 if data.loc[data["recipe_id"] == recipe_id, "rating"].mean() >= 4 else 0
                predictions.append(predicted_action)
                ground_truth.append(target_action)
                reward = 1 if predicted_action == target_action else 0
                thompson_bandit.update(predicted_action, reward)

        recommended_recipe_ids = [
            recipe_id for idx, recipe_id in enumerate(unique_recipe_ids[:top_k * 10]) if predictions[idx] == 1
        ]
        recommended_recipes = data[data["recipe_id"].isin(recommended_recipe_ids)]
        top_recipe_names = list(set(recommended_recipes["name"].tolist()))

        relevance = ground_truth[:len(recommended_recipe_ids)]
        predicted_scores = predictions[:len(recommended_recipe_ids)]

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

        return f"""
        <!DOCTYPE html>
        <html>
        <body>
            <h2>Thompson Sampling Recommendations</h2>
            <p><strong>Recommendations:</strong> {", ".join(top_recipe_names[:5])}</p>
            {past_recipes_html}
        </body>
        </html>
        """

    elif method == "linucb":
        # LinUCB-based recommendation
        top_k = 10
        predictions, ground_truth = [], []

        for recipe_id in unique_recipe_ids[:top_k * 10]:
            recipe_embedding = load_embedding(node_id_mapping[recipe_id])
            if recipe_embedding is not None:
                state_vector = np.concatenate([user_embedding, recipe_embedding])
                predicted_action = linucb.select_action(state_vector)
                target_action = 1 if data.loc[data["recipe_id"] == recipe_id, "rating"].mean() >= 4 else 0
                predictions.append(predicted_action)
                ground_truth.append(target_action)
                reward = 1 if predicted_action == target_action else 0
                linucb.update(predicted_action, state_vector, reward)

        recommended_recipe_ids = [
            recipe_id for idx, recipe_id in enumerate(unique_recipe_ids[:top_k * 10]) if predictions[idx] == 1
        ]
        recommended_recipes = data[data["recipe_id"].isin(recommended_recipe_ids)]
        top_recipe_names = list(set(recommended_recipes["name"].tolist()))

        relevance = ground_truth[:len(recommended_recipe_ids)]
        predicted_scores = predictions[:len(recommended_recipe_ids)]

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

        return f"""
        <!DOCTYPE html>
        <html>
        <body>
            <h2>LinUCB Recommendations</h2>
            <p><strong>Recommendations:</strong> {", ".join(top_recipe_names[:5])}</p>
            {past_recipes_html}
        </body>
        </html>
        """

    else:
        return "<p>Error: Invalid method specified.</p>", 400


if __name__ == "__main__":
    app.run(debug=True)
