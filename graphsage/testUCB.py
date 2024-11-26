# Test trained UCB model

import pickle
from UCB import LinUCB, load_embedding_from_zip
from proc_embed import get_top_k_ids
import pandas as pd
import zipfile
from tqdm import tqdm
import numpy as np
from metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k

# Load model
linucb = LinUCB(num_actions=2, feature_dim=128*2)
linucb = pickle.load('linucb_model.pkl')

# Get top-k ids
k = 50
top_k_ids = get_top_k_ids()

# Load interaction data
data = pd.read_csv('food-data/merged_recipes_interactions.csv')

# Get unique user and recipe IDs
unique_user_ids = data['user_id'].unique()
unique_recipe_ids = data['recipe_id'].unique()

# Create mappings for IDs
all_ids = list(unique_user_ids) + list(unique_recipe_ids)
node_id_mapping = {id_val: idx for idx, id_val in enumerate(all_ids)}

# Open the ZIP file containing embeddings
zip_path = "graphsage_embeddings.zip"
zipf = zipfile.ZipFile(zip_path, 'r')

# Get test dataset
num_train = int(len(data) * 0.80)  # Ensure it's an integer
sampled_data = data.sample(n=num_train, random_state=42)
num_test = int(len(data) * 0.20)
test_data = data.drop(sampled_data.index)  # Test data is the rest
test_data = test_data.sample(n=100, random_state=42)

# Obtain ranks
sorted_recipes = {}  # Dictionary to store sorted recipes for each user

for _, row in tqdm(test_data.iterrows(), desc="Testing LinUCB", total=len(test_data)):
    user_id = row['user_id']
    user_embedding = load_embedding_from_zip(zipf, node_id_mapping[user_id])

    top_k_recipes = top_k_ids[user_id]
    recipe_scores = []  # List to store (recipe_id, score)

    for recipe_id in tqdm(top_k_recipes, desc=f"Processing recipes"):
        recipe_embedding = load_embedding_from_zip(zipf, node_id_mapping[recipe_id])
        state_vector = np.concatenate([user_embedding, recipe_embedding])

        # Get action values
        action_values = linucb.select_action(state_vector, return_val=True)
        recipe_scores.append((recipe_id, action_values))

    # Sort recipes based on scores (from high to low)
    sorted_recipe_scores = sorted(recipe_scores, key=lambda x: x[1], reverse=True)
    sorted_recipes[user_id] = [recipe_id for recipe_id, _ in sorted_recipe_scores]

# Now `sorted_recipes` contains the recipe IDs sorted by score for each user

# Evaluate metrics
k = 10  # Set top-k recommendations to consider
precision_scores = []
recall_scores = []
ndcg_scores = []
hit_rate_scores = []

for _, row in tqdm(test_data.iterrows(), desc="Evaluating Metrics", total=len(test_data)):
    user_id = row['user_id']

    # Ground truth: Recipes rated 4 or 5 are considered relevant
    relevant_recipes = set(test_data[(test_data['user_id'] == user_id) & (test_data['rating'] >= 4)]['recipe_id'])

    # Skip users with no relevant items
    if len(relevant_recipes) == 0:
        continue

    # Get recommended recipes for the user
    recommended_recipes = sorted_recipes.get(user_id, [])

    # Calculate metrics for the user
    precision_scores.append(precision_at_k(recommended_recipes, relevant_recipes, k))
    recall_scores.append(recall_at_k(recommended_recipes, relevant_recipes, k))
    ndcg_scores.append(ndcg_at_k(recommended_recipes, relevant_recipes, k))
    hit_rate_scores.append(hit_rate_at_k(recommended_recipes, relevant_recipes, k))

# Compute averages
avg_precision = np.mean(precision_scores) if precision_scores else 0
avg_recall = np.mean(recall_scores) if recall_scores else 0
avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
avg_hit_rate = np.mean(hit_rate_scores) if hit_rate_scores else 0

# Output results
print(f"Average Precision@{k}: {avg_precision:.4f}")
print(f"Average Recall@{k}: {avg_recall:.4f}")
print(f"Average NDCG@{k}: {avg_ndcg:.4f}")
print(f"Average Hit Rate@{k}: {avg_hit_rate:.4f}")
