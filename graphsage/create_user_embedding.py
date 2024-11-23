import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load the interactions dataset
interact = pd.read_csv("food-data/merged_recipes_interactions.csv")

# Load the recipe embeddings from the pickle file
with open('food-data/recipe_embeddings.pkl', 'rb') as f:
    recipe_embeddings = pickle.load(f)

# Create user embeddings using weighted averaging
user_embeddings = {}

for user_id in tqdm(interact['user_id'].unique(), desc="Creating User Embeddings"):
    # Get all interactions for the user
    user_data = interact[interact['user_id'] == user_id]

    # Get recipe IDs and corresponding ratings
    user_recipes = user_data['recipe_id']
    user_ratings = user_data['rating'] + 1

    # Fetch the embeddings for these recipes
    recipe_embs = [
        recipe_embeddings[recipe_id]
        for recipe_id in user_recipes
        if recipe_id in recipe_embeddings
    ]

    # Use ratings as weights
    if recipe_embs:
        recipe_embs = np.array(recipe_embs)
        ratings = user_ratings.values

        # Normalize ratings to sum to 1 for weighted averaging
        weights = ratings / ratings.sum()
        user_embedding = np.average(recipe_embs, axis=0, weights=weights)
        user_embeddings[user_id] = user_embedding

# Save user embeddings to a pickle file
with open("food-data/user_embeddings.pkl", "wb") as f:
    pickle.dump(user_embeddings, f)

print("User embeddings created and saved successfully.")
