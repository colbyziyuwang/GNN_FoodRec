import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load the interactions dataset
interact = pd.read_csv("food-data/merged_recipes_interactions.csv")

# Load the recipe embeddings from the pickle file
with open('food-data/recipe_embeddings.pkl', 'rb') as f:
    recipe_embeddings = pickle.load(f)

# Create user embeddings by averaging recipe embeddings
user_embeddings = {}

for user_id in tqdm(interact['user_id'].unique(), desc="Creating User Embeddings"):
    # Get all recipe IDs rated by the user
    user_recipes = interact[interact['user_id'] == user_id]['recipe_id']

    # Fetch the embeddings for these recipes
    recipe_embs = [recipe_embeddings[recipe_id] for recipe_id in user_recipes if recipe_id in recipe_embeddings]

    # Aggregate embeddings (e.g., mean pooling)
    if recipe_embs:
        recipe_embs = np.array(recipe_embs)
        user_embedding = recipe_embs.mean(axis=0)  # Mean pooling across all recipe embeddings
        user_embeddings[user_id] = user_embedding

# Save user embeddings to a pickle file
with open("food-data/user_embeddings.pkl", "wb") as f:
    pickle.dump(user_embeddings, f)

print("User embeddings created and saved successfully.")
