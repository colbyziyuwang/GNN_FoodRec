import pandas as pd
import numpy as np
from tqdm import tqdm

# Function to retrieve the original ID from a node ID
def get_original_id(node_id, node_id_reverse_mapping):
    return node_id_reverse_mapping.get(node_id)

def get_top_k_ids(k=50):
    """
    Computes the top-k similar recipes for each user based on the dot product similarity 
    of their embeddings.

    Parameters:
    - k: Number of top similar recipes to retrieve for each user.

    Returns:
    - top_k_idx: A dictionary mapping each user ID to the indices of their top-k recipes.
    """
    # Load interaction data
    data = pd.read_csv('food-data/merged_recipes_interactions.csv')

    # Get unique user and recipe IDs
    unique_user_ids = data['user_id'].unique()
    unique_recipe_ids = data['recipe_id'].unique()

    # Create combined unique IDs and mappings
    num_users = len(unique_user_ids)
    all_ids = list(unique_user_ids) + list(unique_recipe_ids)
    node_id_reverse_mapping = {idx: id_val for idx, id_val in enumerate(all_ids)}

    # Load precomputed embeddings
    embedding = np.memmap('graphsage_embeddings.npy', dtype='float32', mode='r', shape=(len(all_ids), 512))

    # Separate user and recipe embeddings
    user_matrix = embedding[:num_users, :]  # User embeddings
    recipe_matrix = embedding[num_users:, :]  # Recipe embeddings

    # Compute similarity matrix [num_users x num_recipes]
    sim_matrix = np.matmul(user_matrix, recipe_matrix.T)

    # Find top-k indices for each user
    # np.argpartition finds the k-largest indices efficiently
    # Sorting those k indices ensures descending order of similarity
    top_k_idx = {}
    for user_idx in tqdm(range(sim_matrix.shape[0]), desc="Finding top-k recipes"):
        user_sim = sim_matrix[user_idx]
        top_indices = np.argpartition(-user_sim, k)[:k]  # Get top-k indices (unsorted)
        sorted_indices = top_indices[np.argsort(-user_sim[top_indices])]  # Sort top-k indices
        top_k_idx[get_original_id(user_idx, node_id_reverse_mapping)] = [
            get_original_id(idx + num_users, node_id_reverse_mapping) for idx in sorted_indices
        ]

    return top_k_idx

