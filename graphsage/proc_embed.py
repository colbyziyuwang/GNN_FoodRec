import pandas as pd
import numpy as np
from tqdm import tqdm
import zipfile

# Function to retrieve the original ID from a node ID
def get_original_id(node_id, node_id_reverse_mapping):
    return node_id_reverse_mapping.get(node_id)

def load_embedding_from_zip(zipf, node_id):
    """
    Load an embedding for a specific node ID from the ZIP file.
    """
    embedding_filename = f"embedding_{node_id}.npy"
    try:
        with zipf.open(embedding_filename) as file:
            return np.load(file)
    except KeyError:
        raise ValueError(f"Embedding for node ID {node_id} not found in ZIP file.")

def get_top_k_ids(zip_filepath='graphsage_embeddings.zip', k=50, chunk_size=5000):
    """
    Computes the top-k similar recipes for each user based on the dot product similarity 
    of their embeddings, processing users in chunks to reduce memory usage.

    Parameters:
    - zip_filepath: Path to the ZIP file containing embeddings.
    - k: Number of top similar recipes to retrieve for each user.
    - chunk_size: Number of users to process at a time.

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

    top_k_idx = {}

    with zipfile.ZipFile(zip_filepath, 'r') as zipf:
        # Load all recipe embeddings into memory
        recipe_embeddings = [load_embedding_from_zip(zipf, recipe_idx + num_users) for recipe_idx in tqdm(range(len(unique_recipe_ids)), desc="Loading recipes")]
        recipe_matrix = np.array(recipe_embeddings)

        # Process users in chunks
        for start_idx in tqdm(range(0, num_users, chunk_size), desc="Processing users in chunks"):
            end_idx = min(start_idx + chunk_size, num_users)
            user_embeddings = [load_embedding_from_zip(zipf, user_idx) for user_idx in range(start_idx, end_idx)]
            user_matrix = np.array(user_embeddings)

            # Compute similarity matrix for the current chunk
            sim_matrix = np.matmul(user_matrix, recipe_matrix.T)

            # Find top-k indices for each user in the chunk
            for user_chunk_idx, user_idx in enumerate(range(start_idx, end_idx)):
                user_sim = sim_matrix[user_chunk_idx]
                top_indices = np.argpartition(-user_sim, k)[:k]  # Get top-k indices (unsorted)
                sorted_indices = top_indices[np.argsort(-user_sim[top_indices])]  # Sort top-k indices
                top_k_idx[get_original_id(user_idx, node_id_reverse_mapping)] = [
                    get_original_id(idx + num_users, node_id_reverse_mapping) for idx in sorted_indices
                ]

    return top_k_idx
