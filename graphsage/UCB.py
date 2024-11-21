from proc_embed import get_top_k_ids
import pandas as pd
import numpy as np
import pickle

from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import inv
from tqdm import tqdm

class LinUCB:
    """
    LinUCB algorithm for contextual multi-armed bandits using sparse matrices.

    Attributes:
    - num_actions: Number of possible actions.
    - feature_dim: Dimension of the feature vector for each state/action.
    - alpha: Exploration-exploitation parameter.
    - A: List of sparse covariance matrices for each action.
    - b: List of sparse bias vectors for each action.
    """
    def __init__(self, num_actions, feature_dim, alpha=1.0):
        self.num_actions = num_actions
        self.alpha = alpha
        # Initialize sparse covariance matrices and bias vectors
        self.A = {action: csc_matrix(identity(feature_dim)) for action in range(num_actions)}
        self.b = {action: csc_matrix(np.zeros((feature_dim, 1))) for action in range(num_actions)}
    
    def select_action(self, state_idx, state_vector, top_k_indices, node_id_mapping, num_users):
        """
        Selects an action based on the LinUCB algorithm.

        Parameters:
        - state_idx: The index of the current state (user).
        - state_vector: The feature vector for the current state.
        - top_k_indices: Precomputed top-k indices for the current state.
        - node_id_mapping: Mapping of original IDs to internal indices.

        Returns:
        - The index of the chosen action.
        """
        p_values = []
        state_vector = csc_matrix(state_vector).reshape(-1, 1)  # Ensure column vector as sparse matrix

        for action_id in top_k_indices[state_idx]:
            action_idx = node_id_mapping[action_id] - num_users  # Convert action ID to internal index
            A_inv = inv(self.A[action_idx])
            theta_hat = A_inv @ self.b[action_idx]
            # Compute the upper confidence bound (UCB)
            p = (state_vector.T @ theta_hat).toarray()[0, 0] + \
                self.alpha * np.sqrt((state_vector.T @ A_inv @ state_vector).toarray()[0, 0])
            p_values.append(p)

        # Return the index of the action with the highest UCB
        best_action_idx = top_k_indices[state_idx][np.argmax(p_values)]
        return node_id_mapping.get(best_action_idx) - num_users  # Return the recipe_index

    def update(self, action_idx, state_vector, reward):
        """
        Updates the model parameters after receiving a reward.

        Parameters:
        - action_idx: The internal index of the chosen action.
        - state_vector: The feature vector for the current state.
        - reward: The observed reward for the chosen action.
        """
        state_vector = csc_matrix(state_vector).reshape(-1, 1)
        self.A[action_idx] += state_vector @ state_vector.T
        self.b[action_idx] += reward * state_vector


def get_node_id(user_or_recipe_id, node_id_mapping):
    """
    Retrieve the internal node ID for a given user or recipe ID.
    """
    return node_id_mapping.get(user_or_recipe_id)

# Load interaction data
data = pd.read_csv('food-data/interactions_train.csv')

# Get unique user and recipe IDs
unique_user_ids = data['u'].unique()
num_users = len(unique_user_ids)
unique_recipe_ids = data['recipe_id'].unique()

# Create mappings for IDs
all_ids = list(unique_user_ids) + list(unique_recipe_ids)
node_id_mapping = {id_val: idx for idx, id_val in enumerate(all_ids)}
node_id_reverse_mapping = {idx: id_val for id_val, idx in node_id_mapping.items()}

# Load top-k indices
top_k_idx = get_top_k_ids()

# Load embeddings using memmap
embedding_array = np.memmap('graphsage_embeddings.npy', dtype='float32', mode='r', shape=(len(all_ids), 512))

# Initialize LinUCB
num_actions = len(unique_recipe_ids)
feature_dim = 512  # Dimensionality of embeddings
alpha = 1.0  # Exploration parameter
linucb = LinUCB(num_actions, feature_dim, alpha)

# Sample 10,000 random rows from the dataset
sampled_data = data.sample(n=10000, random_state=42)

# Training Loop
for _, row in tqdm(sampled_data.iterrows(), desc="Training LinUCB", total=len(sampled_data)):
    user_id = row['u']
    recipe_id = row['recipe_id']
    reward = row['rating']

    # Get user and recipe indices
    user_idx = get_node_id(user_id, node_id_mapping)
    recipe_idx = get_node_id(recipe_id, node_id_mapping)

    # Get embeddings
    user_embedding = embedding_array[user_idx]
    # recipe_embedding = embedding_array[recipe_idx]

    # Select an action (from top-k indices for the current user)
    chosen_action_idx = linucb.select_action(user_id, user_embedding, top_k_idx, node_id_mapping, num_users)

    # Update the model with observed reward
    linucb.update(chosen_action_idx, user_embedding, reward)

# Save the LinUCB model
with open('linucb_model.pkl', 'wb') as f:
    pickle.dump(linucb, f)
