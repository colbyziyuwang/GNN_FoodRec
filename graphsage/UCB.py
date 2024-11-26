import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import zipfile

class LinUCB:
    """
    LinUCB algorithm for contextual multi-armed bandits.
    """
    def __init__(self, num_actions, feature_dim, alpha=1.0):
        self.num_actions = num_actions
        self.alpha = alpha
        # Initialize dense matrices for covariance
        self.A = [np.identity(feature_dim) for _ in range(num_actions)]
        self.b = [np.zeros((feature_dim, 1)) for _ in range(num_actions)]

    def select_action(self, state_vector, return_val=False):
        """
        Selects an action based on the LinUCB algorithm.
        """
        state_vector = state_vector.reshape(-1, 1)  # Ensure column vector
        p_values = []

        for a in range(self.num_actions):
            A_inv = np.linalg.inv(self.A[a])  # Inverse of A[a]
            theta_hat = A_inv @ self.b[a]  # Estimate of theta
            # Compute the upper confidence bound (UCB)
            p = (state_vector.T @ theta_hat)[0, 0] + self.alpha * np.sqrt((state_vector.T @ A_inv @ state_vector)[0, 0])
            p_values.append(p)

        if (return_val == True): # return array of p_values
            return p_values
        
        return np.argmax(p_values)

    def update(self, action, state_vector, reward):
        """
        Updates the model parameters after receiving a reward.
        """
        state_vector = state_vector.reshape(-1, 1)
        self.A[action] += state_vector @ state_vector.T  # Update covariance matrix
        self.b[action] += reward * state_vector  # Update bias vector

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

# Initialize LinUCB
num_actions = 2  # Recommend or not recommend (0 means not recommend)
feature_dim = 128 * 2  # Concatenated user and recipe embeddings
alpha = 1.0  # Exploration parameter
linucb = LinUCB(num_actions, feature_dim, alpha)

# Sample 80% random rows from the dataset
num_train = int(len(data) * 0.80)  # Ensure it's an integer
sampled_data = data.sample(n=num_train, random_state=42)

# Training Loop
for _, row in tqdm(sampled_data.iterrows(), desc="Training LinUCB", total=len(sampled_data)):
    user_id = row['user_id']
    recipe_id = row['recipe_id']
    rating = row['rating']

    # Validate and preprocess rating
    if not (0 <= rating <= 5):
        raise ValueError(f"Unexpected rating value: {rating}")
    rating = int(rating)  # Ensure integer type

    # Get embeddings
    user_embedding = load_embedding_from_zip(zipf, node_id_mapping[user_id])
    recipe_embedding = load_embedding_from_zip(zipf, node_id_mapping[recipe_id])
    state_vector = np.concatenate([user_embedding, recipe_embedding])

    # Select an action
    chosen_action = linucb.select_action(state_vector)

    # Define target action
    target_action = 1 if rating >= 4 else 0  # Recommend if rating >= 4

    # Observe reward
    observed_reward = 1 if chosen_action == target_action else 0  # Binary reward

    # Update LinUCB model
    linucb.update(chosen_action, state_vector, observed_reward)

# Close the ZIP file
zipf.close()

# Save the LinUCB model
with open('linucb_model.pkl', 'wb') as f:
    pickle.dump(linucb, f)
