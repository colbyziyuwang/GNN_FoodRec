import numpy as np
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import inv
import pandas as pd
import pickle
from tqdm import tqdm
class LinUCB:
    """
    LinUCB algorithm for contextual multi-armed bandits.

    Attributes:
    - num_actions: Number of possible actions (e.g., ratings 0-5).
    - feature_dim: Dimension of the feature vector for each state/action.
    - alpha: Exploration-exploitation parameter.
    - A: List of covariance matrices for each action.
    - b: List of bias vectors for each action.
    """
    def __init__(self, num_actions, feature_dim, alpha=1.0):
        self.num_actions = num_actions
        self.alpha = alpha
        self.A = [csc_matrix(identity(feature_dim)) for _ in range(num_actions)]
        self.b = [np.zeros((feature_dim, 1)) for _ in range(num_actions)]

    def select_action(self, state_vector):
        """
        Selects an action based on the LinUCB algorithm.

        Parameters:
        - state_vector: The feature vector for the current state.

        Returns:
        - The index of the chosen action.
        """
        state_vector = state_vector.reshape(-1, 1)  # Ensure column vector
        p_values = []

        for a in range(self.num_actions):
            A_inv = inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            # Compute the upper confidence bound (UCB)
            p = (state_vector.T @ theta_hat)[0, 0] + self.alpha * np.sqrt((state_vector.T @ A_inv @ state_vector)[0, 0])
            p_values.append(p)
        
        return np.argmax(p_values)

    def update(self, action, state_vector, reward):
        """
        Updates the model parameters after receiving a reward.

        Parameters:
        - action: The index of the chosen action.
        - state_vector: The feature vector for the current state.
        - reward: The observed reward for the chosen action.
        """
        state_vector = state_vector.reshape(-1, 1)
        self.A[action] += state_vector @ state_vector.T
        self.b[action] += reward * state_vector

def test_model(linucb, user_embedding, recipe_embedding):
    """
    Test the LinUCB model with a given user and recipe.

    Parameters:
    - linucb: Trained LinUCB model.
    - user_embedding: Embedding for the user.
    - recipe_embedding: Embedding for the recipe.

    Returns:
    - Predicted action (rating).
    """
    state_vector = np.concatenate([user_embedding, recipe_embedding])
    return linucb.select_action(state_vector)

# Load interaction data
data = pd.read_csv('food-data/interactions_train.csv')

# Get unique user and recipe IDs
unique_user_ids = data['u'].unique()
unique_recipe_ids = data['recipe_id'].unique()

# Create mappings for IDs
all_ids = list(unique_user_ids) + list(unique_recipe_ids)
node_id_mapping = {id_val: idx for idx, id_val in enumerate(all_ids)}

# Load embeddings using memmap
embedding_array = np.memmap('graphsage_embeddings.npy', dtype='float32', mode='r', shape=(len(all_ids), 512))

# Initialize LinUCB
num_actions = 6  # Ratings 0 to 5
feature_dim = 512 * 2  # Concatenated user and recipe embeddings
alpha = 1.0  # Exploration parameter
linucb = LinUCB(num_actions, feature_dim, alpha)

# Sample 10,000 random rows from the dataset
sampled_data = data.sample(n=10000, random_state=42)

# Training Loop
for _, row in tqdm(sampled_data.iterrows(), desc="Training LinUCB", total=len(sampled_data)):
    user_id = row['u']
    recipe_id = row['recipe_id']
    rating = int(row['rating'])  # Reward (integer between 0 and 5)

    # Get embeddings
    user_embedding = embedding_array[node_id_mapping[user_id]]
    recipe_embedding = embedding_array[node_id_mapping[recipe_id]]
    state_vector = np.concatenate([user_embedding, recipe_embedding])

    # Select an action
    chosen_action = linucb.select_action(state_vector)

    # Observe reward
    observed_reward = 1 if chosen_action == rating else 0  # Binary reward

    # Update LinUCB model
    linucb.update(chosen_action, state_vector, observed_reward)

# Save the LinUCB model
with open('linucb_model.pkl', 'wb') as f:
    pickle.dump(linucb, f)
