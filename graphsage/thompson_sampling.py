import numpy as np
import pandas as pd
from tqdm import tqdm
import zipfile
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ThompsonSamplingBandit:
    """
    Thompson Sampling algorithm for contextual multi-armed bandits.
    """
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.successes = np.zeros(num_actions)
        self.failures = np.zeros(num_actions)

    def select_action(self):
        """
        Select an action based on Thompson Sampling.

        :return: Index of the selected action.
        """
        sampled_means = [
            np.random.beta(self.successes[a] + 1, self.failures[a] + 1)
            for a in range(self.num_actions)
        ]
        return np.argmax(sampled_means)

    def update(self, action, reward):
        """
        Update the success and failure counts based on observed reward.

        :param action: Index of the selected action.
        :param reward: Observed reward (1 for success, 0 for failure).
        """
        if reward > 0:
            self.successes[action] += 1
        else:
            self.failures[action] += 1


def load_embedding_from_zip(zipf, node_id):
    """
    Load an embedding for a specific node ID from the ZIP file.

    :param zipf: Opened ZIP file containing embeddings.
    :param node_id: Node ID to retrieve embedding for.
    :return: Numpy array of the embedding.
    """
    embedding_filename = f"embedding_{node_id}.npy"
    try:
        with zipf.open(embedding_filename) as file:
            return np.load(file)
    except KeyError:
        raise ValueError(f"Embedding for node ID {node_id} not found in ZIP file.")


def train_thompson_sampling(ts_bandit, train_data, zipf, node_id_mapping):
    """
    Train the Thompson Sampling bandit model.

    :param ts_bandit: Thompson Sampling bandit instance.
    :param train_data: Training data (DataFrame).
    :param zipf: Opened ZIP file containing embeddings.
    :param node_id_mapping: Mapping of IDs to embedding indices.
    """
    for _, row in tqdm(train_data.iterrows(), desc="Training Thompson Sampling", total=len(train_data)):
        user_id = row['user_id']
        recipe_id = row['recipe_id']
        rating = row['rating']

        target_action = 1 if rating >= 4 else 0

        user_embedding = load_embedding_from_zip(zipf, node_id_mapping[user_id])
        recipe_embedding = load_embedding_from_zip(zipf, node_id_mapping[recipe_id])
        state_vector = np.concatenate([user_embedding, recipe_embedding])

        predicted_action = ts_bandit.select_action()
        reward = 1 if predicted_action == target_action else 0

        ts_bandit.update(predicted_action, reward)


def evaluate_thompson_sampling(ts_bandit, test_data, zipf, node_id_mapping):
    """
    Evaluate the Thompson Sampling bandit model.

    :param ts_bandit: Thompson Sampling bandit instance.
    :param test_data: Test data (DataFrame).
    :param zipf: Opened ZIP file containing embeddings.
    :param node_id_mapping: Mapping of IDs to embedding indices.
    :return: Dictionary with accuracy, precision, and recall scores.
    """
    predictions = []
    ground_truth = []

    for _, row in tqdm(test_data.iterrows(), desc="Evaluating Thompson Sampling", total=len(test_data)):
        user_id = row['user_id']
        recipe_id = row['recipe_id']
        rating = row['rating']

        target_action = 1 if rating >= 4 else 0

        user_embedding = load_embedding_from_zip(zipf, node_id_mapping[user_id])
        recipe_embedding = load_embedding_from_zip(zipf, node_id_mapping[recipe_id])
        state_vector = np.concatenate([user_embedding, recipe_embedding])

        predicted_action = ts_bandit.select_action()

        predictions.append(predicted_action)
        ground_truth.append(target_action)

    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }


if __name__ == "__main__":
    MERGED_FILE_PATH = "food-data/merged_recipes_interactions.csv"
    EMBEDDINGS_ZIP_PATH = "graphsage_embeddings.zip"

    data = pd.read_csv(MERGED_FILE_PATH)
    unique_user_ids = data['user_id'].unique()
    unique_recipe_ids = data['recipe_id'].unique()
    all_ids = list(unique_user_ids) + list(unique_recipe_ids)
    node_id_mapping = {id_val: idx for idx, id_val in enumerate(all_ids)}

    zipf = zipfile.ZipFile(EMBEDDINGS_ZIP_PATH, 'r')
    train_data = data.sample(frac=0.8, random_state=42) 
    test_data = data.drop(train_data.index)

    ts_bandit = ThompsonSamplingBandit(2)

    train_thompson_sampling(ts_bandit, train_data, zipf, node_id_mapping)
    metrics = evaluate_thompson_sampling(ts_bandit, test_data, zipf, node_id_mapping)

    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

    zipf.close()
