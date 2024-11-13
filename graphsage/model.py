import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

from tqdm import tqdm

import pandas as pd
from transformers import pipeline
import numpy as np
import pickle
import pandas as pd

"""
Simple supervised GraphSAGE model to learn useful node embeddings for user
and recipes
"""

class SupervisedGraphSage(nn.Module):
    def __init__(self, enc, user_project_layer, recipe_project_layer, 
                 user_dim=25076, recipe_dim=768, common_dim=512):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc

        self.user_dim = user_dim
        self.recipe_dim = recipe_dim
        self.common_dim = common_dim

        # Projection layers
        self.user_project = user_project_layer  # projects user embedding to common_dim
        self.recipe_project = recipe_project_layer  # projects recipe embedding to common_dim

        # Loss Parameters
        self.alpha_n=1.0
        self.alpha_l=0.5
        self.delta_n=1.0
        self.delta_l=0.5

    def max_margin_loss(self, user_embedding, pos_embedding, neg_embedding, low_rank_embedding):
        """
        Parameters:
        - user_embedding: User embedding.
        - pos_embedding: Embedding of high-rated (positive) recipes (4-5).
        - neg_embedding: Embedding of non-interacted (negative) recipes.
        - low_rank_embedding: Embedding of low-rated (low-rank positive) recipes (1-3).
        - alpha_n: Weight for negative term in loss.
        - alpha_l: Weight for low-rank positive term in loss.
        - delta_n: Margin for negative term.
        - delta_l: Margin for low-rank positive term.

        Returns:
        - Max-margin loss with low-rank positive augmentation.
        """
        # Compute positive interaction scores
        pos_score = torch.sum(user_embedding * pos_embedding, dim=1)

        # Compute negative interaction scores
        neg_score = torch.sum(user_embedding * neg_embedding, dim=1)

        # Compute low-rank positive interaction scores
        low_rank_score = torch.sum(user_embedding * low_rank_embedding, dim=1)

        # Max-margin loss for negatives
        neg_loss = F.relu(-pos_score + neg_score + self.delta_n)

        # Max-margin loss for low-rank positives
        low_rank_loss = F.relu(-pos_score + low_rank_score + self.delta_l)

        # Total loss with weighting
        loss = self.alpha_n * neg_loss + self.alpha_l * low_rank_loss

        return loss.mean()

def create_food_embedding():
    dataset_path = "food-data/"

    # Load RAW_recipes.csv
    raw_recipes = pd.read_csv(dataset_path + "RAW_recipes.csv")
    print(raw_recipes.head())

    # Get recipe embeddings and save to a dict
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    embedding_dict = {}
    embedding = pipeline('feature-extraction', model='alexdseo/RecipeBERT', framework='pt', device=device)
    for i in tqdm(range(len(raw_recipes)), desc="bert"):
        food_data = str(raw_recipes['name'][i])
        food_rep = np.mean(embedding(food_data)[0], axis=0)
        embedding_dict[raw_recipes['id'][i]] = food_rep
        
    with open('food-data/recipe_embeddings.pkl', 'wb') as f:
        pickle.dump(embedding_dict, f)

    print("Embeddings saved successfully.")

def load_embedding(filepath):
    # Load the dictionary
    with open(filepath, 'rb') as f:
        embedding_dict = pickle.load(f)
    return embedding_dict

def retrieve_embedding(embedding_dict, id):
    # Retrieve embedding for a specific ID
    embedding = embedding_dict[id]
    
    return embedding

def create_user_embedding():
    # Set dataset path
    dataset_path = "food-data/"

    # Load and analyze interactions_train.csv
    interactions_train = pd.read_csv(dataset_path + 'interactions_train.csv')

    # Get the maximum user ID
    max_user_id = max(interactions_train['u'].unique())

    # Create User Embedding (one-hot encoding)
    embedding_dict = {}
    for i in tqdm(range(max_user_id + 1), desc="user"):
        user_embedding = np.zeros(max_user_id + 1)
        user_embedding[i] = 1
        embedding_dict[i] = user_embedding
        
    # Save the embedding dictionary to a file
    with open(dataset_path + 'user_embedding.pkl', 'wb') as f:
        pickle.dump(embedding_dict, f)

    print("Embeddings saved successfully.")

# Function to retrieve node ID from the mapping
def get_node_id(user_or_recipe_id, node_id_mapping):
    return node_id_mapping.get(user_or_recipe_id)

# Define function to load food embeddings with projection layers
def load_food(recipe_embedding_dict, user_embedding_dict):    
    # Load data
    data = pd.read_csv('food-data/interactions_train.csv')

    # Get unique user and recipe IDs
    unique_user_ids = data['u'].unique()
    unique_recipe_ids = data['recipe_id'].unique()

    # Create combined unique IDs and node ID mapping
    all_ids = list(unique_user_ids) + list(unique_recipe_ids)
    node_id_mapping = {id_val: idx for idx, id_val in enumerate(all_ids)}

    # Initialize projection layers for users and recipes
    user_dim = 25076  # Input dimension for user embeddings
    recipe_dim = 768  # Input dimension for recipe embeddings
    common_dim = 512  # Target dimension for both projections

    # Initialize the user and recipe projection layers
    user_project_layer = nn.Linear(user_dim, common_dim, dtype=torch.float32)
    recipe_project_layer = nn.Linear(recipe_dim, common_dim, dtype=torch.float32)

    # Initialize feature matrix
    num_nodes = len(all_ids)
    feat_data = np.zeros((num_nodes, common_dim), dtype=np.float32)

    # Project user embeddings
    for user_id in tqdm(unique_user_ids, desc="user"):
        user_embedding = retrieve_embedding(user_embedding_dict, user_id).astype(np.float32)
        feat_data[get_node_id(user_id, node_id_mapping)] = user_project_layer(
            torch.tensor(user_embedding)).detach().numpy()
    
    # Project recipe embeddings
    for recipe_id in tqdm(unique_recipe_ids, desc="recipe"):
        recipe_embedding = retrieve_embedding(recipe_embedding_dict, recipe_id).astype(np.float32)
        feat_data[get_node_id(recipe_id, node_id_mapping)] = recipe_project_layer(
            torch.tensor(recipe_embedding)).detach().numpy()

    print("Projected embeddings saved in feat_data.")

    # Build adjacency list and labels
    adj_lists = defaultdict(set)
    labels = {}
    for _, row in data.iterrows():
        recipe_id = row["recipe_id"]
        user_id = row["u"]
        rating = row["rating"]

        node_id1 = get_node_id(recipe_id, node_id_mapping)
        node_id2 = get_node_id(user_id, node_id_mapping)
        adj_lists[node_id1].add(node_id2)
        adj_lists[node_id2].add(node_id1)

        # Create label entries for undirected edges
        labels[(node_id1, node_id2)] = rating
        labels[(node_id2, node_id1)] = rating

    return feat_data, user_project_layer, recipe_project_layer, adj_lists, labels

def run_food(recipe_embedding_dict, user_embedding_dict):
    np.random.seed(1)
    random.seed(1)
    feat_data, user_project_layer, recipe_project_layer, adj_lists, labels = load_food(recipe_embedding_dict, user_embedding_dict)
    num_nodes = feat_data.shape[0]
    features = nn.Embedding(num_nodes, 512)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 512, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists,
                   agg2, base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(enc2, user_project_layer, recipe_project_layer)
    optimizer = optim.SGD(graphsage.parameters(), lr=0.001)
    
    # Load data
    data = pd.read_csv('food-data/interactions_train.csv')

    # Get unique user and recipe IDs
    unique_user_ids = data['u'].unique()
    unique_recipe_ids = data['recipe_id'].unique()

    # Create combined unique IDs and node ID mapping
    all_ids = list(unique_user_ids) + list(unique_recipe_ids)
    node_id_mapping = {id_val: idx for idx, id_val in enumerate(all_ids)}

    # Aggregate loss for each user
    batch_size = 256
    for batch in tqdm(range(100), desc="train"):
        optimizer.zero_grad()

        # Sample a batch of users
        batch_users = random.sample(list(unique_user_ids), 
                                    min(batch_size, len(unique_user_ids)))

        # Initialize total loss
        loss = 0.0

        for user in tqdm(batch_users, desc="user train"):
            # Map user ID to node ID
            mapped_user_id = get_node_id(user, node_id_mapping)
            user_embedding = feat_data[mapped_user_id]

            # Get connected recipe IDs for the user
            recipe_mapped_ids = adj_lists[mapped_user_id]

            # Get positive recipe embeddings (ratings >= 4)
            pos_embedding = feat_data[[get_node_id(recipe, node_id_mapping) 
                                    for recipe in recipe_mapped_ids 
                                    if labels[(mapped_user_id, recipe)] >= 4]]

            # Get low-rank positive recipe embeddings (ratings 1-3)
            low_rank_embedding = feat_data[[get_node_id(recipe, node_id_mapping) 
                                            for recipe in recipe_mapped_ids 
                                            if 1 <= labels[(mapped_user_id, recipe)] <= 3]]

            # Sample negative recipe IDs that are not interacted with by the user
            negative_recipe_ids = [recipe_id for recipe_id in unique_recipe_ids 
                                if (mapped_user_id, get_node_id(recipe_id)) not in labels]

            # Ensure there are enough negative samples to choose from
            if len(negative_recipe_ids) > 0:
                sampled_negative_ids = random.sample(negative_recipe_ids, min(16, len
                (negative_recipe_ids)))
                neg_embedding = feat_data[[get_node_id(neg_id, node_id_mapping) for neg_id in sampled_negative_ids]]
            else:
                continue  # Skip if no valid negative samples are available

            # Compute max-margin loss
            loss += graphsage.max_margin_loss(user_embedding, pos_embedding, 
                                            neg_embedding, low_rank_embedding)

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Batch {batch}, Loss: {loss.item()}")
    
    # Save the model
    torch.save(graphsage.state_dict(), "graphsage_model.pth")
    print("saved model")

if __name__ == "__main__":
    # Get embedding and run training
    recipe_embedding_dict = load_embedding('food-data/recipe_embeddings.pkl')
    user_embedding_dict = load_embedding('food-data/user_embedding.pkl')
    run_food(recipe_embedding_dict, user_embedding_dict)
