import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict
from encoders import Encoder
from aggregators import MeanAggregator
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import pandas as pd
import zipfile
from torch.nn import init
from torch.utils.data import DataLoader, Dataset

"""
Simple supervised GraphSAGE model to learn useful node embeddings for user
and recipes
"""

class SupervisedGraphSage(nn.Module):
    def __init__(self, enc, num_classes=6):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc

        # Fully connected layer for prediction
        self.fc = nn.Linear(enc.embed_dim * 2, num_classes)  # For concatenated user and recipe embeddings
        self.xent = nn.CrossEntropyLoss()

        # Initialize weights of the fully connected layer
        init.xavier_uniform_(self.fc.weight)

    def forward(self, user_nodes, recipe_nodes):
        # Get user and recipe embeddings
        user_embeds = self.enc(user_nodes)  # Shape: [embed_dim, batch_size]
        recipe_embeds = self.enc(recipe_nodes)  # Shape: [embed_dim, batch_size]

        # Concatenate user and recipe embeddings
        concatenated_embeds = torch.cat((user_embeds, recipe_embeds), dim=0)  # Shape: [embed_dim * 2, batch_size]

        # Predict scores
        scores = self.fc(concatenated_embeds.T)  # Shape: [batch_size, num_classes]
        return scores

    def loss(self, user_nodes, recipe_nodes, labels):
        # Forward pass to compute scores
        scores = self.forward(user_nodes, recipe_nodes)

        # Compute cross-entropy loss
        return self.xent(scores, labels)

class InteractionDataset(Dataset):
    def __init__(self, interactions, node_id_mapping):
        self.interactions = interactions
        self.node_id_mapping = node_id_mapping

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_id = self.interactions.iloc[idx]['user_id']
        recipe_id = self.interactions.iloc[idx]['recipe_id']
        rating = self.interactions.iloc[idx]['rating']

        return get_node_id(user_id, self.node_id_mapping), get_node_id(recipe_id, self.node_id_mapping), torch.tensor(rating, dtype=torch.long)
    
def load_embedding(filepath):
    # Load the dictionary
    with open(filepath, 'rb') as f:
        embedding_dict = pickle.load(f)
    return embedding_dict

def retrieve_embedding(embedding_dict, id):
    # Retrieve embedding for a specific ID
    embedding = embedding_dict[id]
    
    return embedding

# Function to retrieve node ID from the mapping
def get_node_id(user_or_recipe_id, node_id_mapping):
    return node_id_mapping.get(user_or_recipe_id)

# Define function to load food embeddings with projection layers
def load_food(recipe_embedding_dict, user_embedding_dict):    
    # Load data
    data = pd.read_csv('food-data/merged_recipes_interactions.csv')

    # Get unique user and recipe IDs
    unique_user_ids = data['user_id'].unique()
    unique_recipe_ids = data['recipe_id'].unique()

    # Create combined unique IDs and node ID mapping
    all_ids = list(unique_user_ids) + list(unique_recipe_ids)
    node_id_mapping = {id_val: idx for idx, id_val in enumerate(all_ids)}

    # Initialize feature matrix
    embedding_dim = 768
    num_nodes = len(all_ids)
    feat_data = np.zeros((num_nodes, embedding_dim), dtype=np.float32)

    # Project and normalize user embeddings
    for user_id in tqdm(unique_user_ids, desc="user"):
        user_embedding = retrieve_embedding(user_embedding_dict, user_id).astype(np.float32)
        # Normalize the embedding
        user_embedding = user_embedding / np.linalg.norm(user_embedding)
        feat_data[get_node_id(user_id, node_id_mapping)] = user_embedding

    # Project and normalize recipe embeddings
    for recipe_id in tqdm(unique_recipe_ids, desc="recipe"):
        recipe_embedding = retrieve_embedding(recipe_embedding_dict, recipe_id).astype(np.float32)
        # Normalize the embedding
        recipe_embedding = recipe_embedding / np.linalg.norm(recipe_embedding)
        feat_data[get_node_id(recipe_id, node_id_mapping)] = recipe_embedding

    # Build adjacency list and labels
    adj_lists = defaultdict(set)
    labels = {}
    for _, row in data.iterrows():
        recipe_id = row["recipe_id"]
        user_id = row["user_id"]
        rating = row["rating"]

        node_id1 = get_node_id(recipe_id, node_id_mapping)
        node_id2 = get_node_id(user_id, node_id_mapping)
        adj_lists[node_id1].add(node_id2)
        adj_lists[node_id2].add(node_id1)

        # Create label entries for undirected edges
        labels[(node_id1, node_id2)] = rating
        labels[(node_id2, node_id1)] = rating

    return feat_data, adj_lists, labels

def run_food(recipe_embedding_dict, user_embedding_dict):
    np.random.seed(1)
    random.seed(1)
    feat_data, adj_lists, labels = load_food(recipe_embedding_dict, user_embedding_dict)
    num_nodes = feat_data.shape[0]
    features = nn.Embedding(num_nodes, 768)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 768, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists,
                   agg2, base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(enc2)
    optimizer = optim.SGD(graphsage.parameters(), lr=0.001)
    
    # Load data
    data = pd.read_csv('food-data/merged_recipes_interactions.csv')

    # Get unique user and recipe IDs
    unique_user_ids = data['user_id'].unique()
    unique_recipe_ids = data['recipe_id'].unique()

    # Create combined unique IDs and node ID mapping
    all_ids = list(unique_user_ids) + list(unique_recipe_ids)
    node_id_mapping = {id_val: idx for idx, id_val in enumerate(all_ids)}
    
    # Create dataset and dataloader
    dataset = InteractionDataset(data,  node_id_mapping)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_user_nodes, batch_recipe_nodes, batch_ratings in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()

            # Forward pass
            loss = graphsage.loss(batch_user_nodes, batch_recipe_nodes, batch_ratings)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    # Open a ZIP file to save embeddings incrementally in binary format
    with zipfile.ZipFile("graphsage_embeddings.zip", "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for node_id in tqdm(range(len(feat_data)), desc="Saving embeddings"):
            # Encode the embedding using the model's encoder
            encoded_embedding = graphsage.enc(torch.tensor([node_id]))  # Pass node ID as a tensor
            encoded_embedding = encoded_embedding.squeeze()

            # Detach and convert to NumPy array for saving
            encoded_array = encoded_embedding.detach().numpy()

            # Convert to bytes and save to ZIP file with unique filename
            embedding_filename = f"embedding_{node_id}.npy"
            with zipf.open(embedding_filename, "w") as file:
                np.save(file, encoded_array)

    print("Embeddings saved successfully in graphsage_embeddings.zip")

if __name__ == "__main__":
    # Get embedding and run training
    recipe_embedding_dict = load_embedding('food-data/recipe_embeddings.pkl')
    user_embedding_dict = load_embedding('food-data/user_embeddings.pkl')
    run_food(recipe_embedding_dict, user_embedding_dict)
