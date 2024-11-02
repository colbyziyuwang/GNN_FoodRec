import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
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
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, enc, user_dim=25076, recipe_dim=768):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

        self.user_dim = user_dim
        self.recipe_dim = recipe_dim
        self.common_dim = 512

        # Projection layers
        self.user_project = nn.Linear(user_dim, self.common_dim)  # projects user embedding to common_dim
        self.recipe_project = nn.Linear(recipe_dim, self.common_dim)  # projects recipe embedding to common_dim

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in tqdm(range(100), desc="train"):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

    # Save the model
    torch.save(graphsage.state_dict(), "graphsage_model.pth")
    print("saved model")

def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in tqdm(range(200), desc="train"):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

    # Save the model
    torch.save(graphsage.state_dict(), "graphsage_model.pth")
    print("saved model")

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

if __name__ == "__main__":
    # create_food_embedding()
    # create_user_embedding()

    recipe_embedding_dict = load_embedding('food-data/recipe_embeddings.pkl')
    # retrieve_embedding(recipe_embedding_dict, 4684)
    user_embedding_dict = load_embedding('food-data/user_embedding.pkl')
    # retrieve_embedding(user_embedding_dict, 0)

    load_food(recipe_embedding_dict, user_embedding_dict)
