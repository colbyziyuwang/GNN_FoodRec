import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class RewardPredictorNN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(RewardPredictorNN, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Scale sigmoid output to range 1 to 5
        return torch.sigmoid(x) * 4 + 1

class InteractionDataset(Dataset):
    def __init__(self, interactions, embedding_array, node_id_mapping):
        self.interactions = interactions
        self.embedding_array = embedding_array
        self.node_id_mapping = node_id_mapping

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_id = self.interactions.iloc[idx]['u']
        recipe_id = self.interactions.iloc[idx]['recipe_id']
        rating = self.interactions.iloc[idx]['rating']

        # Get state vector (user embedding)
        state_embedding = self.embedding_array[get_node_id(user_id, self.node_id_mapping)]
        action_embedding = self.embedding_array[get_node_id(recipe_id, self.node_id_mapping)]

        # Combine state and action embeddings
        combined_embedding = np.concatenate((state_embedding, action_embedding)).astype('float32')

        return combined_embedding, torch.tensor(rating, dtype=torch.float32)

# Function to retrieve node ID from the mapping
def get_node_id(user_or_recipe_id, node_id_mapping):
    return node_id_mapping.get(user_or_recipe_id)

# Load data
data = pd.read_csv('food-data/interactions_train.csv')

# Get unique user and recipe IDs
unique_user_ids = data['u'].unique()
unique_recipe_ids = data['recipe_id'].unique()

# Create combined unique IDs and node ID mapping
all_ids = list(unique_user_ids) + list(unique_recipe_ids)
node_id_mapping = {id_val: idx for idx, id_val in enumerate(all_ids)}

# Use np.memmap to load the file without fully loading into memory
embedding_array = np.memmap('graphsage_embeddings.npy', dtype='float32', mode='r', shape=(len(all_ids), 512))

# Create dataset and dataloader
dataset = InteractionDataset(data, embedding_array, node_id_mapping)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, optimizer, and loss function
rewardNN = RewardPredictorNN(input_dim=512, action_dim=512)
optimizer = optim.SGD(rewardNN.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
num_epochs = 10
rewardNN.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_embeddings, batch_ratings in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()

        # Forward pass
        predictions = rewardNN(batch_embeddings)
        predictions = predictions.squeeze()  # Ensure proper shape for loss calculation

        # Compute loss
        loss = criterion(predictions, batch_ratings)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

print("Neural Network training complete.")
