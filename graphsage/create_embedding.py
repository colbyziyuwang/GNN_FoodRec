import torch
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import pickle

def preprocess_ingredients(ingredients):
    """
    Preprocess the ingredients column to handle stringified lists.
    """
    try:
        # Convert string representation of list to a Python list
        ingredients_list = eval(ingredients)
        # Join ingredients into a single string
        return ", ".join(ingredients_list)
    except:
        return str(ingredients)

def get_embedding(recipe_name, ingredients, embedding_pipeline):
    """
    Get the embedding for a single recipe using RecipeBERT.
    """
    # Combine recipe name and ingredients into a single text
    recipe_text = f"{recipe_name}. Ingredients: {ingredients}"

    # Extract embeddings
    embedding = embedding_pipeline(recipe_text, return_tensors='pt')[0]

    # Perform mean pooling to get a single embedding vector for the recipe
    recipe_embedding = embedding.mean(axis=0)  # Mean pooling over token embeddings

    return recipe_embedding.squeeze().numpy()  # Convert to NumPy array for easier storage

def create_food_embedding():
    # Load RAW_recipes.csv
    dataset_path = "food-data/"
    raw_recipes = pd.read_csv(dataset_path + "RAW_recipes.csv")

    # Preprocess ingredients column
    raw_recipes['ingredients'] = raw_recipes['ingredients'].apply(preprocess_ingredients)

    # Set up device for MPS
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Initialize the RecipeBERT pipeline
    embedding_pipeline = pipeline(
        'feature-extraction',
        model='alexdseo/RecipeBERT',
        framework='pt',
        device=device
    )

    # Create a dictionary to store embeddings
    embedding_dict = {}

    # Generate embeddings for each recipe
    for i in tqdm(range(len(raw_recipes)), desc="bert"):
        recipe_id = raw_recipes['recipe_id'][i]
        recipe_name = str(raw_recipes['recipe_name'][i])
        ingredients = raw_recipes['ingredients'][i]

        # Get embedding for the recipe
        food_rep = get_embedding(recipe_name, ingredients, embedding_pipeline)

        # Store the embedding in the dictionary
        embedding_dict[recipe_id] = food_rep

    # Save embeddings to a pickle file
    with open('food-data/recipe_embeddings.pkl', 'wb') as f:
        pickle.dump(embedding_dict, f)

    print("Embeddings saved successfully.")

# Run the embedding creation
create_food_embedding()
