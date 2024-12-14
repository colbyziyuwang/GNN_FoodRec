import pandas as pd

# Load the uploaded files
interactions_file_path = 'food-data/RAW_interactions.csv'
recipes_file_path = 'food-data/RAW_recipes.csv'

interactions_df = pd.read_csv(interactions_file_path)
recipes_df = pd.read_csv(recipes_file_path)

# Debugging: Print the available columns in recipes_df
print("Available columns in recipes_df:", recipes_df.columns)

# Update the column names based on your dataset
recipes_df_subset = recipes_df[['name', 'id', 'description']]  # Adjust these names as needed
interactions_df_subset = interactions_df[['user_id', 'recipe_id', 'rating', 'review']]

# Merge the two datasets on recipe_id and id
merged_df = pd.merge(
    interactions_df_subset,
    recipes_df_subset,
    left_on='recipe_id',
    right_on='id',
    how='left'
)

# Save the final dataframe to a CSV file and display it to the user
output_file_path = 'food-data/merged_recipes_interactions.csv'
merged_df.to_csv(output_file_path, index=False)

print(f"Merged dataset saved to {output_file_path}")
print(f"Number of rows in merged dataset: {len(merged_df)}")
