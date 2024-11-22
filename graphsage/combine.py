import pandas as pd

# Load the uploaded files
interactions_file_path = 'food-data/RAW_interactions.csv'
recipes_file_path = 'food-data/RAW_recipes.csv'

interactions_df = pd.read_csv(interactions_file_path)
recipes_df = pd.read_csv(recipes_file_path, )

# Re-extract relevant columns
recipes_df_subset = recipes_df[['recipe_name', 'recipe_id', 'recipe_description']]
interactions_df_subset = interactions_df[['user_id', 'recipe_id', 'rating', 'review']]

# Merge the two datasets on recipe_id and id
merged_df = pd.merge(
    interactions_df_subset,
    recipes_df_subset,
    left_on='recipe_id',
    right_on='recipe_id',
    how='left'
)

# Save the final dataframe to a CSV file and display it to the user
output_file_path = 'food-data/merged_recipes_interactions.csv'
merged_df.to_csv(output_file_path, index=False)

print(len(merged_df))
print(len(interactions_df))
