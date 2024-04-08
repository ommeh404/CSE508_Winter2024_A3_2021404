import pandas as pd

# Load your preprocessed data into a DataFrame
df = pd.read_csv()

# Automatically determine the columns for user_id, item_id, and rating
columns = df.columns
user_id_col = None
item_id_col = None
rating_col = None

# Iterate through the columns to find suitable ones
for col in columns:
    if 'user_id' in col.lower():
        user_id_col = col
    elif 'item_id' in col.lower():
        item_id_col = col
    elif 'rating' in col.lower():
        rating_col = col

# Check if all required columns are found
if user_id_col is None or item_id_col is None or rating_col is None:
    print("Error: Required columns not found in the DataFrame.")
    # Handle the error or exit gracefully
    exit()

# Define the user-item matrix
user_item_matrix = df.pivot(index=user_id_col, columns=item_id_col, values=rating_col).fillna(0)




from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_ratings = scaler.fit_transform(user_item_matrix)

from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity matrix
user_similarity_matrix = cosine_similarity(normalized_ratings)

# Find top N similar users
top_similar_users = {}
for N in [10, 20, 30, 40, 50]:
    top_similar_users[N] = {}
    for user_idx in range(len(user_similarity_matrix)):
        similar_users = sorted(list(enumerate(user_similarity_matrix[user_idx])), key=lambda x: x[1], reverse=True)[1:N+1]
        top_similar_users[N][user_idx] = similar_users

# Transpose the normalized ratings matrix to create item-item similarity matrix
item_similarity_matrix = cosine_similarity(normalized_ratings.T)

# Find top N similar items
top_similar_items = {}
for N in [10, 20, 30, 40, 50]:
    top_similar_items[N] = {}
    for item_idx in range(len(item_similarity_matrix)):
        similar_items = sorted(list(enumerate(item_similarity_matrix[item_idx])), key=lambda x: x[1], reverse=True)[1:N+1]
        top_similar_items[N][item_idx] = similar_items

import matplotlib.pyplot as plt

# Plot MAE against K for user-user recommender system
plt.figure(figsize=(10, 6))
for N, similar_users in top_similar_users.items():
    # Calculate MAE for each K using K-fold validation
    mae_values = []  # Store MAE values for each K
    for k in similar_users:
        # Implement K-fold validation, prediction, and error calculation here
        mae_values.append(mae)  # Replace mae with actual calculated MAE
    plt.plot(range(1, 6), mae_values, label=f'N={N}')

plt.title('MAE vs. K for User-User Recommender System')
plt.xlabel('K (Number of Similar Users)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.legend()
plt.show()

# Plot MAE against K for item-item recommender system (similar procedure as above)
