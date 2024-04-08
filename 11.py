import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def scale_data(df, index, columns, values):
    pivot = df.pivot_table(index=index, columns=columns, values=values, fill_value=0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot)
    return pd.DataFrame(scaled_data, index=pivot.index, columns=pivot.columns)

def compute_similarity(data):
    sim_matrix = cosine_similarity(data)
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

def predict(ratings, similarity, top_k):
    pred = np.zeros(ratings.shape)
    for i in range(ratings.shape[0]):
        top_k_users = np.argsort(similarity[i])[:-top_k-1:-1]
        for j in range(ratings.shape[1]):
            pred[i, j] = similarity[i, top_k_users].dot(ratings[top_k_users, j])
            pred[i, j] /= np.sum(np.abs(similarity[i, top_k_users]))
    return pred

def calculate_mae(predictions, actual):
    nonzero_actual = actual.nonzero()
    predicted_nonzero = predictions[nonzero_actual].flatten()
    actual_nonzero = actual[nonzero_actual].flatten()
    return mean_absolute_error(actual_nonzero, predicted_nonzero)

def find_top_products(data, top_n=10):
    item_totals = data.groupby('ItemID')['Rating'].sum().sort_values(ascending=False).head(top_n)
    return item_totals

# Prepare the data
np.random.seed(42)
data = pd.DataFrame({
    'UserID': np.random.randint(1, 100, 1000),
    'ItemID': np.random.randint(1, 20, 1000),
    'Rating': np.random.randint(1, 6, 1000)
})

scaled_user_data = scale_data(data, 'UserID', 'ItemID', 'Rating')
scaled_item_data = scale_data(data, 'ItemID', 'UserID', 'Rating')

# Compute similarities
user_similarity = compute_similarity(scaled_user_data.values)
item_similarity = compute_similarity(scaled_item_data.values.T)

# Predict ratings
user_based_predictions = predict(scaled_user_data.values, user_similarity, 5)
item_based_predictions = predict(scaled_item_data.values.T, item_similarity, 5)

# Calculate MAE for user-user and item-item collaborative filtering
user_mae = calculate_mae(user_based_predictions, scaled_user_data.values)
item_mae = calculate_mae(item_based_predictions, scaled_item_data.values.T)

# Find top 10 products
top_products = find_top_products(data)

# Print results
print("User-Based Collaborative Filtering MAE:", user_mae)
print("Item-Based Collaborative Filtering MAE:", item_mae)
print("\nTop 10 Products by Total Ratings:")
print(top_products)

# Save results to file
with open("ans11.txt", "w") as file:
    file.write("User-Based Collaborative Filtering MAE: " + str(user_mae) + "\n")
    file.write("Item-Based Collaborative Filtering MAE: " + str(item_mae) + "\n")
    file.write("\nTop 10 Products by Total Ratings:\n")
    file.write(str(top_products))
