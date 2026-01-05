# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import random
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ------------------------------
# 2️⃣ Load Dataset
# ------------------------------
df = pd.read_csv('Netflix_Dataset_500rows.csv')
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# ------------------------------
# 3️⃣ Data Preprocessing
# ------------------------------
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

df['UserID_enc'] = user_encoder.fit_transform(df['UserID'])
df['MovieID_enc'] = movie_encoder.fit_transform(df['MovieID'])

num_users = df['UserID_enc'].nunique()
num_movies = df['MovieID_enc'].nunique()

print(f"Number of Users: {num_users}, Movies: {num_movies}")

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ------------------------------
# 4️⃣ Machine Learning: KNN
# ------------------------------
start_ml = time.time()

# User-Item matrix
rating_matrix = train_df.pivot_table(index='UserID_enc', columns='MovieID_enc', values='Rating').fillna(0)

# KNN model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(rating_matrix)

# Predict function
def predict_knn(user_id, movie_id):
    user_vector = rating_matrix.iloc[user_id].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=5)
    neighbor_ratings = rating_matrix.iloc[indices[0], movie_id]
    return neighbor_ratings.mean()

# Apply predictions
test_df = test_df.copy()
test_df['Pred_KNN'] = test_df.apply(lambda x: predict_knn(x['UserID_enc'], x['MovieID_enc']), axis=1)

# Evaluate
ml_rmse = np.sqrt(mean_squared_error(test_df['Rating'], test_df['Pred_KNN']))
end_ml = time.time()
print(f"ML (KNN) RMSE: {ml_rmse:.4f}, Time: {end_ml-start_ml:.2f}s")

# ------------------------------
# 5️⃣ Deep Learning: Autoencoder
# ------------------------------
start_dl = time.time()

# Prepare matrix
user_item_matrix = rating_matrix.values

# Autoencoder
input_layer = Input(shape=(num_movies,))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(num_movies, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Train
autoencoder.fit(user_item_matrix, user_item_matrix, epochs=50, batch_size=16, verbose=0)

# Predict
pred_matrix = autoencoder.predict(user_item_matrix)

# Function for test prediction
def predict_dl(user_id, movie_id):
    return pred_matrix[user_id, movie_id]

test_df['Pred_DL'] = test_df.apply(lambda x: predict_dl(x['UserID_enc'], x['MovieID_enc']), axis=1)

# Evaluate DL
dl_rmse = np.sqrt(mean_squared_error(test_df['Rating'], test_df['Pred_DL']))
end_dl = time.time()
print(f"DL (Autoencoder) RMSE: {dl_rmse:.4f}, Time: {end_dl-start_dl:.2f}s")

# ------------------------------
# 6️⃣ Top-N Recommendations (ML)
# ------------------------------
def top_n_knn(user_id, n=5):
    user_vector = rating_matrix.iloc[user_id].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=5)
    neighbor_ratings = rating_matrix.iloc[indices[0]]
    pred_ratings = neighbor_ratings.mean(axis=0)
    top_movies = pred_ratings.sort_values(ascending=False).head(n)
    return top_movies.index.tolist()

print("\nTop-5 movies (ML) for first 3 users:")
for user in [0,1,2]:
    top_movies = top_n_knn(user)
    print(f"User {user_encoder.inverse_transform([user])[0]}: {top_movies}")

# ------------------------------
# 7️⃣ Top-N Recommendations (DL)
# ------------------------------
def top_n_dl(user_id, n=5):
    user_pred = pred_matrix[user_id]
    top_movies = np.argsort(user_pred)[::-1][:n]
    return top_movies.tolist()

print("\nTop-5 movies (DL) for first 3 users:")
for user in [0,1,2]:
    top_movies = top_n_dl(user)
    print(f"User {user_encoder.inverse_transform([user])[0]}: {top_movies}")

# ------------------------------
# 8️⃣ Genre Analysis
# ------------------------------
test_df = test_df.merge(df[['MovieID','Genre']], on='MovieID', how='left')

genre_ml = test_df.groupby('Genre')['Pred_KNN'].mean()
genre_dl = test_df.groupby('Genre')['Pred_DL'].mean()

print("\nAverage predicted rating per Genre (ML):")
print(genre_ml)
print("\nAverage predicted rating per Genre (DL):")
print(genre_dl)

# ------------------------------
# 9️⃣ Visualizations
# ------------------------------
# RMSE comparison
plt.figure(figsize=(8,5))
plt.bar(['ML','DL'], [ml_rmse, dl_rmse], color=['skyblue','orange'])
plt.title('RMSE Comparison')
plt.ylabel('RMSE')
plt.show()

# Time comparison
plt.figure(figsize=(8,5))
plt.bar(['ML','DL'], [end_ml-start_ml, end_dl-start_dl], color=['skyblue','orange'])
plt.title('Training + Prediction Time')
plt.ylabel('Time (seconds)')
plt.show()

# Genre comparison
plt.figure(figsize=(8,5))
genre_ml.plot(kind='bar', color='skyblue', alpha=0.7, label='ML')
genre_dl.plot(kind='bar', color='orange', alpha=0.5, label='DL')
plt.title('Average Predicted Rating per Genre')
plt.ylabel('Predicted Rating')
plt.legend()
plt.show()

# Actual vs Predicted Ratings
plt.figure(figsize=(6,6))
plt.scatter(test_df['Rating'], test_df['Pred_KNN'], color='blue', alpha=0.5, label='ML KNN')
plt.scatter(test_df['Rating'], test_df['Pred_DL'], color='red', alpha=0.5, label='DL Autoencoder')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.legend()
plt.show()

# ------------------------------
# 10️⃣ Summary
# ------------------------------
print("\n----------- Project Summary -----------")
print(f"ML (KNN) RMSE: {ml_rmse:.4f}, Time: {end_ml-start_ml:.2f}s")
print(f"DL (Autoencoder) RMSE: {dl_rmse:.4f}, Time: {end_dl-start_dl:.2f}s")
print("--------------------------------------")
