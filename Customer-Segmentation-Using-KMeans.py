import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Inspect the first few rows
df.head()

# Exploratory Data Analysis (EDA)

# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30, color='blue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Distribution of Annual Income
plt.figure(figsize=(10, 6))
sns.histplot(df['Annual Income'], kde=True, bins=30, color='green')
plt.title('Distribution of Annual Income')
plt.xlabel('Annual Income')
plt.ylabel('Count')
plt.show()

# Distribution of Spending Score
plt.figure(figsize=(10, 6))
sns.histplot(df['Spending Score (1-100)'], kde=True, bins=30, color='purple')
plt.title('Distribution of Spending Score (1-100)')
plt.xlabel('Spending Score')
plt.ylabel('Count')
plt.show()

# Feature Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'Annual Income', 'Spending Score (1-100)']])

# Finding the optimal number of clusters using the Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Elbow Method plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', color='red')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Applying K-Means with the optimal number of clusters
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualizing Clusters in 2D using PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df['Cluster']

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=df_pca, s=100)
plt.title('Customer Segments (2D PCA)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

# Visualizing Spending Score vs Annual Income with Clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Annual Income', y='Spending Score (1-100)', hue='Cluster', palette='viridis', data=df, s=100)
plt.title('Clusters of Customers (Spending Score vs Annual Income)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score (1-100)')
plt.show()
