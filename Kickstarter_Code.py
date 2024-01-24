# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:03:54 2023

@author: jayac
"""

# Import data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from scipy.spatial.distance import cdist
import seaborn as sns
from matplotlib import pyplot
from sklearn.metrics import silhouette_samples


# Load the dataset
df = pd.read_csv("C:/Users/jayac/OneDrive/Desktop/McGill/DataMining/Individual Project/Kickstarter.csv")

# Create a new column 'amount_usd' by multiplying 'static_usd_rate' and 'goal'
df['amount_usd'] = df['static_usd_rate'] * df['goal']

# List of columns to remove
columns_to_remove = ['id', 'name', 'pledged','currency'	,'deadline','state_changed_at','created_at',	
                     'launched_at','staff_pick','name_len','blurb_len','state_changed_at_weekday',
                     'deadline_month','deadline_day','deadline_yr','deadline_hr','state_changed_at_month',
                    'state_changed_at_day','state_changed_at_yr','state_changed_at_hr','created_at_month',
                    'created_at_day','created_at_yr','created_at_hr','launched_at_month','launched_at_day',
                    'launched_at_yr','launched_at_hr','launch_to_state_change_days','static_usd_rate',
                    'usd_pledged','country','goal'

]

# Remove specified columns
df = df.drop(columns=columns_to_remove)

# Count missing values in each column
missing_values_count = df.isnull().sum()

# Total missing values in the entire dataframe
total_missing_values = df.isnull().sum().sum()

# Print or display the result
print("Missing Values Count per Column:")
print(missing_values_count)
print("\nTotal Missing Values in the DataFrame:", total_missing_values)

# Drop rows with missing values
df_cleaned = df.dropna()

# Count duplicate rows in the dataframe
duplicate_rows_count = df_cleaned.duplicated().sum()

# Print or display the result
print("Duplicate Rows Count:", duplicate_rows_count)

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df_cleaned, columns=['category','deadline_weekday','created_at_weekday','launched_at_weekday'], drop_first=True)
# Check column names in df_encoded
print(df_encoded.columns)

# Select relevant features
X = df_encoded.drop('state', axis=1)  # Predictors
y = df_encoded['state']  # Target variable

# Select only numeric columns for scaling
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
X_scaled_numeric = scaler.fit_transform(X[numeric_columns])

# Replace the scaled numeric values in the original DataFrame
X[numeric_columns] = X_scaled_numeric

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier (Random Forest in this example)
classifier = RandomForestClassifier(random_state=42)

# Train the classifier on the standardized data
classifier.fit(X_train, y_train)

# Make predictions on the standardized test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:\n', classification_rep)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a small decision tree
small_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the classifier on the training data
small_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred_small = small_classifier.predict(X_test)

# Evaluate the accuracy
accuracy_small = accuracy_score(y_test, y_pred_small)
print(f'Small Decision Tree Accuracy: {accuracy_small:.2f}')

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(small_classifier, filled=True, feature_names=X.columns.tolist(), class_names=y.unique().tolist(), rounded=True)
plt.show()

##### Clustering #####
# Load the dataset
df = pd.read_csv("C:/Users/jayac/OneDrive/Desktop/McGill/DataMining/Individual Project/Kickstarter.csv")

# Create a new column 'amount_usd' by multiplying 'static_usd_rate' and 'goal'
df['amount_usd'] = df['static_usd_rate'] * df['goal']
# Filter rows where 'state' is either 'successful' or 'failed'
df = df[df['state'].isin(['successful', 'failed'])]

# Selecting required columns
selected_columns = ['state','country', 'spotlight','backers_count', 'disable_communication',
                    'category', 'name_len_clean', 'blurb_len_clean','deadline_weekday',
                    'created_at_weekday','launched_at_weekday','create_to_launch_days','launch_to_deadline_days','amount_usd']
X = df[selected_columns]

# Count missing values in each column
missing_values_count = X.isnull().sum()

# Total missing values in the entire dataframe
total_missing_values = X.isnull().sum().sum()

# Print or display the result
print("Missing Values Count per Column:")
print(missing_values_count)
print("\nTotal Missing Values in the DataFrame:", total_missing_values)

# Drop rows with missing values
X_cleaned = X.dropna()

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X_cleaned, columns=['state', 'country', 'category',
                                              'deadline_weekday', 'created_at_weekday', 'launched_at_weekday'], drop_first=True)

# Select only numeric columns for scaling
numeric_columns = X_encoded.select_dtypes(include=['float64', 'int64']).columns

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
X_scaled_numeric = scaler.fit_transform(X_encoded[numeric_columns])

# Replace the scaled numeric values in the original DataFrame
X_encoded[numeric_columns] = X_scaled_numeric

# Calculate inertia for each value of k
withinss = []
for i in range (2,10):    
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(X_encoded)
    withinss.append(model.inertia_)

# Create a plot
pyplot.plot([2, 3, 4, 5, 6, 7, 8, 9], withinss)
pyplot.xlabel('Number of Clusters (k)')
pyplot.ylabel('Inertia')
pyplot.title('Elbow Method for Optimal k')
pyplot.show()


# Determining the number of clusters using silhouette score method
for k in range(2, 10):
    model = KMeans(n_clusters=k, random_state=5)
    model.fit(X_encoded)
    pred = model.predict(X_encoded)
    score = silhouette_score(X_encoded, pred)
    print('Silhouette Score for k = {}: {:<.3f}'.format(k, score))

# Perform K-Means Clustering with 5 clusters
kmeans = KMeans(n_clusters=5) 
kmeans.fit(X_encoded)

# Report the number of cereals in each cluster for K-Means Clustering
kmeans_labels = kmeans.labels_
unique, counts = np.unique(kmeans_labels, return_counts=True)
clusters_count_kmeans = dict(zip(unique, counts))
print("Projects in the first cluster:", clusters_count_kmeans[0])
print("Projects in the second cluster:", clusters_count_kmeans[1])
print("Projects in the third cluster:", clusters_count_kmeans[2])
print("Projects in the fourth cluster:", clusters_count_kmeans[3])
print("Projects in the fifth cluster:", clusters_count_kmeans[4])

# agg_labels and kmeans_labels are  cluster labels
kmeans_labels = kmeans.labels_

# Function to plot silhouette scores
def plot_silhouette(X_encoded, labels, title):
    silhouette_avg = silhouette_score(X_encoded, labels)
    sample_silhouette_values = silhouette_samples(X_encoded, labels)

    fig, ax = plt.subplots()
    y_lower = 10

    for i in range(len(np.unique(labels))):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / len(np.unique(labels)))
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title(title)
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster label")

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

# Plot silhouette for K-Means Clustering
plot_silhouette(X_encoded, kmeans_labels, "K-Means Clustering")

X_encoded['kmeans_cluster'] = kmeans_labels

# Group by cluster and calculate median for 'amount_usd'
median_per_cluster = X_encoded.groupby('kmeans_cluster')['amount_usd'].median()

# Print the median for each cluster
for cluster, median_value in median_per_cluster.items():
    print(f"Median for Cluster {cluster}: {median_value}")
    
# Group by cluster and calculate median for 'amount_usd'
median_per_cluster = X_encoded.groupby('kmeans_cluster')['create_to_launch_days'].median()

# Print the median for each cluster
for cluster, median_value in median_per_cluster.items():
    print(f"Median for Cluster {cluster}: {median_value}")

# Perform Agglomerative Clustering with 5 clusters using complete linkage
agg_cluster = AgglomerativeClustering(n_clusters=5, linkage='complete')
agg_labels = agg_cluster.fit_predict(X_encoded)

import numpy as np
# Report the number of projects in each cluster for Agglomerative Clustering
unique, counts = np.unique(agg_labels, return_counts=True)
clusters_count_agg = dict(zip(unique, counts))
print("Projects in the first cluster:", clusters_count_agg[0])
print("Projects in the second cluster:", clusters_count_agg[1])
print("Projects in the third cluster:", clusters_count_agg[2])
print("Projects in the fourth cluster:", clusters_count_agg[3])
print("Projects in the fifth cluster:", clusters_count_agg[4])

# Plotting scores
kmeans_silhouette = silhouette_score(X_encoded, kmeans_labels)
agg_silhouette = silhouette_score(X_encoded, agg_labels)

print(f"Silhouette Score for K-Means Clustering: {kmeans_silhouette}")
print(f"Silhouette Score for Agglomerative Clustering: {agg_silhouette}")

# Calculate silhouette scores for each sample
sample_silhouette_values = silhouette_samples(X_encoded, agg_labels)

# Plot silhouette scores for each sample in each cluster
fig, ax = plt.subplots()
y_lower = 10

for i in range(len(np.unique(agg_labels))):
    ith_cluster_silhouette_values = sample_silhouette_values[agg_labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.nipy_spectral(float(i) / len(np.unique(agg_labels)))
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.set_title("Agglomerative Clustering Silhouette Plot")
ax.set_xlabel("Silhouette Coefficient Values")
ax.set_ylabel("Cluster label")

# The vertical line for average silhouette score across all samples
silhouette_avg = silhouette_score(X_encoded, agg_labels)
ax.axvline(x=silhouette_avg, color="red", linestyle="--")

plt.show()

# Assuming kmeans_labels and agg_labels are your cluster labels
kmeans_silhouette_samples = silhouette_samples(X_encoded, kmeans_labels)
agg_silhouette_samples = silhouette_samples(X_encoded, agg_labels)

# Print the silhouette score for each cluster in K-Means
print("Silhouette Scores for K-Means Clustering:")
for cluster in range(max(kmeans_labels) + 1):
    cluster_silhouette = np.mean(kmeans_silhouette_samples[kmeans_labels == cluster])
    print(f"Cluster {cluster}: {cluster_silhouette}")

# Print the silhouette score for each cluster in Agglomerative Clustering
print("\nSilhouette Scores for Agglomerative Clustering:")
for cluster in range(max(agg_labels) + 1):
    cluster_silhouette = np.mean(agg_silhouette_samples[agg_labels == cluster])
    print(f"Cluster {cluster}: {cluster_silhouette}")

# Assuming X_encoded is your DataFrame with cluster labels
# and 'amount_usd', 'launch_to_deadline_days' are the columns of interest
sns.scatterplot(x='amount_usd', y='launch_to_deadline_days', hue='kmeans_cluster', data=X_encoded, palette='viridis')
plt.title('Cluster Division of amount_usd and launch_to_deadline_days')
plt.xlabel('amount_usd')
plt.ylabel('launch_to_deadline_days')
plt.show()

# and 'amount_usd', 'launch_to_deadline_days' are the columns of interest
sns.scatterplot(x='amount_usd', y='create_to_launch_days', hue='kmeans_cluster', data=X_encoded, palette='viridis')
plt.title('Cluster Division of amount_usd and create_to_launch_days')
plt.xlabel('amount_usd')
plt.ylabel('create_to_launch_days')
plt.show()