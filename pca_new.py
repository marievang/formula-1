import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import os
#import csv
os.chdir("/home/emikot/Desktop/linux/Python/chav/final_project/Downloaded_data")

# Load data
data = pd.read_csv('current_driver_dataset.csv')
df = pd.DataFrame(data)
print(df.head())

#drop non numerical feature
df = df.drop(["code", "constructorRef"], axis=1) 
print(df.head())

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X = std_scaler.fit_transform(df)

# PCA
pca = PCA(n_components=12)
X_pca = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)
print("Cumulative Explained Variance:", np.cumsum(explained_variance))
# Plot explained variance
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()

# Loadings 
loadings = pca.components_

# Define feature names
feature_names = df.columns
loading_df = pd.DataFrame(loadings.T, index=feature_names, columns=[f'PC{i+1}' for i in range(loadings.shape[0])])

# Feature importance (sum of absolute loadings)
important_feat = loading_df.abs().sum(axis=1).sort_values(ascending=False)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
# Top features
print("Top Features:")
print(important_feat.head(10))

# Plot feature importance
important_feat.plot(kind='bar')
plt.title('Feature Importance from PCA Loadings')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.show()

# Display the loadings for each pc
print("Principal Component Loadings:")
print(loading_df)

import seaborn as sns
# Plot heatmap of loadings
plt.figure(figsize=(10, 8))
sns.heatmap(loading_df, annot=True, cmap='coolwarm')
plt.title('Feature Importance in Principal Components')
plt.show()



# Identify the top contributing features for each pc
for i in range(loadings.shape[0]):
    print(f"\nPrincipal Component {i+1} top features:")
    top_features = loading_df.iloc[:, i].abs().sort_values(ascending=False)
    print(top_features.head(10))  # top 10 features

