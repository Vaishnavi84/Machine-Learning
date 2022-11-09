import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

#load the Penguins dataset 
data = pd.read_csv("C:\MSC-IT\Machine Learning\Hierarchical Clustering\penguins.csv")
data.head()

data['species'].value_counts()

print(data.shape) # (344, 9)

#trimming the dataset to the chosen columns and dropping rows with missing data 
df = data[['bill_length_mm', 'flipper_length_mm']]
df = df.dropna(axis=0)

print(df.head())

#use Scipy's hierarchy.linkage() to form clusters and plot them with hierarchy.dendrogram()

clusters = hierarchy.linkage(df, method="ward")

plt.figure(figsize=(8, 6))
dendrogram = hierarchy.dendrogram(clusters)
# Plotting a horizontal line based on the first biggest distance between clusters 
plt.axhline(150, color='red', linestyle='--'); 
# Plotting a horizontal line based on the second biggest distance between clusters 
plt.axhline(100, color='crimson'); 

#perform Agglomerative Clustering with Scikit-Learn to find cluster labels for the three types of penguins
clustering_model = AgglomerativeClustering(n_clusters=3, linkage="ward")
clustering_model.fit(df)
labels = clustering_model.labels_

#plot the data before and after Agglomerative Clustering with 3 clusters
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
sns.scatterplot(ax=axes[0], data=df, x='bill_length_mm', y='flipper_length_mm').set_title('Without cliustering')
sns.scatterplot(ax=axes[1], data=df, x='bill_length_mm', y='flipper_length_mm', hue=clustering_model.labels_).set_title('With clustering');

#Agglomerative Clustering without specifying the number of clusters
clustering_model_no_clusters = AgglomerativeClustering(linkage="ward")
clustering_model_no_clusters.fit(df)
labels_no_clusters = clustering_model_no_clusters.labels_

# plot the data without Agglomerative Clustering, with 3 clusters and with no pre defined clusters
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
sns.scatterplot(ax=axes[0], data=df, x='bill_length_mm', y='flipper_length_mm').set_title('Without cliustering')
sns.scatterplot(ax=axes[1], data=df, x='bill_length_mm', y='flipper_length_mm', hue=clustering_model.labels_).set_title('With 3 clusters')
sns.scatterplot(ax=axes[2], data=df, x='bill_length_mm', y='flipper_length_mm', hue=clustering_model_no_clusters.labels_).set_title('Without choosing number of clusters');
