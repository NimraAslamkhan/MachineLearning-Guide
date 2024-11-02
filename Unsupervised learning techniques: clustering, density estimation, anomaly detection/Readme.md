# Unsupervised Learning

Unsupervised learning focuses on identifying patterns in unlabeled data, making it highly valuable for applications where data lacks labels. Unlike supervised learning, where labeled data is essential, unsupervised techniques can process vast amounts of unlabeled data and uncover insights without human intervention.

## Key Unsupervised Learning Techniques

### Clustering
Clustering groups similar data points together, making it valuable for various tasks, including:
- Customer segmentation
- Data analysis
- Dimensionality reduction
- Feature engineering
- Anomaly detection
- Semi-supervised learning
- Search engines
- Image segmentation

Common clustering algorithms:
- **k-means**
- **DBSCAN**

### Anomaly Detection
Anomaly detection identifies unusual data points, or "outliers," by learning normal data patterns. Applications include:
- Fraud detection
- Defective product identification
- Trend analysis

This technique is useful for catching abnormalities in datasets.

### Density Estimation
Density estimation estimates the probability density function (PDF) of a dataset, revealing how data points are distributed. It is commonly used for:
- Anomaly detection (by identifying low-density regions)
- Data analysis
- Visualization

## Applications
These unsupervised techniques are applied across various fields:
- **Clustering**: Groups customers based on behaviors or segments website users by activity.
- **Anomaly Detection**: Essential for fraud detection.
- **Density Estimation**: Helps identify rare events.

By leveraging these methods, unsupervised learning can extract valuable insights from large datasets, making it a foundation for many AI-driven applications.

# Clustering Algorithms: Basics and Variations

This document provides an overview of clustering algorithms, particularly focusing on **k-means clustering**, a widely-used unsupervised learning technique. Clustering helps in grouping similar data points, typically based on their proximity in feature space.

---

## Image Segmentation and Clustering

Clustering algorithms are useful for:
- Simplifying image colors by grouping similar colors and replacing them with the average color of each cluster.
- Applications like **image segmentation** in object detection and tracking, where clustering helps in identifying distinct regions or objects within images.

---

## Types of Clusters

Different clustering algorithms target different types of clusters:
- **Centroid-based clusters**: Algorithms like **k-means** group data around centroids.
- **Density-based clusters**: Algorithms like **DBSCAN** can detect clusters of varying shapes.
- **Hierarchical clustering**: Organizes clusters into a tree-like structure of sub-clusters.

---

## K-means Clustering

**K-means** is a popular and efficient algorithm that assigns data points to clusters based on proximity to centroids.

### How K-means Works:
1. Specify **k**, the number of clusters, in advance.
2. Iteratively:
   - Update centroids based on the mean of assigned points.
   - Reassign data points to the nearest centroids.
3. Repeat until centroids stabilize (convergence).

> **Note**: K-means may converge to a local optimum depending on initialization.

---

## K-means Variants and Enhancements

To improve performance and flexibility, there are several k-means variants:

- **K-means++**: Enhances initialization by selecting centroids that are distant from each other, reducing the likelihood of suboptimal clustering.
- **Elkan's K-means**: Speeds up training using the triangle inequality, which avoids unnecessary distance calculations (especially beneficial for large datasets).
- **Mini-Batch K-means**: Processes small batches of data instead of the entire dataset, making it faster and more memory-efficient for large datasets. This method may be slightly less accurate but is highly efficient.

---

## Applications of K-means

- **Dimensionality Reduction**: Transforms data into distance features relative to each centroid, making the data easier to work with in lower-dimensional space.
- **Semi-Supervised Learning**: K-means can assist in identifying typical data points (centroids) and use them as class representatives in label-scarce settings.
- **Anomaly Detection**: Outliers can be detected based on their distance from the nearest centroid, as centroids represent the typical data structure.

---
# Clustering Algorithms Overview

## Introduction

K-means is a popular clustering algorithm known for its speed and scalability. However, it has several limitations that can impact its effectiveness in certain scenarios. This document provides an overview of the limits of k-means, its applications in image segmentation, and its use in semi-supervised learning. Additionally, we will explore alternative clustering algorithms, including DBSCAN.

## Limits of K-means

Despite its many merits, k-means is not perfect:

- **Initialization**: The algorithm must be run multiple times to avoid suboptimal solutions.
- **Fixed Clusters**: The number of clusters (k) must be specified beforehand, which can be challenging.
- **Cluster Shape Constraints**: K-means does not perform well with clusters of varying sizes, densities, or nonspherical shapes. For example, in datasets with elliptical clusters, k-means can misassign points. Gaussian Mixture Models (GMM) are often better suited for such cases.

## Using Clustering for Image Segmentation

Image segmentation is the task of partitioning an image into multiple segments. There are several variants:

1. **Color Segmentation**: Groups pixels with similar colors, suitable for applications such as estimating forest coverage from satellite images.
2. **Semantic Segmentation**: Groups pixels belonging to the same object type (e.g., all pedestrian pixels in a self-driving car's vision system).
3. **Instance Segmentation**: Groups pixels of each unique object, such as individual pedestrians.

### Color Segmentation with K-means

K-means can be applied to color segmentation. The process involves:

1. Loading the image as a 3D array (height, width, and color channels).
2. Reshaping the array to create a long list of RGB colors.
3. Clustering these colors using k-means to produce a segmented image.

However, k-means may struggle with small, distinct areas (e.g., a ladybug's color) due to its preference for clusters of similar sizes.

## Using Clustering for Semi-Supervised Learning

Clustering is beneficial in semi-supervised learning when there are many unlabeled instances and few labeled ones. 

### Baseline Model

1. A logistic regression model trained on 50 random labeled instances yields around 74.8% accuracy.
2. **Cluster Representatives**: Clustering the unlabeled data and labeling representative points (closest to cluster centers) increases accuracy to 84.9%.
3. **Label Propagation**: Assigning cluster labels to all points within a cluster boosts accuracy to 89.4%.
4. **Outlier Removal**: By removing the 1% of instances farthest from cluster centers, accuracy rises to 90.9%, surpassing the fully labeled model (90.7%).

## Alternative Clustering Algorithms

### DBSCAN

DBSCAN is another popular clustering algorithm that utilizes local density estimation. This approach allows it to identify clusters of arbitrary shapes, making it suitable for datasets with complex structures, unlike k-means.

## DBSCAN: Density-Based Spatial Clustering of Applications with Noise

DBSCAN is a clustering algorithm that identifies clusters as regions of high density. It works as follows:

1. For each instance, count how many instances are within a distance ε (epsilon) from it, forming its ε-neighborhood.
2. An instance is classified as a core instance if it has at least `min_samples` instances in its ε-neighborhood.
3. All instances in the ε-neighborhood of a core instance belong to the same cluster, potentially forming long sequences of neighboring core instances.
4. Instances that are not core instances and do not have one in their neighborhood are considered anomalies.
# Strengths and Limitations of DBSCAN

## Strengths:
- Can identify clusters of arbitrary shape.
- Robust to outliers.
- Requires only two hyperparameters: `eps` and `min_samples`.

## Limitations:
- Struggles with varying cluster density.
- Computational complexity is approximately O(m n), making it less suitable for large datasets.

# Other Clustering Algorithms

Several other clustering algorithms are available in Scikit-Learn:

1. **Agglomerative Clustering**
   - Builds a hierarchy of clusters by merging the nearest pairs iteratively.
   - Can capture various shapes and scales, producing a flexible cluster tree.
   - Scalability improves with a connectivity matrix but struggles with large datasets without it.

2. **BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)**
   - Designed for very large datasets, it builds a tree structure to cluster instances efficiently without requiring all instances to be stored in memory.

3. **Mean-Shift**
   - Iteratively shifts circles centered on instances to find local density maxima.
   - Suitable for identifying clusters of any shape but can struggle with internal density variations.

4. **Affinity Propagation**
   - Instances exchange messages to elect exemplars representing clusters.
   - Does not require specifying the number of clusters in advance, but has a high computational complexity (O(m²)).

5. **Spectral Clustering**
   - Uses a similarity matrix to create a low-dimensional embedding for clustering.
   - Effective for complex cluster structures but does not scale well to large datasets.


# Gaussian Mixture Models (GMM)

## Overview
A Gaussian mixture model (GMM) is a probabilistic model that assumes instances are generated from a mixture of several Gaussian distributions with unknown parameters. Clusters formed by instances from a single Gaussian distribution typically resemble ellipsoids. Each cluster can vary in shape, size, density, and orientation.

## GMM Variants
The simplest variant, implemented in the `GaussianMixture` class, requires prior knowledge of the number \( k \) of Gaussian distributions. The process for generating the dataset \( X \) involves:

1. Randomly selecting a cluster from \( k \) clusters, where the probability of choosing cluster \( j \) is its weight \( \phi \).
2. Sampling the instance from the chosen cluster’s Gaussian distribution characterized by mean \( \mu \) and covariance matrix \( \Sigma \).

## Using GMM in Scikit-Learn
The `GaussianMixture` class in Scikit-Learn simplifies parameter estimation:
# Gaussian Mixture Models (GMM)

## Overview
A Gaussian mixture model (GMM) is a probabilistic model that assumes that the instances were generated from a mixture of several Gaussian distributions. Each cluster formed by these distributions appears as an ellipsoid. The GMM provides a flexible method for modeling complex data distributions.

## Estimated Parameters
After fitting a GMM to a dataset, you can access the estimated parameters:
The algorithm uses the Expectation-Maximization (EM) method, which consists of two iterative steps:

Expectation step: Assigns instances to clusters based on current parameters.
Maximization step: Updates cluster parameters based on weighted instances.

Model Predictions
The GMM allows for both hard and soft clustering:

Hard Clustering: Use predict() method to assign instances to the most likely cluster.

Soft Clustering: Use predict_proba() method to estimate the probability of each instance belonging to each cluster.

Generating New Instances
You can sample new instances from the fitted model using the sample() method:

Density Estimation
Estimate the density at given locations using the score_samples() method, which provides the log probability density function values:

Covariance Types
Different covariance types can be specified to constrain the shapes of the clusters:

"spherical": Clusters are spherical but can have different diameters.
"diag": Clusters can have any ellipsoidal shape, with axes parallel to the coordinate axes.
"tied": All clusters share the same covariance matrix.
"full" (default): Each cluster can take on any shape, size, and orientation.
Anomaly Detection
GMM can be used for anomaly detection by identifying instances in low-density regions. You can define a density threshold based on the desired false positive and negative rates.

Example of Anomaly Detection
To identify outliers, you can calculate the density scores and set a threshold:
# Selecting the Number of Clusters

## K-Means Clustering
- Use **inertia** or the **silhouette score** to determine the optimal number of clusters.

## Gaussian Mixture Models (GMM)
- Inertia and silhouette scores are unreliable for non-spherical or varying-sized clusters.
- Instead, utilize information criteria like the **Bayesian Information Criterion (BIC)** and the **Akaike Information Criterion (AIC)** to select the number of clusters. Both metrics penalize complexity (number of parameters) and reward good data fit.
  
  **Formulas:**
  - **BIC**: 
    \[
    BIC = \log(m)p - 2\log(\hat{L})
    \]
  - **AIC**: 
    \[
    AIC = 2p - 2\log(\hat{L})
    \]
  
  Where:
  - \( m \) is the number of instances.
  - \( p \) is the number of model parameters.
  - \( \hat{L} \) is the maximized likelihood.

## Bayesian Gaussian Mixture
- Instead of manually determining clusters, use the `BayesianGaussianMixture` class to automatically disregard unnecessary clusters.
- Set `n_components` to a value greater than the expected optimal number, allowing the algorithm to adjust the weights accordingly.

## Limitations of Gaussian Mixture Models
- GMM performs well with ellipsoidal clusters but struggles with clusters of differing shapes, as demonstrated with a dataset resembling moons.

# Alternative Algorithms for Anomaly and Novelty Detection

1. **Fast-MCD**: 
   - Detects outliers by estimating a single Gaussian distribution, ignoring likely outliers to better define the data envelope.

2. **Isolation Forest**: 
   - Efficiently isolates anomalies in high-dimensional data by recursively partitioning the dataset.

3. **Local Outlier Factor (LOF)**:
   - Compares local densities to identify anomalies, which tend to have lower density than their neighbors.

4. **One-Class SVM**:
   - Used for novelty detection by separating a single class of data from the origin in a high-dimensional space.

5. **PCA and Dimensionality Reduction Techniques**:
   - Anomalies can be identified by comparing reconstruction errors; anomalies typically have larger errors than normal instances.

