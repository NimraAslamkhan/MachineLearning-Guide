What Is Machine Learning?
Machine learning is the science (and art) of programming computers so they
can learn from data.


The part of a machine learning system that learns and
makes predictions is called a model

Neural networks and random forests are
examples of models.

Why Use Machine Learning?

Automated Learning: ML-based spam filters automatically detect patterns in spam emails by comparing word frequencies in spam vs. ham (non-spam) emails. It adapts over time as new spam patterns emerge (e.g., detecting "For U" without manual intervention).
Better Performance & Simplicity: ML models are easier to maintain, shorter in code, and generally more accurate because they continuously improve based on the data.

data mining
ML models can be inspected to understand what features (e.g., certain words) are the best predictors for spam, often revealing new trends. This process of discovering hidden patterns in large datasets is called data mining

Examples of Applications
1 image classification, typically performed using convolutional
neural networks

his is semantic image segmentation, where each pixel in the image is
classified (as we want to determine the exact location and shape of
tumors), typically using CNNs or transformers.
This is natural language processing (NLP), and more specifically text
classification, which can be tackled using recurrent neural networks
(RNNs) and CNNs, but transformers work even better


Types of Machine Learning Systems:
Based on Training Supervision:

Supervised Learning: Involves labeled data to train models for tasks like classification and regression.

Unsupervised Learning: Deals with unlabeled data to find hidden patterns, such as clustering and dimensionality reduction.
Semi-Supervised Learning: Combines a small amount of labeled data with a large amount of unlabeled data.
Self-Supervised Learning: Generates labels from the data itself to train models without external labeling.
Reinforcement Learning: Trains agents to make sequences of decisions by maximizing cumulative rewards.


Based on Learning Incrementally:

Online Learning: Models learn continuously by updating with new data on the fly.
Batch Learning: Models are trained on a fixed dataset all at once and require retraining with new data.
Based on Learning Approach:

Instance-Based Learning: Models memorize training instances and make predictions by comparing new data to these instances (e.g., k-Nearest Neighbors).
Model-Based Learning: Models generalize patterns from the training data to make predictions (e.g., linear regression, neural networks).