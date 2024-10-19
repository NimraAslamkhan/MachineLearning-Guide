## California Housing Prices Model(project )
Welcome to the Machine Learning Housing Corporation! Your first task is to use California census data to build a model of housing prices in the state. This data includes metrics such as the population, median income, and median housing price for each block group in California. Block groups are the smallest geographical unit for which the US Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). I will call them “districts” for short. Your model should learn from this data and be able to predict the median housing price in any district, given all the other metrics.

**TIP**  
Since you are a well-organized data scientist, the first thing you should do is pull out your machine learning project checklist. You can start with the one in Appendix A; it should work reasonably well for most machine learning projects, but make sure to adapt it to your needs. In this chapter we will go through many checklist items, but we will also skip a few, either because they are self-explanatory or because they will be discussed in later chapters.

## Frame the Problem
The first question to ask your boss is what exactly the business objective is. Building a model is probably not the end goal. How does the company expect to use and benefit from this model? Knowing the objective is important because it will determine how you frame the problem, which algorithms you will select, which performance measure you will use to evaluate your model, and how much effort you will spend tweaking it.

Your boss answers that your model’s output (a prediction of a district’s median housing price) will be fed to another machine learning system, along with many other signals. This downstream system will determine whether it is worth investing in a given area. Getting this right is critical, as it directly affects revenue.

The next question to ask your boss is what the current solution looks like (if any). The current situation will often give you a reference for performance, as well as insights on how to solve the problem. Your boss answers that the district housing prices are currently estimated manually by experts: a team gathers up-to-date information about a district, and when they cannot get the median housing price, they estimate it using complex rules.

This is costly and time-consuming, and their estimates are not great; in cases where they manage to find out the actual median housing price, they often realize that their estimates were off by more than 30%. This is why the company thinks that it would be useful to train a model to predict a district’s median housing price, given other data about that district. The census data looks like a great dataset to exploit for this purpose, since it includes the median housing prices of thousands of districts, as well as other data.

## PIPELINES
A sequence of data processing components is called a data pipeline. Pipelines are very common in machine learning systems, since there is a lot of data to manipulate and many data transformations to apply. Components typically run asynchronously. Each component pulls in a large amount of data, processes it, and spits out the result in another data store. Then, some time later, the next component in the pipeline pulls in this data and spits out its own output. 

Each component is fairly self-contained: the interface between components is simply the data store. This makes the system simple to grasp (with the help of a data flow graph), and different teams can focus on different components. Moreover, if a component breaks down, the downstream components can often continue to run normally (at least for a while) by just using the last output from the broken component. This makes the architecture quite robust. On the other hand, a broken component can go unnoticed for some time if proper monitoring is not implemented. The data gets stale and the overall system’s performance drops.

With all this information, you are now ready to start designing your system. First, determine what kind of training supervision the model will need: is it a supervised, unsupervised, semi-supervised, self-supervised, or reinforcement learning task? And is it a classification task, a regression task, or something else? Should you use batch learning or online learning techniques? Before you read on, pause and try to answer these questions for yourself.

Have you found the answers? Let’s see. This is clearly a typical supervised learning task, since the model can be trained with labeled examples (each instance comes with the expected output, i.e., the district’s median housing price). It is a typical regression task, since the model will be asked to predict a value. More specifically, this is a multiple regression problem, since the system will use multiple features to make a prediction (the district’s population, the median income, etc.). It is also a univariate regression problem, since we are only trying to predict a single value for each district. If we were trying to predict multiple values per district, it would be a multivariate regression problem. Finally, there is no continuous flow of data coming into the system, there is no particular need to adjust to changing data rapidly, and the data is small enough to fit in memory, so plain batch learning should do just fine.

**TIP**  
If the data were huge, you could either split your batch learning work across multiple servers (using the MapReduce technique) or use an online learning technique.

# Machine Learning Model Deployment and Maintenance

## Overview
This document outlines the crucial steps and considerations for deploying and maintaining machine learning (ML) models in a production environment.

## Model Deployment Strategies

1. **Web Application Integration**:
   - For example, the model can be integrated into a website: the user types in some data about a new district and clicks the "Estimate Price" button. This sends a query containing the data to the web server, which forwards it to your web application, and finally your code calls the model’s `predict()` method.
   - It's important to load the model upon server startup rather than every time it is used.

2. **Dedicated Web Services**:
   - Alternatively, you can wrap the model within a dedicated web service that your web application can query through a REST API.
   - This approach makes it easier to upgrade your model to new versions without interrupting the main application.
   - It also simplifies scaling, as you can start as many web services as needed and load-balance the requests coming from your web application across these web services.
   - Additionally, this allows your web application to use any programming language, not just Python.

3. **Cloud Deployment (e.g., Google Vertex AI)**:
   - Another popular strategy is to deploy your model to the cloud, such as Google’s Vertex AI (formerly known as Google Cloud AI Platform and Google Cloud ML Engine).
   - Save your model using `joblib` and upload it to Google Cloud Storage (GCS), then create a new model version pointing to the GCS file in Vertex AI.
   - This setup provides a simple web service that manages load balancing and scaling for you.
   - It accepts JSON requests with input data (e.g., of a district) and returns JSON responses containing the predictions.
   - Deploying TensorFlow models on Vertex AI is similar to deploying Scikit-Learn models.

## Performance Monitoring

- However, deployment is not the end of the story. You need to write monitoring code to check your system’s live performance at regular intervals and trigger alerts when it drops.
- Performance may decline quickly if a component breaks in your infrastructure, or it could decay slowly, going unnoticed for a long time. This is often due to **model rot:** if the model was trained with last year’s data, it may not fit today’s data.

### Monitoring Approaches

1. **Infer Model Performance**:
   - In some cases, you may infer your model’s performance from downstream metrics.
   - For instance, if your model is part of a recommender system suggesting products, you can monitor the sales of recommended products daily.
   - A drop in these numbers (compared to non-recommended products) may indicate an issue with the model or data pipeline.

2. **Human Analysis**:
   - In other scenarios, human analysis may be necessary to assess model performance.
   - For example, with an image classification model detecting product defects on a production line, sending samples of classified pictures (especially the uncertain ones) to human raters can alert you to performance drops before defective products are shipped.
   - Depending on the task, raters could be experts or nonspecialists, such as crowd-workers (e.g., Amazon Mechanical Turk). In some cases, it could even be the users responding through surveys or repurposed captchas.

3. **Implement Monitoring Systems**:
   - Establish a monitoring system (with or without human raters) to evaluate the live model and create processes to manage failures.

4. **Regular Updates**:
   - If your data evolves, you’ll need to update datasets and retrain models regularly.
   - Automate the entire process as much as possible:
     - Collect fresh data and label it (e.g., using human raters).
     - Write scripts for model training and hyperparameter tuning that run automatically at defined intervals (daily or weekly).
     - Create another script to evaluate both the new and previous models on the updated test set, deploying the new model if performance has not decreased.

### Input Data Quality

- Assessing the quality of your model’s input data is essential.
- Performance might degrade slightly due to poor-quality signals (e.g., malfunctioning sensors) or stale outputs from other teams, but it may take time for noticeable performance issues to trigger alerts.
- Monitor input data continually to catch issues early; for example, alert if features are increasingly missing, if the mean or standard deviation drifts too far from the training set, or if new categories appear in categorical features.

### Backup Processes

- Ensure you maintain backups of every model created to quickly roll back to a previous model if the new version fails significantly.
- Keep backups of every version of your datasets too, allowing you to return to a prior dataset if the new one proves corrupted or full of outliers.
- These backups enable easy comparison of new models with previous versions and allow evaluation against any previous dataset.

## MLOps and Infrastructure

- As discussed, machine learning projects demand considerable infrastructure.
- This extensive aspect is known as **ML Operations (MLOps),** which deserves detailed study.
- Initial projects may take substantial effort and time to develop and deploy to production; however, once infrastructure is established, transitioning from idea to production becomes much faster.

## Try It Out!

- Hopefully, this chapter has provided insight into what a machine learning project entails and introduced some tools for developing effective systems.
- Much of the effort lies in data preparation: building monitoring tools, setting up human evaluation pipelines, and automating regular model training. While knowledge of machine learning algorithms is essential, it’s often better to master a selection of three or four algorithms than to explore every advanced option.

### Get Started!

- If you haven't already, now’s a great time to pick up a laptop, select an interesting dataset, and try to navigate the entire process from A to Z.
- A good starting point is competition websites like Kaggle, which offer datasets, clear goals, and community support for a shared experience. Have fun!

## Exercises

1. **Support Vector Machine (SVM) Regressor**:
   - Try an SVM regressor (`sklearn.svm.SVR`) with different hyperparameters (e.g., `kernel="linear"` with various values for the C hyperparameter, or `kernel="rbf"` with different values for C and gamma).
   - Due to SVMs not scaling well to large datasets, train your model on a limited number of instances and use only 3-fold cross-validation.
   - Evaluate the performance of the best SVR predictor.

2. **Randomized Search**:
   - Replace `GridSearchCV` with `RandomizedSearchCV`.

3. **Feature Selection**:
   - Add a `SelectFromModel` transformer to the preparation pipeline to select only the most important attributes.

4. **Custom Transformer**:
   - Create a custom transformer that fits a K-nearest neighbors regressor (`sklearn.neighbors.KNeighborsRegressor`) in its `fit()` method and outputs predictions in its `transform()` method, incorporating latitude and longitude as inputs.
   - This effectively introduces a feature in the model corresponding to the housing median price of the nearest districts.

5. **Preparation Exploration**:
   - Automatically explore preparation options using `GridSearchCV`.

6. **Custom Scaler Implementation**:
   - Reimplement the `StandardScalerClone` class from scratch.
   - Add support for the `inverse_transform()` method to ensure that `scaler.inverse_transform(scaler.fit_transform(X))` returns an array close to X.
   - Also, add support for feature names: set `feature_names_in_` in the `fit()` method if the input is a DataFrame, and implement the `get_feature_names_out()` method.
