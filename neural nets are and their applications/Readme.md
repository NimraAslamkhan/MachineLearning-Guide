artificial neural networks (ANNs) are inspired by the brain’s structure, with networks of biological neurons serving as the model. However, just as planes don’t need to flap their wings to fly, ANNs have evolved and become quite different from biological neurons. Some researchers even suggest moving away from biological terms like “neurons” to avoid limiting innovation.

ANNs are fundamental to deep learning, and their versatility, power, and scalability make them ideal for complex tasks such as image classification, speech recognition, video recommendations, or even beating champions at the game of Go (as in DeepMind’s AlphaGo).

This chapter introduces ANNs, starting with early architectures and progressing to multilayer perceptrons, which are widely used today. The second part focuses on implementing neural networks using TensorFlow's Keras API, a simple yet flexible tool for building, training, and evaluating neural networks. While Keras is easy to use, it offers enough flexibility for various use cases, and if needed, you can extend its capabilities with lower-level APIs or TensorFlow itself. But first, let's explore the history of artificial neural networks!

Artificial neural networks (ANNs) have a long history, dating back to 1943 when neurophysiologist Warren McCulloch and mathematician Walter Pitts introduced the first ANN model. They presented a simplified version of how biological neurons in animal brains might work together to perform complex tasks using propositional logic. This marked the beginning of ANNs.

In the 1960s, early successes led people to believe we would soon have intelligent machines. However, when progress stalled, funding shifted elsewhere, and ANNs entered a period of stagnation known as the "AI winter." In the 1980s, new architectures and better training methods sparked renewed interest in ANNs, but by the 1990s, other machine learning techniques, like support vector machines, took the spotlight, leaving ANNs in the background again.

Today, we are experiencing a resurgence in ANN research, and there are several reasons to believe that this time, ANNs will have a lasting impact:

Data Availability: There is now an enormous amount of data available to train neural networks, and ANNs often outperform other techniques on complex tasks.
Computing Power: The dramatic increase in computing power, especially with GPUs developed for gaming, has made it possible to train large neural networks in reasonable timeframes. Cloud platforms have also made this power widely accessible.
Improved Algorithms: While the core training algorithms have remained similar to those from the 1990s, small improvements have had a significant positive impact.
Theoretical Limitations: Some theoretical concerns about ANNs, like getting stuck in local optima, have proven to be less problematic than expected, especially with larger networks.
Positive Feedback Loop: The success of ANN-based products has created a cycle of increased funding and progress, leading to even more impressive advancements. 
Biological neurons, commonly found in animal brains, are unique cells with a complex structure. They consist of a cell body that holds the nucleus, numerous short branches called dendrites, and one long branch called the axon. The axon can be relatively short or incredibly long. Near its end, the axon splits into branches with small tips called synapses, which connect to other neurons’ dendrites or cell bodies.

Neurons send electrical signals called action potentials (APs) along their axons. When the signal reaches the synapse, it releases neurotransmitters, chemical signals that may activate or inhibit the next neuron. If a neuron receives enough activating signals within a short time, it fires its own electrical impulses. Despite this simplicity, neurons form large, interconnected networks, with each neuron often connecting to thousands of others. These networks perform complex computations, similar to how the simple actions of ants create intricate anthills.

Biological neural networks (BNNs) are often organized in layers, especially in the brain's outer layer, the cerebral cortex. Such layered structures help perform specific functions. For example:

Identity Function: If neuron A activates, neuron C also activates (since it receives signals from A); if A is off, C is also off.
AND Logic: Neuron C activates only when both neurons A and B are active; one signal alone isn’t enough.
OR Logic: Neuron C activates if either neuron A or B (or both) are active.
NOT Logic: If neuron A is active but neuron B is not, neuron C activates. If A is always active, C becomes active only when B is off.
The Perceptron is a basic type of artificial neural network (ANN) architecture developed by Frank Rosenblatt in 1957. It’s based on a simple unit called a Threshold Logic Unit (TLU) or Linear Threshold Unit (LTU), which operates similarly to a logistic regression model but uses a step function rather than a logistic function for activation.

Here’s how the TLU works:

Linear Combination of Inputs: The TLU takes multiple inputs (each associated with a weight) and computes a linear combination, 
𝑧
=
𝑤
1
𝑥
1
+
𝑤
2
𝑥
2
+
⋯
+
𝑤
𝑛
𝑥
𝑛
+
𝑏
z=w 
1
​
 x 
1
​
 +w 
2
​
 x 
2
​
 +⋯+w 
n
​
 x 
n
​
 +b, where:

𝑥
1
,
𝑥
2
,
…
,
𝑥
𝑛
x 
1
​
 ,x 
2
​
 ,…,x 
n
​
  are the input values.
𝑤
1
,
𝑤
2
,
…
,
𝑤
𝑛
w 
1
​
 ,w 
2
​
 ,…,w 
n
​
  are the corresponding weights.
𝑏
b is the bias term.
Activation Function: The TLU applies a step function to this result:

If 
𝑧
≥
0
z≥0, the output is 1.
If 
𝑧
<
0
z<0, the output is 0 (or -1 if using the sign function).
The Perceptron is useful for binary classification tasks where the goal is to separate data into two classes.

Perceptron Architecture
A single-layer perceptron consists of one or more TLUs connected to all inputs, which form a fully connected or dense layer. When configured with multiple output neurons, the perceptron can handle multilabel classification tasks. It can also be used for multiclass classification.

Training the Perceptron
The Perceptron training algorithm updates the weights to minimize classification errors by applying a modified version of Hebb’s rule (Hebbian learning). Hebbian learning increases the connection strength between neurons that often activate together. In perceptron training:

For each training instance, the perceptron makes a prediction.
If the prediction is incorrect, the algorithm adjusts the weights to correct it.
The weight update rule is:
𝑤
𝑖
,
𝑗
←
𝑤
𝑖
,
𝑗
+
𝜂
(
𝑦
𝑗
−
𝑦
^
𝑗
)
𝑥
𝑖
w 
i,j
​
 ←w 
i,j
​
 +η(y 
j
​
 − 
y
^
​
  
j
​
 )x 
i
​
 
where:
𝑤
𝑖
,
𝑗
w 
i,j
​
  is the weight between the 
𝑖
i-th input and 
𝑗
j-th neuron.
𝜂
η is the learning rate.
𝑦
𝑗
y 
j
​
  is the target output, and 
𝑦
^
𝑗
y
^
​
  
j
​
  is the predicted output.
Rosenblatt proved that if the data is linearly separable, this algorithm will converge (Perceptron Convergence Theorem).

Limitations and Multilayer Perceptron (MLP)
Perceptrons cannot solve non-linearly separable problems like XOR, a limitation shared by other linear models. In 1969, Marvin Minsky and Seymour Papert highlighted this issue, which temporarily reduced interest in neural networks.

However, by stacking multiple perceptrons into layers (forming a Multilayer Perceptron (MLP)), it’s possible to solve non-linear problems. An MLP can, for instance, solve the XOR problem by using two layers of neurons, with hidden layers enabling complex decision boundaries.

MLPs form the basis of more advanced neural networks, capable of complex classification and regression tasks, and serve as a foundational model in modern deep learning.


Multilayer Perceptron (MLP) Overview
An MLP consists of:
One input layer that receives data.
One or more hidden layers with threshold logic units (TLUs).
One output layer of TLUs that gives the prediction.
The term Deep Neural Network (DNN) applies when an MLP has multiple hidden layers.
2. Backpropagation
Backpropagation enables training by calculating the gradient of the error with respect to each model parameter.
The Reverse-Mode Automatic Differentiation algorithm (or reverse-mode autodiff) computes these gradients efficiently in two passes: a forward pass and a backward pass.
David Rumelhart, Geoffrey Hinton, and Ronald Williams popularized backpropagation in 1985, demonstrating its power in training neural networks.
3. Backpropagation Workflow
Step 1: Perform a forward pass for each mini-batch, saving intermediate results for later use.
Step 2: Calculate the network’s output error using a loss function.
Step 3: During the backward pass, propagate error gradients back through each layer using the chain rule, calculating each parameter’s contribution to the error.
Step 4: Use these gradients to perform a gradient descent step, updating the weights and biases.
4. Activation Functions
Activation functions add non-linearity to prevent the network from acting like a single-layer linear model.
Common choices:
Sigmoid function (σ(z) = 1 / (1 + exp(–z))) is S-shaped and useful for binary output.
Hyperbolic tangent function (tanh), with outputs from -1 to 1, helps with faster convergence by centering outputs around 0.
Rectified Linear Unit (ReLU), defined as ReLU(z) = max(0, z), is efficient and effective for many neural networks.
5. MLP for Regression (MLPRegressor)
For regression tasks, MLP can have one or multiple output neurons depending on the target.
Example: Using Scikit-Learn’s MLPRegressor on the California housing dataset.
Preprocessing is essential (e.g., standardizing features).
ReLU activation function for hidden layers and the Adam optimizer for gradient descent.
Output layer typically has no activation function for unrestricted predictions.
If output constraints are needed:
ReLU or Softplus for non-negative outputs.
Sigmoid or tanh for outputs within specific ranges.
MLPRegressor only supports the mean squared error (MSE) loss by default, suitable for typical regression but not for datasets with many outliers.

Using MLPs for Classification
MLPs are highly flexible neural networks that can be adapted for various types of classification tasks:

Binary Classification: For a binary classification, the MLP needs only a single output neuron with a sigmoid activation function, outputting a probability between 0 and 1. This allows the model to predict the likelihood of one class, with the probability of the other class being 1 minus this value.

Multilabel Classification: For tasks where each instance can belong to multiple classes (like predicting if an email is spam or urgent), each label gets its own output neuron with a sigmoid activation function. Each neuron outputs the probability of the respective label, allowing combinations of labels (e.g., spam-urgent, ham-urgent).

Multiclass Classification: When an instance belongs to only one out of multiple classes (like digits from 0 to 9), each class has a dedicated output neuron, and the softmax activation function is used to ensure the output probabilities sum to 1.

The cross-entropy loss function is commonly used for classification tasks, as it optimizes the model’s probability output accuracy.

Implementing MLPs with Keras
Keras offers a high-level API to create and train neural networks. Here’s a simplified example of building an image classifier using Fashion MNIST, a dataset of 28x28 pixel grayscale images representing clothing items:

Load the Dataset: Keras provides built-in functions for loading datasets like Fashion MNIST. This dataset includes 60,000 training images and 10,000 test images, which are grayscale and initially in the integer range [0, 255].

Data Preprocessing:

Normalization: Scale the pixel values to the 0–1 range by dividing by 255.0.
Shape Transformation: Since each image is initially in 2D format (28x28 pixels), a Flatten layer is used to reshape it into a 1D array (size 784) for processing.
Building the Model:

Sequential Model: The simplest way to stack layers, where each layer is sequentially connected to the next.
Layers:
Flatten Layer: Reshapes each input into a 1D array.
Dense Layers:
The first hidden layer has 300 neurons with the ReLU activation function.
The second hidden layer has 100 neurons with ReLU.
The output layer has 10 neurons (one per class) with the softmax activation function.
Model Summary: Keras allows you to print a summary of the model architecture, showing each layer’s name, shape, and number of parameters. Dense layers often have many parameters due to the connection weights and biases.

Training: With the model defined, it can be compiled and trained. Typically, the compile method specifies the optimizer (like adam), loss function (like sparse_categorical_crossentropy for integer class labels), and metrics (like accuracy).

Unique Layer Naming: Each layer is automatically assigned a unique name by Keras, allowing flexibility in merging models without conflicts.

compiling, training, and evaluating a model in Keras, along with guidance on using optimizers, loss functions, and metrics:

Compiling the Model: After creating a model, you must compile it using the compile() method. This step involves specifying the loss function, optimizer, and any additional metrics you want to monitor. For example:

python
Copy code
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)
loss="sparse_categorical_crossentropy": This is suitable for classification problems with sparse labels (where each target is a class index). If you had one-hot encoded labels, you'd use "categorical_crossentropy".
optimizer="sgd": Stochastic Gradient Descent (SGD) is used to optimize the model parameters. You can also use tf.keras.optimizers.SGD().
metrics=["accuracy"]: Measures the model's classification accuracy during training and evaluation.
Loss Function Choice:

Sparse Categorical Crossentropy: Use this for classification tasks with sparse labels.
Binary Crossentropy: Use this for binary classification or multilabel classification with sigmoid activation in the output layer.
Categorical Crossentropy: Use this for multi-class classification with one-hot encoded labels.
Training with fit(): To train the model, call fit() and provide training data, labels, and parameters like the number of epochs and batch size:

python
Copy code
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
This returns a History object, which stores training metrics and loss values for each epoch in history.history. Plotting these values can help visualize training progress and detect overfitting.

Handling Imbalanced Data: If your dataset is imbalanced (some classes have more instances than others), you can use class_weight in fit() to assign higher weights to underrepresented classes. Alternatively, use sample_weight to provide per-instance weights.

Evaluating and Tuning:

evaluate(): Use this method to estimate your model’s generalization on a test set.
Hyperparameter Tuning: If performance isn’t satisfactory, tune hyperparameters such as learning rate, number of layers, neurons, or activation functions. Always adjust the learning rate after changing any hyperparameter.
Making Predictions: After training, you can use the model's predict() method to generate predictions for new data:

python
Copy code
y_proba = model.predict(X_new)
For classification, you can get the predicted class by applying argmax() to obtain the class with the highest probability.

Functional API: For complex model architectures like Wide & Deep networks, use Keras' Functional API. This enables building models with custom topologies, multiple inputs, and outputs by treating layers as functions applied sequentially from input to output.

These are the essential steps for compiling, training, and evaluating neural networks in Keras. You can also experiment with advanced optimizers like Adam, which often improve training efficiency for large datasets or deep models.
