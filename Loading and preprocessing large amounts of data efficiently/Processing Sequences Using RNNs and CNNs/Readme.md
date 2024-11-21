# Processing Sequences with RNNs and CNNs

This chapter explores how **Recurrent Neural Networks (RNNs)** and **Convolutional Neural Networks (CNNs)** can process sequences and predict future patterns in time series data. RNNs excel at handling sequences of arbitrary lengths, making them useful for tasks such as time series forecasting and natural language processing (NLP) applications like automatic translation and speech-to-text.

## Key Topics

### RNN Fundamentals
- RNNs analyze sequential data, such as:
  - Daily active users on a website.
  - Hourly temperatures.
  - Home power consumption.
  - Vehicle trajectories.
- They learn patterns from past data to forecast future trends, assuming patterns persist.

### Applications of RNNs
- **Time Series Forecasting**
- **Sentence and Document Processing**
- **Audio Analysis**

### Training RNNs
- RNNs are trained using **Backpropagation Through Time (BPTT)**.
- They are compared to traditional models like **ARMA (AutoRegressive Moving Average)** to measure performance.

### Challenges and Solutions
#### Unstable Gradients
- Techniques to stabilize training:
  - **Recurrent Dropout**
  - **Layer Normalization**

#### Limited Memory
- Memory extended using advanced architectures:
  - **LSTM (Long Short-Term Memory) cells**
  - **GRU (Gated Recurrent Unit) cells**

### Alternative Approaches
- **Dense Networks**: Work well for small sequences.
- **Convolutional Neural Networks (CNNs)**: Handle very long sequences effectively (e.g., audio samples, text).

### WaveNet
- A **CNN architecture** designed for processing long sequences.
- Capable of handling tens of thousands of time steps.

- # Recurrent Neurons and Layers

This section provides an overview of Recurrent Neural Networks (RNNs), highlighting their ability to handle sequential data by maintaining memory through recurrent connections.

## Key Concepts

- **Recurrent Connections**: Unlike feedforward networks, RNNs include loopback connections, enabling the network to remember previous inputs.
- **Unrolling Through Time**: RNNs are visualized over multiple time steps, where each neuron processes both the current input and the previous time step's output.

---

## Mathematical Representation

### Single Recurrent Neuron
The output of a single recurrent neuron at time step `t` is computed as:

 The output of a single recurrent neuron at time step `t` is computed as:

$$
\hat{y}(t) = \phi(W_x^T x(t) + W_y^T \hat{y}(t-1) + b)
$$

Where:
- \( x(t) \): Input vector at time \( t \).
- \( \hat{y}(t) \): Output at time \( t \).
- \( W_x, W_y \): Weight matrices for the inputs and previous outputs, respectively.
- \( b \): Bias vector.
- \( \phi \): Activation function (e.g., ReLU).

- **x(t)**: Input vector at time `t`.
- **ŷ(t)**: Output at time `t`.
- **Wₓ, Wᵧ**: Weight matrices for the inputs and previous outputs, respectively.
- **b**: Bias vector.
- **ϕ**: Activation function (e.g., ReLU).
- The output of a single recurrent neuron at time step `t` is computed as:

ŷ(t) = ϕ(Wₓᵀ x(t) + Wᵧᵀ ŷ(t-1) + b)

Where:
- x(t): Input vector at time t.
- ŷ(t): Output at time t.
- Wₓ, Wᵧ: Weight matrices for the inputs and previous outputs, respectively.
- b: Bias vector.
- ϕ: Activation function (e.g., ReLU).


### Batch Processing
For a mini-batch, the outputs for all instances are computed as:

 
- **X(t)**: Input matrix for all instances at time `t`.
- **Ŷ(t)**: Output matrix for all instances at time `t`.

---

## Memory Cells

- **State Preservation**: A recurrent neuron or layer acts as a memory cell, maintaining state across time steps.
- **Learning Dependencies**:
  - **Basic Cells**: Short-term patterns (e.g., ~10 steps).
  - **Advanced Cells**: Long-term patterns using LSTM or GRU.

---

## Input and Output Sequences

RNNs can handle various configurations of input and output sequences:

### Sequence-to-Sequence
- **Input**: Sequence of `N` time steps.
- **Output**: Corresponding sequence shifted by one time step.
- **Example**: Predict daily power consumption for the next day.

### Sequence-to-Vector
- **Input**: Sequence of time steps.
- **Output**: Single value.
- **Example**: Sentiment analysis of a movie review.

### Vector-to-Sequence
- **Input**: Single vector repeated across time steps.
- **Output**: Sequence of values.
- **Example**: Generating a caption for an image.

### Encoder-Decoder
- **Structure**: Combines a sequence-to-vector encoder with a vector-to-sequence decoder.
- **Example**: Machine translation, where:
  - The encoder represents a sentence as a single vector.
  - The decoder generates the translated sentence.

---

## Advantages of Encoder-Decoder Architecture

- **Full Context**: Captures the complete context of input sequences before generating output.
- **Effectiveness**: Outperforms direct sequence-to-sequence RNNs for tasks where later inputs affect earlier outputs.

# RNN Normalization Techniques and Memory Handling

This document provides an overview of Batch Normalization (BN) and Layer Normalization (LN) in Recurrent Neural Networks (RNNs), their limitations, and solutions for handling long-term memory challenges using LSTM and GRU cells.

---

## **Batch Normalization (BN) vs. Layer Normalization (LN) in RNNs**

### **Batch Normalization (BN)**
**Limitations in RNNs:**
1. Inefficient for normalization within time steps.
2. Suboptimal results when applied across time steps (horizontally).
3. Slows down training when applied vertically (between recurrent layers), although it marginally improves performance compared to no normalization.

---

### **Layer Normalization (LN)**
**Advantages in RNNs:**
1. Normalizes across the feature dimension instead of the batch dimension.
2. Works consistently during both training and testing, independent of batch statistics.
3. More suitable for RNNs as it computes statistics for each time step independently.

**Implementation:**
- LN is applied after the linear combination of inputs and hidden states in RNNs.
- Integrated via custom RNN cells, e.g., `LNSimpleRNNCell` in Keras.

---

## **Handling RNN Challenges**

### **1. Unstable Gradients**
- **Solution:** Apply Layer Normalization (LN) and dropout (both input and recurrent) to mitigate unstable gradient issues.

### **2. Short-Term Memory Problem**
- **Problem:** RNNs tend to lose information as it propagates through time steps, leading to poor memory retention for earlier inputs.

---

## **Long-Term Memory Solutions**

### **LSTM Cells**
**Concept:**
- Incorporates a long-term state (`c`) and a short-term state (`h`) for better information management.
- Uses three gates to control the flow of information:
  - **Forget Gate:** Decides what to erase from the long-term state.
  - **Input Gate:** Determines what information to add to the long-term state.
  - **Output Gate:** Filters and outputs parts of the long-term state.

**Benefits:**
- Captures long-term dependencies effectively in time series, text, and audio data.
- Selectively stores and erases information to mitigate short-term memory loss.

#### **LSTM Cell Equations**
1. **Input Gate:**
   \[
   i(t) = \sigma(W_{xi}^T x(t) + W_{hi}^T h(t-1) + b_i)
   \]
2. **Forget Gate:**
   \[
   f(t) = \sigma(W_{xf}^T x(t) + W_{hf}^T h(t-1) + b_f)
   \]
3. **Output Gate:**
   \[
   o(t) = \sigma(W_{xo}^T x(t) + W_{ho}^T h(t-1) + b_o)
   \]
4. **Cell State:**
   \[
   g(t) = \tanh(W_{xg}^T x(t) + W_{hg}^T h(t-1) + b_g)
   \]
   \[
   c(t) = f(t) \odot c(t-1) + i(t) \odot g(t)
   \]
5. **Hidden State (Output):**
   \[
   h(t) = o(t) \odot \tanh(c(t))
   \]

---

### **GRU Cells**
**Concept:**
- Simplified version of LSTMs with fewer gates.
- Combines the forget and input gates into a single **update gate** (`z`).
- Adds a **reset gate** (`r`) to control the influence of the previous state.
- Outputs the entire state at each step, as there is no explicit output gate.

#### **GRU Cell Equations**
1. **Update Gate:**
   \[
   z(t) = \sigma(W_{xz}^T x(t) + W_{hz}^T h(t-1) + b_z)
   \]
2. **Reset Gate:**
   \[
   r(t) = \sigma(W_{xr}^T x(t) + W_{hr}^T h(t-1) + b_r)
   \]
3. **Candidate Activation:**
   \[
   g(t) = \tanh(W_{xg}^T x(t) + W_{hg}^T (r(t) \odot h(t-1)) + b_g)
   \]
4. **Hidden State (Output):**
   \[
   h(t) = z(t) \odot h(t-1) + (1 - z(t)) \odot g(t)
   \]

---

## **Conclusion**
- Use **Layer Normalization (LN)** for better consistency and efficiency in RNNs.
- Choose **LSTM** or **GRU** depending on the complexity and data requirements:
  - **LSTM:** For capturing long-term dependencies with more control.
  - **GRU:** For simpler architectures with comparable performance.


