# Reinforcement Learning (RL) 

Reinforcement Learning (RL) is an exciting and longstanding field of machine learning that gained significant attention in 2013 when DeepMind demonstrated a system capable of learning to play Atari games from scratch using only raw pixels as input. This breakthrough was followed by AlphaGo's historic victories over professional Go players in 2016 and 2017. The key to DeepMind's success was combining deep learning with reinforcement learning, which dramatically advanced the field.


In RL, an agent interacts with an environment by making observations and taking actions, receiving rewards in return. The agent’s goal is to maximize its expected rewards over time. Positive rewards are associated with desirable outcomes, while negative rewards signal undesirable actions. This trial-and-error learning approach is applicable to a wide range of tasks, including:

## Robot control: 
The agent receives rewards for reaching a target location and penalties for going off-course.
## Games: 
In games like Ms. Pac-Man, the agent’s actions are joystick movements, and the rewards are game points.
## Board games: 
The agent gets a reward for winning, such as in Go.
## Smart systems: 
For example, a thermostat may receive rewards for maintaining a target temperature while saving energy.
# Policy Search
In reinforcement learning, the agent's behavior is determined by its policy, which can be represented by any algorithm, including a neural network. The policy could be deterministic or stochastic, where actions are taken with some probability. For example, a robotic vacuum cleaner could have a stochastic policy where it moves forward or randomly rotates based on certain probabilities. This ensures it covers all reachable areas, though the challenge is to optimize the dust collected in a fixed time.

To train the robot, you can tweak policy parameters like movement probability and rotation angle. A simple method is brute-force policy search, testing various parameter combinations to find the best-performing one. For larger policy spaces, genetic algorithms can be used to evolve policies by selecting the best performers and introducing random variations.

Alternatively, optimization techniques like policy gradients (PG) can be employed, where the agent's parameters are adjusted based on the rewards received, gradually improving performance. The goal is to fine-tune the parameters to maximize rewards, and this will be explored using TensorFlow in future steps. To implement this, an environment like OpenAI Gym is needed for the agent to interact with.

# Introduction to OpenAI Gym

OpenAI Gym is a toolkit for developing and comparing reinforcement learning (RL) algorithms, providing a wide range of simulated environments such as Atari games, board games, and physical simulations. It allows you to train RL agents in a controlled, virtual setting, making it ideal for bootstrapping training before deploying in the real world, which can be slow and expensive.

To use Gym, you first need to install it, especially if you're coding on your own machine. You can install the latest version using pip install gym, along with additional dependencies for various environments like classic control tasks, Box2D physics, and Atari games.

Once Gym is set up, you can create environments using gym.make(). For example, the "CartPole-v1" environment simulates a cart balancing a pole. The agent can accelerate the cart left or right to keep the pole upright. After creating the environment, you initialize it with reset(), which returns the first observation (such as the cart's position and the pole's angle).

You can render the environment using render(), which provides an image of the simulation. The agent's actions are specified by the action_space, and you can use step() to execute actions and get feedback such as the new observation, reward, and whether the episode is done or truncated.

A simple policy can be implemented where the agent accelerates left or right depending on the pole's tilt. Running this policy over 500 episodes showed that the basic policy performed poorly, with the pole never staying upright for more than 63 steps. This highlights the need for more advanced methods, such as using a neural network to improve the policy.

### Neural Network Policy
The neural network policy maps observations to action probabilities:

Input: Observation space (CartPole has 4 inputs: position, velocity, angle, and angular velocity).
Hidden Layer: 5 neurons with ReLU activation (since the task is simple).
Output Layer: Single neuron with sigmoid activation to predict the probability 
𝑝
p of moving left. The probability of moving right is 
## Policy Gradient Algorithm


### Play multiple episodes:

Collect rewards and gradients for each action.
Store them for later evaluation.
### Compute Action Advantages:
Normalize rewards (to estimate advantages) by subtracting the mean and dividing by the standard deviation.

### Apply Gradients:

If an action's advantage is positive, reinforce it by applying the corresponding gradients.
For negative advantages, reduce the probability of choosing those actions.
# Temporal Difference (TD) Learning and Q-Learning

## Temporal Difference (TD) Learning
TD learning is a reinforcement learning algorithm used to estimate the value of states when the agent lacks knowledge of:
- Transition probabilities \(T(s, a, s')\)
- Rewards \(R(s, a, s')\)

### Key Features
- Updates state value estimates based on observed transitions and rewards during exploration.
- Gradually learns the optimal value of each state, assuming the agent acts optimally.

### TD Learning Equation
\[
V_{k+1}(s) \leftarrow V_k(s) + \alpha \cdot \delta_k(s, r, s')
\]
Where:
\[
\delta_k(s, r, s') = r + \gamma \cdot V_k(s') - V_k(s)
\]

---

## Q-Learning
Q-Learning extends TD learning to estimate the value of state-action pairs even when transition probabilities and rewards are unknown.

### Key Features
- Relies on the agent exploring the environment (e.g., via a random policy).
- Updates Q-values for state-action pairs based on observed rewards and expected future rewards.
- Derives the optimal policy by selecting actions with the highest Q-values.

### Q-Learning Equation
\[
Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]
\]
Where:
\[
\max_{a'} Q(s', a') 
\]
assumes the agent will act optimally from the next state onward.

---

## Implementation Highlights
1. **Exploration:**  
   A random policy ensures the agent tries all states and actions.
2. **Q-Value Update:**  
   Running averages are updated iteratively using:
   - **Learning rate (\(\alpha\))** with decay.
   - **Discount factor (\(\gamma\))**.
3. **Convergence:**
   - **Q-Value Iteration** (known transition probabilities) converges in a few iterations.
   - **Q-Learning** (unknown transition probabilities and rewards) requires thousands of iterations to converge.

---

## Comparison
| Feature             | TD Learning                 | Q-Learning                  |
|---------------------|-----------------------------|-----------------------------|
| **Focus**           | State values \(V(s)\)      | State-action values \(Q(s, a)\) |
| **Exploration**     | Random or exploration policy | Random or exploration policy |
| **Policy Type**     | On-policy                  | Off-policy                  |
| **Optimal Policy**  | Derived after convergence  | Derived after convergence   |

### Off-Policy vs On-Policy
- **Off-Policy (Q-Learning):**  
  Trains a policy without directly using it during exploration (random actions are taken).
- **On-Policy (e.g., Policy Gradients):**  
  Trains and executes the same policy during exploration.
# Deep Q-Learning with Exploration Policies

This README provides an overview of Deep Q-Learning with exploration strategies, implementation details, and key observations.

---

## **Exploration Policies**

### **Random vs. ε-Greedy Policy**
1. **Random Policy**:
   - Guarantees full exploration but is inefficient for learning.
   
2. **ε-Greedy Policy**:
   - Balances exploration and exploitation:
     - With probability **ε**, a random action is taken (exploration).
     - With probability **1−ε**, the action with the highest Q-value is chosen (exploitation).
   - **Decaying ε**: Gradually reduce **ε** from a high value (e.g., 1.0) to a small value (e.g., 0.05) to focus more on optimal actions as learning progresses.

### **Exploration Bonuses**
- Encourage less-tried actions by adding a bonus term to Q-values:
  \[
  f(Q, N) = Q + \frac{\kappa}{1 + N}
  \]
  - **N**: Number of times an action has been taken.
  - **κ**: Curiosity factor.

---

## **Deep Q-Learning**

### **Challenge with Large State Spaces**
- Environments like Ms. Pac-Man have massive state spaces, making traditional Q-learning infeasible due to memory and computational constraints.

### **Solution: Deep Q-Networks (DQN)**
1. **Neural Network Approximation**:
   - Use a neural network to predict \( Q(s, a) \) instead of storing Q-values explicitly.
   - The neural network learns parameters **θ** to approximate \( Q(s, a) \).

2. **Target Q-Values**:
   - Based on the Bellman equation:
     \[
     y(s, a) = r + \gamma \cdot \max_{a'} Q_\theta(s', a')
     \]
     - \( y(s, a) \): Target Q-value.
     - \( r \): Immediate reward.
     - \( \gamma \): Discount factor.
     - \( Q_\theta \): Neural network's predicted Q-value.

3. **Loss Function**:
   - Minimize the difference between predicted and target Q-values:
     - **Mean Squared Error (MSE)**
     - **Huber Loss** (for robustness to outliers)

---

## **Implementation Steps**

### **1. Model Design**
Define a neural network with:
- **Input**: State representation.
- **Output**: Q-values for all possible actions.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="elu", input_shape=[4]),  # Input layer
    tf.keras.layers.Dense(32, activation="elu"),                  # Hidden layer
    tf.keras.layers.Dense(2)                                     # Output layer (2 possible actions)
])
```

## 2. Replay Buffer
Store past experiences 

Use a deque for efficient memory management.
## 3. Training Loop
Exploration Policy: Gradually decay ε to balance exploration and exploitation.
Gradient Updates:
Train the network using mini-batches of experiences from the replay buffer.
Minimize loss to improve Q-value predictions.
## 4. Batch Training
Sample a mini-batch of experiences from the replay buffer.
Compute 

Update the network using gradient descent.
## Full Training Loop
### Play an Episode:

Interact with the environment and store experiences in the replay buffer.
### Train the Network:

Periodically train the model using mini-batches after sufficient data is accumulated in the replay buffer.
Challenges and Observations
Learning Curve
### Progress is initially slow due to:
High ε (random exploration).
Limited experiences in the replay buffer.
Learning is often erratic, with improvements and setbacks.
Catastrophic Forgetting
Learned policies in one region may degrade due to updates in other regions.
### Partially addressed using:
Experience Replay: Diverse training data prevents overfitting.
Target Networks: Stabilize training by holding a fixed copy of the network for computing target Q-values.
# Understanding and Enhancing Deep Q-Learning

This document provides insights into common challenges in Deep Q-Learning and solutions to improve stability and performance, highlighting key contributions by DeepMind.

---

## 1. Why Loss is a Poor Performance Indicator

- **Issue:** Loss might decrease, but the agent's behavior can worsen due to overfitting specific regions of the environment.
- **Solution:** Use rewards as a measure of performance. Rewards reflect the agent's actual behavior and offer a better indication of progress.

---

## 2. Variants of Deep Q-Learning

### a. Fixed Q-Value Targets
- **Problem:** Using the same model for predictions and targets creates instability (feedback loop).
- **Solution:**  
  - Introduce a target model (a clone of the online model) to compute targets.
  - Update the target model weights less frequently (e.g., every 10,000 steps).
- **Benefits:** Stabilizes training by dampening feedback loops.

---

### b. Double DQN (DDQN)
- **Problem:** Target networks can overestimate Q-values due to random approximations.
- **Solution:**  
  - Use the online model to select the best action for the next state.
  - Use the target model to estimate the Q-value of the selected action.
- **Benefits:** Reduces overestimation bias and improves performance.

---

### c. Prioritized Experience Replay (PER)
- **Problem:** Uniform sampling from the replay buffer might overlook critical experiences.
- **Solution:**  
  - Assign priorities to experiences based on their TD error:  
    \[
    \delta = \lvert r + \gamma \cdot V(s') - V(s) \rvert
    \]
  - Sample experiences with higher TD error more frequently.
  - Compensate for sampling bias by adjusting training weights:  
    \[
    w = (n \cdot P)^{-\beta}
    \]
- **Benefits:** Focuses training on impactful experiences, leading to faster learning.

---

### d. Dueling DQN
- **Problem:** Learning state values and action advantages separately can improve efficiency.
- **Solution:**  
  - Split the network to estimate:
    - State value: \( V(s) \)
    - Action advantage: \( A(s, a) \)
  - Combine them as:  
    \[
    Q(s, a) = V(s) + A(s, a) - \max(A(s, a))
    \]
- **Benefits:** Separates the value of a state from the relative advantages of actions, leading to more robust learning.

---

## 3. Best Practices for Stable Training

- **Learning Rate:** Use a small learning rate (e.g., 0.00025).
- **Epsilon Decay:** Decrease \( \epsilon \) gradually over a long period (e.g., from 1 to 0.1 over 1 million steps).
- **Replay Buffer Size:** Use a large buffer (e.g., 1 million experiences).
- **Episodes:** Train over many episodes (e.g., 50 million steps for Atari games).

---

## 4. Summary of Contributions by DeepMind

1. Stabilizing training with **Fixed Q-Value Targets**.
2. Reducing overestimation bias using **Double DQN**.
3. Accelerating learning with **Prioritized Experience Replay**.
4. Improving model expressiveness through **Dueling DQN**.

---

This README serves as a concise reference for understanding and implementing advanced techniques in Deep Q-Learning.

# Popular Reinforcement Learning (RL) Algorithms

## 1. AlphaGo and Variants
- **Core Idea**: Combines **Monte Carlo Tree Search (MCTS)** with deep neural networks.
- **Variants**:
  - **AlphaGo Zero**: Uses a single neural network for move selection and state evaluation.
  - **AlphaZero**: Generalizes to Go, chess, and shogi.
  - **MuZero**: Learns without prior game knowledge, improving performance further.

## 2. Actor-Critic Algorithms
- **Core Idea**: Combines **policy gradients** with **Q-learning**.
- **Components**:
  - **Policy Network**: Selects actions.
  - **Critic (DQN)**: Evaluates action values, enabling faster learning.

## 3. Asynchronous Advantage Actor-Critic (A3C)
- **Core Idea**: Multiple agents learn in parallel across different environment instances.
- **Key Features**:
  - Shared learning through a master network.
  - Advantage estimation stabilizes training (instead of using full Q-values).

## 4. Advantage Actor-Critic (A2C)
- **Core Idea**: A synchronous version of A3C.
- **Key Features**:
  - Uses larger batch updates, maximizing GPU efficiency.

## 5. Soft Actor-Critic (SAC)
- **Core Idea**: Encourages exploration by maximizing both rewards and entropy (unpredictability of actions).
- **Benefit**: Higher sample efficiency and faster learning.

## 6. Proximal Policy Optimization (PPO)
- **Core Idea**: Simplified version of **Trust Region Policy Optimization (TRPO)**.
- **Key Feature**: Clips the loss function to prevent destabilizing updates.
- **Notable Use**: Used by **OpenAI Five** in Dota 2 to defeat champions.

## 7. Curiosity-Based Exploration
- **Core Idea**: Intrinsically motivates agents by focusing on curiosity rather than external rewards.
- **Key Idea**: Agents seek surprising, unpredictable outcomes.
- **Application**: Successfully trained agents in video games without external penalties.

## 8. Open-Ended Learning (OEL)
- **Core Idea**: Enables agents to continuously learn new tasks.
- **Example**: **POET Algorithm**:
  - Generates progressively difficult environments (curriculum learning).
  - Agents share knowledge across tasks, enhancing adaptability.





