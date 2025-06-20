# Deep Q-Learning (DQN) Implementation

This document explains the Deep Q-Learning (DQN) implementation for the Flappy Bird AI project.

## 🧠 Overview

The DQN implementation replaces the traditional Q-table approach with a neural network, allowing the AI to handle continuous state spaces and generalize better to unseen situations.

## 🏗️ Architecture

### Neural Network Structure
```python
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
```

### State Representation
The DQN uses continuous state representation:
- Bird Y position
- Bird velocity
- Distance to next pipe
- Gap center Y position
- Bird-gap difference

### Experience Replay Buffer
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
```

## 🚀 Usage

### Training
```bash
python trains/dQLearning/dqn_train.py
```

### Testing
```bash
python tests/test_dqn.py
```

## ⚙️ Configuration

DQN-specific parameters in `config/config.py`:

```python
# DQN-specific parameters
DQN_LEARNING_RATE = 0.001
DQN_DISCOUNT_FACTOR = 0.99
DQN_EPSILON = 0.1
DQN_EPSILON_DECAY = 0.995
DQN_EPSILON_MIN = 0.01
DQN_BATCH_SIZE = 32
DQN_TARGET_UPDATE_FREQ = 1000
DQN_MEMORY_SIZE = 10000
DQN_HIDDEN_SIZE = 128
```

## 🔄 Training Process

1. **State Observation**: Get continuous state from game
2. **Action Selection**: Use epsilon-greedy policy
3. **Experience Storage**: Store (state, action, reward, next_state, done) in replay buffer
4. **Network Update**: Sample batch and update Q-network
5. **Target Update**: Periodically update target network

## 📊 Key Features

### Experience Replay
- Stores experiences in a circular buffer
- Samples random batches for training
- Breaks correlation between consecutive experiences

### Target Network
- Separate network for computing target Q-values
- Updated less frequently than main network
- Provides stable training targets

### Epsilon-Greedy Policy
- Balances exploration vs exploitation
- Epsilon decays over time
- Minimum epsilon maintains some exploration

## 🎯 Advantages over Q-Table

| Feature | Q-Table | DQN |
|---------|---------|-----|
| State Space | Discrete | Continuous |
| Memory Usage | High | Low |
| Generalization | Poor | Excellent |
| Training Stability | Good | Requires careful tuning |
| Scalability | Limited | High |

## 📈 Performance Metrics

The DQN tracks several metrics during training:
- Average loss
- Memory buffer size
- Exploration rate
- Q-value statistics

## 🔧 Hyperparameter Tuning

### Learning Rate
- Start with 0.001
- Reduce if training is unstable
- Increase if learning is too slow

### Batch Size
- Larger batches = more stable gradients
- Smaller batches = more frequent updates
- 32 is a good starting point

### Memory Size
- Larger memory = more diverse experiences
- Smaller memory = faster updates
- 10,000 is typically sufficient

## 🐛 Common Issues

### Training Instability
- Reduce learning rate
- Increase batch size
- Check reward scaling

### Poor Performance
- Increase exploration (higher epsilon)
- Adjust reward function
- Check state normalization

### Memory Issues
- Reduce batch size
- Reduce memory buffer size
- Use gradient clipping

## 📚 References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [Deep Q-Learning Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## 🔮 Future Improvements

- **Double DQN**: Separate action selection and evaluation
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Experience Replay**: Sample important experiences more frequently
- **Multi-step Learning**: Use n-step returns
- **Distributional RL**: Learn full return distribution

## 🎮 Integration

The DQN integrates seamlessly with the existing game engine:
- Uses the same reward system
- Compatible with all game features
- Supports all visualization options
- Maintains the same control scheme

For more information, see the [main documentation](README.md). 