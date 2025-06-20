# Deep Q-Learning (DQN) for Flappy Bird

This implementation replaces the traditional Q-table approach with a Deep Q-Network (DQN) that uses a neural network to approximate Q-values, enabling better handling of complex state spaces.

## 🧠 What is Deep Q-Learning?

Deep Q-Learning (DQN) is a reinforcement learning algorithm that combines Q-learning with deep learning to handle complex environments. Instead of a Q-table, which becomes unwieldy for large state spaces, DQN uses a neural network to approximate the Q-values, enabling it to learn optimal actions in complex scenarios.

### Key Advantages over Q-Table Approach:

1. **Continuous State Space**: No need for state discretization
2. **Better Generalization**: Neural network can generalize to unseen states
3. **Memory Efficiency**: No need to store a large Q-table
4. **Scalability**: Can handle much larger state spaces
5. **Experience Replay**: Improves learning stability and efficiency

## 🏗️ Architecture

### Neural Network Structure
```
Input Layer (6 neurons) → Hidden Layer 1 (128 neurons) → Hidden Layer 2 (128 neurons) → Hidden Layer 3 (64 neurons) → Output Layer (2 neurons)
```

### State Representation (Continuous)
The DQN uses a 6-dimensional continuous state vector:
- `bird_y_norm`: Normalized bird Y position (0-1)
- `velocity_norm`: Normalized bird velocity (-1 to 1)
- `pipe_distance_norm`: Normalized distance to next pipe (0-1)
- `gap_y_norm`: Normalized gap center Y position (0-1)
- `bird_gap_diff_norm`: Normalized difference between bird and gap center (-1 to 1)
- `has_pipe`: Binary flag indicating if there's a pipe (0 or 1)

### Experience Replay Buffer
- Stores (state, action, reward, next_state, done) tuples
- Random sampling for training to break correlations
- Configurable buffer size (default: 10,000 experiences)

### Target Network
- Separate network for computing target Q-values
- Updated periodically to improve training stability
- Reduces overestimation bias

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training the DQN
```bash
python dqn_train.py
```

### Testing the Trained DQN
```bash
python test_dqn.py
```

## 🎮 Controls

During training:
- **Q**: Quit training
- **A**: Toggle coordinate axes
- **H**: Toggle hitboxes
- **C**: Toggle collision zones
- **B**: Clear pipe heatmap
- **G**: Toggle gap distance display

## ⚙️ Configuration

DQN parameters can be adjusted in `config/config.py`:

```python
# DQN-specific parameters
DQN_LEARNING_RATE = 0.001          # Neural network learning rate
DQN_DISCOUNT_FACTOR = 0.99         # Future reward discount
DQN_EPSILON = 0.1                  # Initial exploration rate
DQN_EPSILON_DECAY = 0.995          # Exploration decay rate
DQN_EPSILON_MIN = 0.01             # Minimum exploration rate
DQN_BATCH_SIZE = 32                # Training batch size
DQN_TARGET_UPDATE_FREQ = 1000      # Target network update frequency
DQN_MEMORY_SIZE = 10000            # Experience replay buffer size
DQN_HIDDEN_SIZE = 128              # Hidden layer size
```

## 📊 Training Progress

The training script displays:
- **Generation**: Current training episode
- **Score**: Current game score
- **Best**: Best score achieved so far
- **High**: All-time high score
- **Avg**: Average score over last 100 episodes
- **ε (Epsilon)**: Current exploration rate
- **Loss**: Average training loss over last 100 updates
- **Memory**: Number of experiences in replay buffer

## 💾 Model Persistence

The DQN automatically saves:
- **Model weights**: Neural network parameters
- **Training state**: Epsilon, episode count, loss history
- **Statistics**: Exploration/exploitation counts

Models are saved as `dqn_model.pth` and automatically loaded on startup.

## 🔍 Visualization Features

### Real-time Q-Values
- Shows Q-values for WAIT and FLAP actions
- Updates in real-time during gameplay
- Helps understand AI decision-making

### Training Statistics
- Loss curve tracking
- Exploration vs exploitation ratio
- Memory buffer utilization

### Visual Debugging
- Hitbox visualization
- Collision zone highlighting
- Gap distance measurements
- Coordinate axes

## 🧪 Comparison with Q-Table Approach

| Feature | Q-Table | DQN |
|---------|---------|-----|
| State Space | Discrete | Continuous |
| Memory Usage | High (grows with states) | Low (fixed network size) |
| Generalization | Poor | Excellent |
| Training Speed | Fast initial, slows down | Consistent |
| Scalability | Limited | High |
| Implementation | Simple | Complex |

## 🎯 Expected Performance

With proper training, the DQN should achieve:
- **Consistent scores**: 50+ pipes regularly
- **Adaptive behavior**: Handles varying pipe positions
- **Smooth flight**: Natural-looking movement patterns
- **Robust performance**: Works well across different scenarios

## 🔧 Troubleshooting

### Common Issues:

1. **Slow Training**: Reduce batch size or learning rate
2. **Poor Performance**: Increase exploration rate or memory size
3. **Unstable Loss**: Reduce learning rate or increase target update frequency
4. **Memory Issues**: Reduce replay buffer size

### Performance Tips:

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster training
2. **Hyperparameter Tuning**: Experiment with learning rates and network sizes
3. **Reward Shaping**: Adjust reward system for better learning signals
4. **Regular Saving**: Models are saved every 100 generations

## 📈 Advanced Features

### Future Enhancements:
- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates value and advantage functions
- **Prioritized Experience Replay**: Focuses on important experiences
- **Multi-step Learning**: Uses n-step returns for better learning

## 🤝 Contributing

Feel free to experiment with:
- Different network architectures
- Alternative state representations
- Novel reward functions
- Advanced DQN variants

The modular design makes it easy to extend and improve the implementation! 