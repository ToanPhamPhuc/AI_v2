# Flappy Bird AI - Complete Documentation

This project implements different AI approaches to play Flappy Bird, from traditional Q-Learning to modern Deep Q-Learning (DQN).

## 🏗️ Project Structure

```
AI_v2/
├── ai/                    # AI implementations
│   ├── ai_agent.py       # Q-Table based AI
│   ├── dqn_agent.py      # Deep Q-Learning AI
│   └── training_loop.py  # Training utilities
├── config/               # Configuration files
│   └── config.py         # Game and AI parameters
├── data/                 # Data storage (organized)
│   ├── scores/           # High score files
│   ├── jsons/            # Q-tables and JSON data
│   └── pth/              # PyTorch model files
├── game/                 # Game engine
│   ├── main.py           # Main game logic
│   └── reward_system.py  # Reward calculation
├── readme/               # Documentation
│   ├── README.md         # Main documentation (this file)
│   └── README_DQN.md     # DQN-specific documentation
├── tests/                # Test scripts
│   ├── test_ai.py        # Test Q-table AI
│   └── test_dqn.py       # Test DQN AI
├── trains/               # Training scripts (organized)
│   ├── qLearning/        # Q-Table training scripts
│   │   ├── continuous_train.py
│   │   ├── multi_ai_train.py
│   │   └── train_ai.py
│   └── dQLearning/       # DQN training scripts
│       └── dqn_train.py
├── flappy-bird/          # Original game assets
├── requirements.txt      # Python dependencies
└── README.md            # Root overview
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training Options

#### 1. Q-Table AI (Traditional)
```bash
python trains/qLearning/continuous_train.py
```
- Uses discrete state space
- Fast initial learning
- Good for understanding RL basics

#### 2. Deep Q-Learning (DQN)
```bash
python trains/dQLearning/dqn_train.py
```
- Uses neural network
- Continuous state space
- Better generalization
- More advanced approach

#### 3. Multi-AI Training
```bash
python trains/qLearning/multi_ai_train.py
```
- Trains multiple AIs simultaneously
- Faster strategy discovery
- Knowledge sharing between AIs

### Testing Trained AIs

#### Test Q-Table AI
```bash
python tests/test_ai.py
```

#### Test DQN AI
```bash
python tests/test_dqn.py
```

## 🎮 Controls

During training:
- **Q**: Quit training
- **A**: Toggle coordinate axes
- **H**: Toggle hitboxes
- **C**: Toggle collision zones
- **B**: Clear pipe heatmap
- **G**: Toggle gap distance display

## 📊 Performance Comparison

| Approach | State Space | Memory Usage | Learning Speed | Generalization |
|----------|-------------|--------------|----------------|----------------|
| Q-Table | Discrete | High | Fast initial | Poor |
| DQN | Continuous | Low | Consistent | Excellent |
| Multi-AI | Discrete | High | Very fast | Good |

## 🔧 Configuration

All parameters are configurable in `config/config.py`:

- **Game settings**: Screen size, physics, pipe gap
- **AI parameters**: Learning rates, exploration, rewards
- **DQN settings**: Network architecture, batch size, memory
- **File paths**: Organized data storage locations

## 📁 Data Organization

- **Scores**: `data/scores/` - High score tracking
- **Models**: `data/pth/` - PyTorch DQN models
- **Q-Tables**: `data/jsons/` - Q-table and JSON data
- **Tests**: `tests/` - AI testing scripts

## 🧪 Testing

Each AI approach has dedicated test scripts that:
- Load trained models
- Run without exploration (pure exploitation)
- Display real-time Q-values and statistics
- Show AI decision-making process

## 📈 Expected Results

With proper training:
- **Q-Table AI**: 20-50 pipes consistently
- **DQN AI**: 50+ pipes regularly
- **Multi-AI**: 30-60 pipes with faster learning

## 🔍 Advanced Features

- **Pipe Heatmap**: Visual collision tracking
- **Adaptive Gap**: Dynamic difficulty adjustment
- **Experience Replay**: DQN memory buffer
- **Target Network**: Stable DQN training
- **Real-time Visualization**: Training progress display

## 🤝 Contributing

Feel free to experiment with:
- Different network architectures
- Alternative reward functions
- Novel training strategies
- Advanced RL algorithms

The modular design makes it easy to extend and improve!

## 📚 Documentation

- [DQN Implementation Details](README_DQN.md)
- [Configuration Guide](../config/config.py)
- [Training Scripts](../trains/)

## 🎯 Next Steps

1. **Start with Q-Table**: Understand basic RL concepts
2. **Try DQN**: Experience modern deep RL
3. **Experiment**: Tune parameters and try new approaches
4. **Extend**: Add new features or algorithms

Happy training! 🐦🚀 