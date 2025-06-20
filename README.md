# Flappy Bird AI - Multiple Approaches

This project implements different AI approaches to play Flappy Bird, from traditional Q-Learning to modern Deep Q-Learning (DQN).

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

#### 2. Deep Q-Learning (DQN)
```bash
python trains/dQLearning/dqn_train.py
```

#### 3. Multi-AI Training
```bash
python trains/qLearning/multi_ai_train.py
```

### Testing Trained AIs

#### Test Q-Table AI
```bash
python tests/test_ai.py
```

#### Test DQN AI
```bash
python tests/test_dqn.py
```

## 📁 Project Structure

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
│   ├── README.md         # Main documentation
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
└── README.md            # This file
```

## 📚 Documentation

- **[Main Documentation](readme/README.md)** - Complete project overview and usage guide
- **[DQN Implementation Details](readme/README_DQN.md)** - Deep Q-Learning specific documentation

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

## 🎯 Next Steps

1. **Start with Q-Table**: Understand basic RL concepts
2. **Try DQN**: Experience modern deep RL
3. **Experiment**: Tune parameters and try new approaches
4. **Extend**: Add new features or algorithms

Happy training! 🐦🚀 