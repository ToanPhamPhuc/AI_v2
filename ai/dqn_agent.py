import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import json
import os
from collections import deque
from config.config import *

class DQN(nn.Module):
    """Deep Q-Network for Flappy Bird"""
    def __init__(self, input_size, hidden_size=DQN_HIDDEN_SIZE, output_size=2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights for better training
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ExperienceReplayBuffer:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity=DQN_MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states), 
                torch.LongTensor(actions), 
                torch.FloatTensor(rewards), 
                torch.FloatTensor(next_states), 
                torch.BoolTensor(dones))
    
    def __len__(self):
        return len(self.buffer)

class FlappyBirdDQN:
    """Deep Q-Learning agent for Flappy Bird"""
    def __init__(self, learning_rate=DQN_LEARNING_RATE, discount_factor=DQN_DISCOUNT_FACTOR, 
                 epsilon=DQN_EPSILON, epsilon_decay=DQN_EPSILON_DECAY, epsilon_min=DQN_EPSILON_MIN, 
                 batch_size=DQN_BATCH_SIZE, target_update_freq=DQN_TARGET_UPDATE_FREQ, 
                 memory_size=DQN_MEMORY_SIZE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # State and action spaces
        self.input_size = 6  # Continuous state representation
        self.output_size = 2  # 0: wait, 1: flap
        
        # Networks
        self.q_network = DQN(self.input_size).to(self.device)
        self.target_network = DQN(self.input_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ExperienceReplayBuffer(memory_size)
        
        # Training stats
        self.episode_count = 0
        self.total_updates = 0
        self.exploration_count = 0
        self.exploitation_count = 0
        self.loss_history = []
        
        # Load existing model if available
        self.load_model()
    
    def get_continuous_state(self, bird, pipes):
        """Get continuous state representation (no discretization)"""
        if not pipes:
            # Default state when no pipes
            return np.array([
                bird.y / SCREEN_HEIGHT,  # Normalized bird Y position
                bird.velocity / 10.0,    # Normalized velocity
                0.0,                     # No pipe distance
                0.5,                     # Center gap position
                0.0,                     # No bird-gap difference
                1.0                      # No pipe flag
            ], dtype=np.float32)
        
        next_pipe = pipes[0]
        gap_center_y = next_pipe.top_height + PIPE_GAP // 2
        
        # Normalize all values to [0, 1] or [-1, 1] range
        bird_y_norm = bird.y / SCREEN_HEIGHT
        velocity_norm = np.clip(bird.velocity / 10.0, -1.0, 1.0)  # Clip velocity
        pipe_distance_norm = np.clip((next_pipe.x - bird.x) / SCREEN_WIDTH, 0.0, 1.0)
        gap_y_norm = gap_center_y / SCREEN_HEIGHT
        bird_gap_diff_norm = np.clip((bird.y - gap_center_y) / (SCREEN_HEIGHT / 2), -1.0, 1.0)
        has_pipe = 0.0  # 0.0 means there is a pipe
        
        return np.array([
            bird_y_norm,
            velocity_norm,
            pipe_distance_norm,
            gap_y_norm,
            bird_gap_diff_norm,
            has_pipe
        ], dtype=np.float32)
    
    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            self.exploration_count += 1
            return random.randint(0, 1)
        
        self.exploitation_count += 1
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def get_smart_action(self, state, bird, pipes):
        """Smart action selection with safety checks"""
        if not pipes:
            return 0
        
        # Safety checks
        if bird.y + bird.radius > SCREEN_HEIGHT - GROUND_HEIGHT - 10:
            return 1  # Flap to avoid ground
        if bird.y - bird.radius < 10:
            return 0  # Don't flap near ceiling
        
        # Use DQN for decision making
        return self.get_action(state)
    
    def update(self):
        """Update the Q-network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network
        if self.total_updates % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.total_updates += 1
        self.loss_history.append(loss.item())
    
    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def end_episode(self):
        """Called at the end of each episode"""
        self.episode_count += 1
        self.update_epsilon()
        
        if self.episode_count % 100 == 0:
            avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0
            total_actions = self.exploration_count + self.exploitation_count
            exploration_rate = self.exploration_count / total_actions if total_actions > 0 else 0
            
            print(f"Episode {self.episode_count}:")
            print(f"  Epsilon: {self.epsilon:.4f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Exploration rate: {exploration_rate:.2%}")
            print(f"  Memory size: {len(self.memory)}")
    
    def get_learning_stats(self):
        """Get current learning statistics"""
        total_actions = self.exploration_count + self.exploitation_count
        exploration_rate = self.exploration_count / total_actions if total_actions > 0 else 0
        avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0
        
        return {
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'avg_loss': avg_loss,
            'exploration_rate': exploration_rate,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count,
            'memory_size': len(self.memory)
        }
    
    def save_model(self, filename=DQN_MODEL_FILE):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_updates': self.total_updates,
            'loss_history': self.loss_history,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename=DQN_MODEL_FILE):
        """Load a trained model"""
        if os.path.exists(filename):
            try:
                checkpoint = torch.load(filename, map_location=self.device)
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.episode_count = checkpoint.get('episode_count', 0)
                self.total_updates = checkpoint.get('total_updates', 0)
                self.loss_history = checkpoint.get('loss_history', [])
                self.exploration_count = checkpoint.get('exploration_count', 0)
                self.exploitation_count = checkpoint.get('exploitation_count', 0)
                print(f"Model loaded from {filename}")
            except Exception as e:
                print(f"Error loading model: {e}. Starting with fresh model.")
        else:
            print("No existing model found. Starting with fresh model.")
    
    def get_q_values(self, state):
        """Get Q-values for a given state (for visualization)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy()[0] 