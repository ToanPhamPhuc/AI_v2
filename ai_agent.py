import numpy as np
import random
import json
import os
from config import *

class FlappyBirdAI:
    def __init__(self, learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR, epsilon=EPSILON):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.q_table = {}
        self.state_size = 4  # bird_y, bird_velocity, pipe_x, pipe_gap_y
        self.action_size = 2  # flap or don't flap
        self.episode_count = 0
        
    def get_state(self, bird, pipes):
        """Convert game state to discrete state for Q-learning"""
        if not pipes:
            return (0, 0, 0, 0)
        
        # Get the next pipe
        next_pipe = pipes[0]
        
        # Normalize and discretize state values
        bird_y = int(bird.y / BIRD_Y_DIVISOR)  # Discretize bird position
        bird_velocity = int(bird.velocity / BIRD_VELOCITY_DIVISOR)  # Discretize velocity
        pipe_x = int(next_pipe.x / PIPE_X_DIVISOR)  # Discretize pipe distance
        pipe_gap_y = int((next_pipe.top_height + PIPE_GAP/2) / PIPE_GAP_Y_DIVISOR)  # Discretize gap center
        
        return (bird_y, bird_velocity, pipe_x, pipe_gap_y)
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, 1)  # Random action
        
        # Get Q-values for current state
        q_values = self.q_table.get(state, [0, 0])
        return np.argmax(q_values)  # Best action
    
    def update_epsilon(self):
        """Decay epsilon over time to reduce exploration"""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning algorithm"""
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0]
        
        # Q-learning update formula
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def end_episode(self):
        """Called at the end of each episode to update learning parameters"""
        self.episode_count += 1
        self.update_epsilon()
        
        # Print epsilon every 1000 episodes
        if self.episode_count % 1000 == 0:
            print(f"Episode {self.episode_count}, Epsilon: {self.epsilon:.4f}")
    
    def save_q_table(self, filename=Q_TABLE_FILE):
        """Save Q-table to file"""
        # Convert tuple keys to strings for JSON serialization
        serializable_q_table = {str(k): v for k, v in self.q_table.items()}
        with open(filename, 'w') as f:
            json.dump(serializable_q_table, f)
    
    def load_q_table(self, filename=Q_TABLE_FILE):
        """Load Q-table from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    loaded_q_table = json.load(f)
                    # Convert string keys back to tuples
                    self.q_table = {eval(k): v for k, v in loaded_q_table.items()}
                    print(f"Loaded Q-table with {len(self.q_table)} states")
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                print(f"Error loading Q-table: {e}. Starting with empty Q-table.")
                self.q_table = {}
        else:
            print("No existing Q-table found. Starting with empty Q-table.")
            self.q_table = {} 