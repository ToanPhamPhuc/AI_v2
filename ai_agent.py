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
        
        # Learning progress tracking
        self.total_updates = 0
        self.avg_q_change = 0.0
        self.max_q_value = 0.0
        self.min_q_value = 0.0
        self.exploration_count = 0
        self.exploitation_count = 0
        
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
        """Choose action using epsilon-greedy policy with smart initialization"""
        if random.random() < self.epsilon:
            self.exploration_count += 1
            return random.randint(0, 1)  # Random action
        
        self.exploitation_count += 1
        
        # Get Q-values for current state
        q_values = self.q_table.get(state, [0, 0])
        
        # If this is a new state, provide some smart guidance
        if state not in self.q_table:
            # For new states, prefer not flapping initially (more conservative)
            # This helps prevent the AI from flapping unnecessarily at the start
            return 0  # Don't flap for new states initially
        
        # If Q-values are equal, prefer not flapping (more conservative)
        if q_values[0] == q_values[1]:
            return 0
        
        return np.argmax(q_values)  # Best action
    
    def get_smart_action(self, state, bird, pipes):
        """Get action with additional smart logic for edge cases and goal-seeking"""
        if not pipes:
            return 0  # No pipes, don't flap
        
        # Get the next pipe
        next_pipe = pipes[0]
        goal_center_y = next_pipe.top_height + PIPE_GAP // 2  # Center of the gap
        
        # Emergency flap if bird is about to hit the ground
        if bird.y + bird.radius > SCREEN_HEIGHT - GROUND_HEIGHT - 20:
            return 1  # Flap to avoid ground
        
        # Don't flap if bird is too high (near ceiling) - less restrictive
        if bird.y < SCREEN_HEIGHT // 6:  # Upper sixth of screen (less restrictive)
            return 0  # Don't flap when very near ceiling
        
        # NEW: Smart goal-seeking logic (less restrictive)
        distance_to_goal = abs(bird.y - goal_center_y)
        
        # If bird is significantly above the goal, don't flap (let gravity work)
        if bird.y < goal_center_y - 60:  # Bird is well above goal (increased threshold)
            return 0  # Don't flap, let bird fall toward goal
        
        # If bird is below the goal and pipe is close, flap to get through
        if bird.y > goal_center_y + 10 and next_pipe.x < bird.x + 200:  # Bird below goal, pipe close (more lenient)
            return 1  # Flap to get through gap
        
        # If bird is very close to goal center, minimal flapping
        if distance_to_goal < 15:  # Bird is very close to goal center (smaller threshold)
            if bird.velocity < -5:  # Bird is moving up fast (increased threshold)
                return 0  # Don't flap, let it settle
            elif bird.velocity > 5:  # Bird is falling fast (increased threshold)
                return 1  # Flap to slow down
        
        # If pipe is very close and bird is too low, flap
        if next_pipe.x < bird.x + 100 and bird.y > next_pipe.top_height + PIPE_GAP - 30:
            return 1  # Flap to get through gap
        
        # Use regular Q-learning action
        return self.get_action(state)
    
    def update_epsilon(self):
        """Decay epsilon over time to reduce exploration"""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using improved Q-learning algorithm"""
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0]
        
        # Store old Q-value for tracking changes
        old_q = self.q_table[state][action]
        
        # Q-learning update formula: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Track learning progress
        self.total_updates += 1
        q_change = abs(new_q - old_q)
        self.avg_q_change = (self.avg_q_change * (self.total_updates - 1) + q_change) / self.total_updates
        
        # Update min/max Q-values
        all_q_values = [q for state_qs in self.q_table.values() for q in state_qs]
        if all_q_values:
            self.max_q_value = max(self.max_q_value, max(all_q_values))
            self.min_q_value = min(self.min_q_value, min(all_q_values))
    
    def get_learning_stats(self):
        """Get comprehensive learning statistics"""
        total_actions = self.exploration_count + self.exploitation_count
        exploration_rate = self.exploration_count / total_actions if total_actions > 0 else 0
        
        return {
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'total_updates': self.total_updates,
            'avg_q_change': self.avg_q_change,
            'max_q_value': self.max_q_value,
            'min_q_value': self.min_q_value,
            'exploration_rate': exploration_rate,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count
        }
    
    def end_episode(self):
        """Called at the end of each generation to update learning parameters"""
        self.episode_count += 1
        self.update_epsilon()
        
        # Print detailed stats every 1000 generations
        if self.episode_count % 1000 == 0:
            stats = self.get_learning_stats()
            print(f"Generation {self.episode_count}:")
            print(f"  Epsilon: {stats['epsilon']:.4f}")
            print(f"  Q-table size: {stats['q_table_size']} states")
            print(f"  Avg Q-change: {stats['avg_q_change']:.4f}")
            print(f"  Q-value range: [{stats['min_q_value']:.2f}, {stats['max_q_value']:.2f}]")
            print(f"  Exploration rate: {stats['exploration_rate']:.2%}")
    
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
                    
                    # Update max/min Q-values from loaded data
                    all_q_values = [q for state_qs in self.q_table.values() for q in state_qs]
                    if all_q_values:
                        self.max_q_value = max(all_q_values)
                        self.min_q_value = min(all_q_values)
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                print(f"Error loading Q-table: {e}. Starting with empty Q-table.")
                self.q_table = {}
        else:
            print("No existing Q-table found. Starting with empty Q-table.")
            self.q_table = {} 