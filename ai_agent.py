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
        self.episode_count = 0
        self.total_updates = 0
        self.avg_q_change = 0.0
        self.max_q_value = 0.0
        self.min_q_value = 0.0
        self.exploration_count = 0
        self.exploitation_count = 0
    
    def get_state(self, bird, pipes):
        if not pipes:
            return (0, 0, 0, 0, 0)
        next_pipe = pipes[0]
        gap_center_y = next_pipe.top_height + PIPE_GAP // 2
        bird_gap_diff = bird.y - gap_center_y
        bird_y = int(bird.y / BIRD_Y_DIVISOR)
        bird_velocity = int(bird.velocity / BIRD_VELOCITY_DIVISOR)
        pipe_x = int(next_pipe.x / PIPE_X_DIVISOR)
        pipe_gap_y = int(gap_center_y / PIPE_GAP_Y_DIVISOR)
        bird_gap_diff_discrete = int(bird_gap_diff / 20)
        return (bird_y, bird_velocity, pipe_x, pipe_gap_y, bird_gap_diff_discrete)
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            self.exploration_count += 1
            return random.randint(0, 1)
        self.exploitation_count += 1
        q_values = self.q_table.get(state, [0, 0])
        if state not in self.q_table:
            return 0
        if q_values[0] == q_values[1]:
            return 0
        return np.argmax(q_values)
    
    def get_smart_action(self, state, bird, pipes):
        if not pipes:
            return 0
        from main import GLOBAL_PIPE_HEATMAP, get_adaptive_gap_center
        next_pipe = pipes[0]
        gap_center_y = get_adaptive_gap_center(next_pipe.top_height)
        upper_pipe_mouth = next_pipe.top_height
        lower_pipe_mouth = SCREEN_HEIGHT - next_pipe.bottom_height - GROUND_HEIGHT
        if bird.y + bird.radius > SCREEN_HEIGHT - GROUND_HEIGHT - 10:
            return 1
        if bird.y - bird.radius < 10:
            return 0
        if bird.y - bird.radius < upper_pipe_mouth:
            return 0
        distance_to_gap = bird.y - gap_center_y
        if abs(distance_to_gap) < 10 and abs(bird.velocity) < 4:
            return 0
        if distance_to_gap < -10:
            return 0
        if distance_to_gap > 50:
            return 1  # Bird is well below the gap center, flap to move up
        predicted_y = int(bird.y + bird.velocity)
        if next_pipe.x < bird.x + 120 and 0 <= predicted_y < SCREEN_HEIGHT:
            top_hits = GLOBAL_PIPE_HEATMAP['top'][predicted_y]
            bottom_hits = GLOBAL_PIPE_HEATMAP['bottom'][predicted_y]
            danger_threshold = 1
            if predicted_y < gap_center_y and top_hits >= danger_threshold:
                return 1
            if predicted_y > gap_center_y and bottom_hits >= danger_threshold:
                return 0
        if bird.y > gap_center_y + 10 and next_pipe.x < bird.x + 120:
            return 1
        if bird.y > gap_center_y and bird.velocity > 4 and next_pipe.x < bird.x + 150:
            return 1
        return self.get_action(state)
    
    def update_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    
    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0]
        old_q = self.q_table[state][action]
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        self.total_updates += 1
        q_change = abs(new_q - old_q)
        self.avg_q_change = (self.avg_q_change * (self.total_updates - 1) + q_change) / self.total_updates
        all_q_values = [q for state_qs in self.q_table.values() for q in state_qs]
        if all_q_values:
            self.max_q_value = max(self.max_q_value, max(all_q_values))
            self.min_q_value = min(self.min_q_value, min(all_q_values))
    
    def get_learning_stats(self):
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
        self.episode_count += 1
        self.update_epsilon()
        if self.episode_count % 1000 == 0:
            stats = self.get_learning_stats()
            print(f"Generation {self.episode_count}:")
            print(f"  Epsilon: {stats['epsilon']:.4f}")
            print(f"  Q-table size: {stats['q_table_size']} states")
            print(f"  Avg Q-change: {stats['avg_q_change']:.4f}")
            print(f"  Q-value range: [{stats['min_q_value']:.2f}, {stats['max_q_value']:.2f}]")
            print(f"  Exploration rate: {stats['exploration_rate']:.2%}")
    
    def save_q_table(self, filename=Q_TABLE_FILE):
        serializable_q_table = {str(k): v for k, v in self.q_table.items()}
        with open(filename, 'w') as f:
            json.dump(serializable_q_table, f)
    
    def load_q_table(self, filename=Q_TABLE_FILE):
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    loaded_q_table = json.load(f)
                    self.q_table = {eval(k): v for k, v in loaded_q_table.items()}
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