from main import *

class RewardSystem:
    def __init__(self):
        self.last_score = 0
        self.last_bird_y = 0
        self.survival_reward = 1.0
        self.score_reward = 10.0
        self.death_penalty = -100.0
        self.height_penalty = -0.1
        
    def calculate_reward(self, bird, pipes, score, game_over):
        """Calculate reward based on current game state"""
        reward = 0
        
        # Death penalty
        if game_over:
            reward += self.death_penalty
            return reward
        
        # Score reward
        if score > self.last_score:
            reward += self.score_reward
            self.last_score = score
        
        # Survival reward
        reward += self.survival_reward
        
        # Height penalty (encourage staying in middle)
        center_y = SCREEN_HEIGHT // 2
        distance_from_center = abs(bird.y - center_y)
        if distance_from_center > 100:
            reward += self.height_penalty
        
        # Pipe proximity reward/penalty
        if pipes:
            next_pipe = pipes[0]
            pipe_center = next_pipe.top_height + PIPE_GAP // 2
            distance_to_pipe_center = abs(bird.y - pipe_center)
            
            # Reward for staying close to pipe center
            if distance_to_pipe_center < 50:
                reward += 2.0
            elif distance_to_pipe_center > 100:
                reward -= 1.0
        
        return reward
    
    def reset(self):
        """Reset reward system for new game"""
        self.last_score = 0
        self.last_bird_y = 0 