from config import *

class RewardSystem:
    def __init__(self):
        self.last_score = 0
        self.last_bird_y = 0
        self.survival_reward = SURVIVAL_REWARD
        self.score_reward = SCORE_REWARD
        self.death_penalty = DEATH_PENALTY
        self.height_penalty = HEIGHT_PENALTY
        self.flap_penalty = FLAP_PENALTY
        
    def calculate_reward(self, bird, pipes, score, game_over, action=None):
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
                reward += PIPE_PROXIMITY_REWARD
            elif distance_to_pipe_center > 100:
                reward += PIPE_DISTANCE_PENALTY
        
        # Flap penalty (discourage unnecessary flapping)
        if action == 1:  # If the AI chose to flap
            reward += self.flap_penalty
        
        return reward
    
    def reset(self):
        """Reset reward system for new game"""
        self.last_score = 0
        self.last_bird_y = 0 