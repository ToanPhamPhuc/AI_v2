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
        self.ceiling_penalty = CEILING_COLLISION_PENALTY
        self.consecutive_flaps = 0  # Track consecutive flapping
        
    def calculate_reward(self, bird, pipes, score, game_over, action=None):
        """Calculate reward based on current game state with improved goal-seeking"""
        reward = 0
        
        # Death penalty
        if game_over:
            reward += self.death_penalty
            return reward
        
        # Ceiling collision penalty
        if bird.hit_ceiling:
            reward += self.ceiling_penalty
        
        # Score reward
        if score > self.last_score:
            reward += self.score_reward
            self.last_score = score
        
        # Survival reward
        reward += self.survival_reward
        
        # Goal-seeking reward (IMPROVED - based on bird-gap difference)
        if pipes:
            next_pipe = pipes[0]
            gap_center_y = next_pipe.top_height + PIPE_GAP // 2
            bird_gap_diff = abs(bird.y - gap_center_y)
            
            # Reward for being close to the gap center
            if bird_gap_diff < 15:  # Very close to gap center
                reward += 2.0  # Strong reward for being in the sweet spot
            elif bird_gap_diff < 30:  # Close to gap center
                reward += 1.0  # Good reward for being near gap center
            elif bird_gap_diff < 50:  # Reasonable distance
                reward += 0.2  # Small reward for being in reasonable range
            elif bird_gap_diff > 80:  # Far from gap center
                reward += -1.0  # Penalty for being far from gap center
        
        # Height penalty (encourage staying in middle, discourage being too high)
        center_y = SCREEN_HEIGHT // 2
        distance_from_center = abs(bird.y - center_y)
        
        # Stronger penalty for being too high (near ceiling)
        if bird.y < 50:  # Very close to ceiling
            reward += self.height_penalty * 5  # Strong penalty for being too high
        elif distance_from_center > 150:
            reward += self.height_penalty
        
        # Smart flap penalty (discourage unnecessary flapping)
        if action == 1:  # If the AI chose to flap
            self.consecutive_flaps += 1
            
            # Extra penalty for consecutive flapping (prevents rapid flapping)
            if self.consecutive_flaps > 2:
                reward += self.flap_penalty * 3  # Triple penalty for excessive flapping
            
            # Penalty for flapping when bird is going up (unnecessary)
            if bird.velocity < -3:  # Bird is moving upward
                reward += self.flap_penalty * 2
            
            # Strong penalty for flapping when bird is too high above gap
            if pipes:
                next_pipe = pipes[0]
                gap_center_y = next_pipe.top_height + PIPE_GAP // 2
                if bird.y < gap_center_y - 40:  # Bird is significantly above gap
                    reward += self.flap_penalty * 3  # Strong penalty for unnecessary flapping
            
            # Reward for flapping when bird is below gap (needs to go up)
            if pipes:
                next_pipe = pipes[0]
                gap_center_y = next_pipe.top_height + PIPE_GAP // 2
                if bird.y > gap_center_y + 20:  # Bird is below gap
                    reward += 0.5  # Small reward for flapping when needed
            
            # Base flap penalty
            reward += self.flap_penalty
        else:
            # Reset consecutive flap counter when not flapping
            self.consecutive_flaps = 0
            
            # Small reward for not flapping when bird is above gap
            if pipes:
                next_pipe = pipes[0]
                gap_center_y = next_pipe.top_height + PIPE_GAP // 2
                if bird.y < gap_center_y - 20:  # Bird is above gap
                    reward += 0.3  # Small reward for letting gravity work
        
        return reward
    
    def reset(self):
        """Reset reward system for new game"""
        self.last_score = 0
        self.last_bird_y = 0
        self.consecutive_flaps = 0 