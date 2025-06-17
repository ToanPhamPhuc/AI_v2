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
        
        # Goal-seeking reward (NEW! - encourage staying near pipe gaps)
        if pipes:
            next_pipe = pipes[0]
            goal_center_y = next_pipe.top_height + PIPE_GAP // 2  # Center of the gap
            distance_to_goal = abs(bird.y - goal_center_y)
            
            # Reward for being close to the goal
            if distance_to_goal < 30:  # Very close to goal
                reward += 1.5  # Strong reward for being in the sweet spot (reduced from 2.0)
            elif distance_to_goal < 60:  # Close to goal
                reward += 0.5  # Good reward for being near goal (reduced from 1.0)
            elif distance_to_goal > 100:  # Far from goal
                reward += -0.5  # Penalty for being far from goal (reduced from -1.0)
        
        # Height penalty (encourage staying in middle, discourage being too high)
        center_y = SCREEN_HEIGHT // 2
        distance_from_center = abs(bird.y - center_y)
        
        # Stronger penalty for being too high (near ceiling)
        if bird.y < SCREEN_HEIGHT // 3:  # Upper third of screen
            reward += self.height_penalty * 3  # Triple penalty for being too high
        elif distance_from_center > 100:
            reward += self.height_penalty
        
        # Smart flap penalty (discourage unnecessary flapping)
        if action == 1:  # If the AI chose to flap
            self.consecutive_flaps += 1
            
            # Extra penalty for consecutive flapping (prevents rapid flapping)
            if self.consecutive_flaps > 2:
                reward += self.flap_penalty * 2  # Double penalty for excessive flapping
            
            # Penalty for flapping when bird is going up (unnecessary)
            if bird.velocity < -5:  # Bird is moving upward
                reward += self.flap_penalty * 1.5
            
            # NEW: Strong penalty for flapping when bird is too high above goal
            if pipes:
                next_pipe = pipes[0]
                goal_center_y = next_pipe.top_height + PIPE_GAP // 2
                if bird.y < goal_center_y - 60:  # Bird is significantly above goal (increased threshold)
                    reward += self.flap_penalty * 2  # Reduced penalty (was 3x)
            
            # NEW: Reward for flapping when bird is below goal (needs to go up)
            if pipes:
                next_pipe = pipes[0]
                goal_center_y = next_pipe.top_height + PIPE_GAP // 2
                if bird.y > goal_center_y + 20:  # Bird is below goal (increased threshold)
                    reward += 0.3  # Small reward for flapping when needed (reduced from 0.5)
            
            # Base flap penalty
            reward += self.flap_penalty
        else:
            # Reset consecutive flap counter when not flapping
            self.consecutive_flaps = 0
            
            # NEW: Small reward for not flapping when bird is above goal
            if pipes:
                next_pipe = pipes[0]
                goal_center_y = next_pipe.top_height + PIPE_GAP // 2
                if bird.y < goal_center_y - 30:  # Bird is above goal
                    reward += 0.3  # Small reward for letting gravity work
        
        return reward
    
    def reset(self):
        """Reset reward system for new game"""
        self.last_score = 0
        self.last_bird_y = 0
        self.consecutive_flaps = 0 