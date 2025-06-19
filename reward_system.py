from config import *

class RewardSystem:
    def __init__(self):
        self.last_score = 0
        self.survival_reward = SURVIVAL_REWARD
        self.score_reward = SCORE_REWARD
        self.death_penalty = DEATH_PENALTY
        self.height_penalty = HEIGHT_PENALTY
        self.flap_penalty = FLAP_PENALTY
        self.ceiling_penalty = CEILING_COLLISION_PENALTY
        self.consecutive_flaps = 0
    
    def calculate_reward(self, bird, pipes, score, game_over, action=None):
        reward = 0
        if game_over:
            reward += self.death_penalty
            return reward
        if bird.hit_ceiling:
            reward += self.ceiling_penalty
        if score > self.last_score:
            reward += self.score_reward
            self.last_score = score
        reward += self.survival_reward
        if pipes:
            next_pipe = pipes[0]
            gap_center_y = next_pipe.top_height + PIPE_GAP // 2
            distance_to_gap = abs(bird.y - gap_center_y)
            if distance_to_gap < 10:
                reward += 5.0
            elif distance_to_gap < 25:
                reward += 2.0
            elif distance_to_gap < 50:
                reward += 0.5
            elif distance_to_gap > 80:
                reward += -2.0
        center_y = SCREEN_HEIGHT // 2
        distance_from_center = abs(bird.y - center_y)
        if bird.y < 50:
            reward += self.height_penalty * 5
        elif distance_from_center > 150:
            reward += self.height_penalty
        if action == 1:
            self.consecutive_flaps += 1
            if self.consecutive_flaps > 2:
                reward += self.flap_penalty * 3
            if bird.velocity < -3:
                reward += self.flap_penalty * 2
            if pipes:
                next_pipe = pipes[0]
                gap_center_y = next_pipe.top_height + PIPE_GAP // 2
                if bird.y < gap_center_y - 40:
                    reward += self.flap_penalty * 3
                if bird.y > gap_center_y + 20:
                    reward += 0.5
            reward += self.flap_penalty
        else:
            self.consecutive_flaps = 0
            if pipes:
                next_pipe = pipes[0]
                gap_center_y = next_pipe.top_height + PIPE_GAP // 2
                if bird.y < gap_center_y - 20:
                    reward += 0.3
        if pipes:
            from main import GLOBAL_PIPE_HEATMAP
            next_pipe = pipes[0]
            gap_center_y = next_pipe.top_height + PIPE_GAP // 2
            y_int = int(bird.y)
            if 0 <= y_int < SCREEN_HEIGHT:
                top_hits = GLOBAL_PIPE_HEATMAP['top'][y_int]
                if y_int < next_pipe.top_height and top_hits >= 1:
                    reward += -5.0
        return reward
    
    def reset(self):
        self.last_score = 0
        self.consecutive_flaps = 0 