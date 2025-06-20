import pygame
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.ai_agent import FlappyBirdAI
from config.config import *
from game.main import get_adaptive_gap_center, ground_img, screen

HIGH_SCORE_FILE = 'data/scores/multi_ai_high_score.txt'

def load_high_score():
    try:
        with open(HIGH_SCORE_FILE, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0

def save_high_score(score):
    try:
        with open(HIGH_SCORE_FILE, 'w') as f:
            f.write(str(score))
    except Exception:
        pass

class MultiAITrainer:
    def __init__(self, num_ais=4):
        self.num_ais = num_ais
        self.ais = []
        self.reward_systems = []
        
        # Initialize multiple AIs
        for i in range(num_ais):
            ai = FlappyBirdAI()
            ai.load_q_table(f"data/jsons/q_table_ai_{i+1}.json")
            self.ais.append(ai)
            
            reward_system = __import__('game.reward_system', fromlist=['RewardSystem']).RewardSystem()
            self.reward_systems.append(reward_system)
        
        self.generation = 0
        self.best_score = 0
        self.high_score = load_high_score()
    
    def get_best_performing_ai(self, scores):
        """Get the index of the AI with the highest score"""
        return scores.index(max(scores))
    
    def train_generation(self):
        """Train all AIs for one generation"""
        from game.main import Bird, Pipe, clock, save_pipe_heatmap, load_pipe_heatmap
        
        # Initialize game state for all AIs
        birds = [Bird() for _ in range(self.num_ais)]
        pipes_list = [[Pipe(SCREEN_WIDTH + 200)] for _ in range(self.num_ais)]
        scores = [0] * self.num_ais
        game_overs = [False] * self.num_ais
        states = [ai.get_state(birds[i], pipes_list[i]) for i, ai in enumerate(self.ais)]
        
        # Load pipe heatmap
        load_pipe_heatmap()
        
        while not all(game_overs):
            # Handle input events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return False
            
            # Update each AI that's still playing
            for i in range(self.num_ais):
                if game_overs[i]:
                    continue
                
                # Get AI action
                action = self.ais[i].get_smart_action(states[i], birds[i], pipes_list[i])
                
                # Apply action
                if action == 1:
                    birds[i].flap()
                
                # Update game state
                birds[i].move()
                
                # Update pipes
                for pipe in pipes_list[i]:
                    pipe.move()
                    if pipe.collides_with(birds[i]):
                        game_overs[i] = True
                
                # Remove off-screen pipes and add new ones
                if pipes_list[i][0].is_off_screen():
                    pipes_list[i].pop(0)
                    pipes_list[i].append(Pipe(SCREEN_WIDTH + 200))
                    scores[i] += 1
                
                # Check boundaries
                if birds[i].y + birds[i].radius > SCREEN_HEIGHT - GROUND_HEIGHT:
                    game_overs[i] = True
                
                # Get next state
                next_state = self.ais[i].get_state(birds[i], pipes_list[i])
                
                # Calculate reward
                reward = self.reward_systems[i].calculate_reward(birds[i], pipes_list[i], scores[i], game_overs[i], action)
                
                # Update Q-table
                self.ais[i].update_q_table(states[i], action, reward, next_state)
                
                # Update state
                states[i] = next_state
            
            # Render the best performing AI
            best_ai_idx = self.get_best_performing_ai(scores)
            self.render_frame(birds[best_ai_idx], pipes_list[best_ai_idx], scores[best_ai_idx], 
                            self.generation, self.ais[best_ai_idx].epsilon, self.high_score, 
                            self.ais[best_ai_idx], states[best_ai_idx], 0, best_ai_idx)
        
        # End generation for all AIs
        for ai in self.ais:
            ai.end_episode()
        
        # Update best score
        max_score = max(scores)
        if max_score > self.best_score:
            self.best_score = max_score
            # Save the best performing AI's Q-table
            best_ai_idx = self.get_best_performing_ai(scores)
            self.ais[best_ai_idx].save_q_table()
            print(f"🎉 New best score: {self.best_score} (AI {best_ai_idx + 1}, Generation {self.generation + 1})")
        
        # Update high score
        if max_score > self.high_score:
            self.high_score = max_score
            save_high_score(self.high_score)
        
        # Reset reward systems
        for reward_system in self.reward_systems:
            reward_system.reset()
        
        # Save heatmap
        save_pipe_heatmap()
        
        # Save all AI Q-tables
        for i, ai in enumerate(self.ais):
            ai.save_q_table(f"data/jsons/q_table_ai_{i+1}.json")
        
        self.generation += 1
        return True
    
    def render_frame(self, bird, pipes, score, generation, epsilon, high_score, ai, state, action, ai_index):
        """Render a single frame for visualization"""
        from game.main import clock, background_img
        
        # Draw background
        screen.blit(background_img, (0, 0))
        
        # Draw ground
        ground_y = SCREEN_HEIGHT - GROUND_HEIGHT
        for x in range(0, SCREEN_WIDTH, ground_img.get_width()):
            screen.blit(ground_img, (x, ground_y))
        
        # Draw pipes
        for pipe in pipes:
            pipe.draw()
        
        # Draw bird
        bird.draw()
        
        # Draw info
        font = pygame.font.Font(None, 24)
        font_small = pygame.font.Font(None, 18)
        
        score_text = font.render(f"Score: {score}", True, BLACK)
        high_score_text = font.render(f"High Score: {high_score}", True, BLACK)
        generation_text = font.render(f"Generation: {generation}", True, BLACK)
        epsilon_text = font.render(f"Epsilon: {epsilon:.3f}", True, BLACK)
        ai_text = font.render(f"AI {ai_index + 1}", True, BLACK)
        
        # Q-values
        q_values = ai.q_table.get(state, [0, 0])
        q_wait_text = font_small.render(f"Q(WAIT): {q_values[0]:.2f}", True, BLACK)
        q_flap_text = font_small.render(f"Q(FLAP): {q_values[1]:.2f}", True, BLACK)
        
        # Position everything
        screen.blit(score_text, (10, 10))
        screen.blit(high_score_text, (10, 35))
        screen.blit(generation_text, (10, 60))
        screen.blit(epsilon_text, (10, 85))
        screen.blit(ai_text, (10, 110))
        
        screen.blit(q_wait_text, (SCREEN_WIDTH - 150, 10))
        screen.blit(q_flap_text, (SCREEN_WIDTH - 150, 30))
        
        pygame.display.flip()
        clock.tick(60)

def multi_ai_train():
    """Train multiple AIs simultaneously"""
    pygame.init()
    
    print("Starting Multi-AI Training for Flappy Bird...")
    print("Multiple AIs will learn simultaneously.")
    print("Press Q to quit")
    
    trainer = MultiAITrainer(num_ais=4)
    
    try:
        while True:
            if not trainer.train_generation():
                break
            
            # Print progress every 50 generations
            if trainer.generation % 50 == 0:
                print(f"Generation {trainer.generation}, Best: {trainer.best_score}, High: {trainer.high_score}")
            
            # Save progress every 100 generations
            if trainer.generation % 100 == 0:
                print(f"✅ Saved progress! Generation {trainer.generation}, Best score: {trainer.best_score}, High score: {trainer.high_score}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        print(f"\n🏁 Training stopped! Final stats:")
        print(f"Total generations: {trainer.generation}")
        print(f"Best score achieved: {trainer.best_score}")
        print(f"High score achieved: {trainer.high_score}")
        print("Final progress has been saved.")
        save_high_score(trainer.high_score)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    multi_ai_train() 