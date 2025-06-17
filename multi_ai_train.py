import pygame
import sys
import time
import threading
from config import *
import json

class MultiAITrainer:
    def __init__(self, num_ais=4):
        self.num_ais = num_ais
        self.ais = []
        self.reward_systems = []
        self.generation = 0
        self.high_score = 0
        self.best_ai_index = 0
        
        # Shared knowledge system
        self.shared_q_table = {}  # Common Q-table for all AIs
        self.knowledge_sharing_frequency = 10  # Share knowledge every N generations
        self.knowledge_sharing_strength = 0.1  # How much to blend Q-values
        
        # Initialize multiple AIs
        from ai_agent import FlappyBirdAI
        from reward_system import RewardSystem
        
        for i in range(num_ais):
            # Each AI starts with slightly different parameters for diversity
            epsilon = EPSILON * (1 + i * 0.1)  # Different exploration rates
            learning_rate = LEARNING_RATE * (1 + i * 0.05)  # Different learning rates
            ai = FlappyBirdAI(epsilon=epsilon, learning_rate=learning_rate)
            reward_system = RewardSystem()
            
            # Load existing Q-table if available
            ai.load_q_table()
            
            # Initialize shared Q-table with AI's existing knowledge
            self.merge_q_tables(ai.q_table, self.shared_q_table)
            
            self.ais.append(ai)
            self.reward_systems.append(reward_system)
        
        print(f"Initialized {num_ais} AIs with shared knowledge system")
        print(f"Knowledge sharing every {self.knowledge_sharing_frequency} generations")
    
    def merge_q_tables(self, source_q_table, target_q_table):
        """Merge Q-values from source to target with weighted averaging"""
        for state, q_values in source_q_table.items():
            if state in target_q_table:
                # Weighted average: blend existing and new knowledge
                target_q_table[state] = [
                    (1 - self.knowledge_sharing_strength) * target_q_table[state][0] + 
                    self.knowledge_sharing_strength * q_values[0],
                    (1 - self.knowledge_sharing_strength) * target_q_table[state][1] + 
                    self.knowledge_sharing_strength * q_values[1]
                ]
            else:
                # New state, add directly
                target_q_table[state] = q_values.copy()
    
    def share_knowledge(self):
        """Share knowledge between all AIs"""
        print(f"🔄 Sharing knowledge between {self.num_ais} AIs...")
        
        # Collect all Q-tables and create shared knowledge
        all_q_tables = [ai.q_table for ai in self.ais]
        
        # Reset shared Q-table
        self.shared_q_table = {}
        
        # Merge all Q-tables into shared knowledge
        for q_table in all_q_tables:
            self.merge_q_tables(q_table, self.shared_q_table)
        
        # Distribute shared knowledge back to all AIs
        for ai in self.ais:
            # Blend AI's current knowledge with shared knowledge
            for state, shared_q_values in self.shared_q_table.items():
                if state in ai.q_table:
                    # Weighted combination: 70% AI's knowledge, 30% shared knowledge
                    ai.q_table[state] = [
                        0.7 * ai.q_table[state][0] + 0.3 * shared_q_values[0],
                        0.7 * ai.q_table[state][1] + 0.3 * shared_q_values[1]
                    ]
                else:
                    # New state for this AI, adopt shared knowledge
                    ai.q_table[state] = shared_q_values.copy()
        
        print(f"✅ Knowledge shared! Shared Q-table size: {len(self.shared_q_table)} states")
    
    def train_generation(self):
        """Train all AIs for one generation"""
        from main import Bird, Pipe, screen, clock, draw_ground
        
        # Initialize game states for all AIs
        birds = [Bird() for _ in range(self.num_ais)]
        pipes_list = [[Pipe(SCREEN_WIDTH + 200)] for _ in range(self.num_ais)]
        scores = [0] * self.num_ais
        game_overs = [False] * self.num_ais
        
        # Get initial states
        states = [ai.get_state(birds[i], pipes_list[i]) for i, ai in enumerate(self.ais)]
        
        # Game loop for all AIs
        while not all(game_overs) and any(not game_over for game_over in game_overs):
            # Handle input events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print("\nStopping training...")
                        return False
                    elif event.key == pygame.K_s:
                        print("\nSaving progress...")
                        self.save_all_ais()
                    elif event.key == pygame.K_r:
                        print("\nResetting training...")
                        self.reset_all_ais()
                    elif event.key == pygame.K_d:
                        print("\nDebug info requested...")
                        self.show_debug_info()
                    elif event.key == pygame.K_k:
                        print("\nManual knowledge sharing...")
                        self.share_knowledge()
            
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
            if best_ai_idx is not None:
                self.render_frame(birds[best_ai_idx], pipes_list[best_ai_idx], scores[best_ai_idx], 
                                self.generation, self.ais[best_ai_idx].epsilon, self.high_score, 
                                self.ais[best_ai_idx], states[best_ai_idx], 
                                self.ais[best_ai_idx].get_smart_action(states[best_ai_idx], birds[best_ai_idx], pipes_list[best_ai_idx]))
        
        # End generation for all AIs
        for ai in self.ais:
            ai.end_episode()
        
        # Share knowledge periodically
        if self.generation % self.knowledge_sharing_frequency == 0:
            self.share_knowledge()
        
        # Update high score
        max_score = max(scores)
        if max_score > self.high_score:
            self.high_score = max_score
            self.best_ai_index = scores.index(max_score)
            print(f"🎉 New high score: {self.high_score} (Generation {self.generation + 1}, AI {self.best_ai_index + 1})")
            self.save_all_ais()
        
        # Reset reward systems
        for reward_system in self.reward_systems:
            reward_system.reset()
        
        self.generation += 1
        return True
    
    def get_best_performing_ai(self, scores):
        """Get the index of the best performing AI"""
        if not scores:
            return None
        return scores.index(max(scores))
    
    def save_all_ais(self):
        """Save all AI Q-tables and shared knowledge"""
        # Save individual AI Q-tables
        for i, ai in enumerate(self.ais):
            filename = f"q_table_ai_{i+1}.json"
            ai.save_q_table(filename)
        
        # Save shared Q-table
        serializable_shared_q_table = {str(k): v for k, v in self.shared_q_table.items()}
        with open("q_table_shared.json", 'w') as f:
            json.dump(serializable_shared_q_table, f)
        
        print(f"💾 All {self.num_ais} AIs + shared knowledge saved!")
    
    def reset_all_ais(self):
        """Reset all AIs and shared knowledge"""
        for ai in self.ais:
            ai.q_table = {}
        self.shared_q_table = {}
        self.high_score = 0
        self.generation = 0
        print("🔄 All AIs and shared knowledge reset! Starting fresh...")
    
    def show_debug_info(self):
        """Show debug information for all AIs and shared knowledge"""
        print(f"🔍 Multi-AI Debug Info:")
        print(f"   Generation: {self.generation + 1}")
        print(f"   High score: {self.high_score}")
        print(f"   Best AI: {self.best_ai_index + 1}")
        print(f"   Shared Q-table size: {len(self.shared_q_table)} states")
        print(f"   Knowledge sharing every {self.knowledge_sharing_frequency} generations")
        
        for i, ai in enumerate(self.ais):
            stats = ai.get_learning_stats()
            print(f"   AI {i+1}:")
            print(f"     Epsilon: {stats['epsilon']:.4f}")
            print(f"     Q-table size: {stats['q_table_size']} states")
            print(f"     Exploration rate: {stats['exploration_rate']:.2%}")
            print(f"     Avg Q-change: {stats['avg_q_change']:.4f}")
    
    def render_frame(self, bird, pipes, score, generation, epsilon, high_score, ai, state, action):
        """Render a single frame for visualization"""
        from main import screen, clock, draw_ground
        
        screen.fill(WHITE)
        
        # Draw ground layer
        draw_ground()
        
        # Draw game objects
        bird.draw()
        for pipe in pipes:
            pipe.draw()
        
        # Draw info and controls
        font = pygame.font.Font(None, 24)
        font_small = pygame.font.Font(None, 18)
        
        # Game info
        score_text = font.render(f"Score: {score}", True, BLACK)
        high_score_text = font.render(f"High Score: {high_score}", True, BLACK)
        generation_text = font.render(f"Generation: {generation}", True, BLACK)
        epsilon_text = font.render(f"Epsilon: {epsilon:.3f}", True, BLACK)
        ai_text = font.render(f"AI {self.best_ai_index + 1} (Best)", True, BLACK)
        
        # Q-values for current state
        q_values = ai.q_table.get(state, [0, 0])
        q_wait_text = font_small.render(f"Q(WAIT): {q_values[0]:.2f}", True, BLACK)
        q_flap_text = font_small.render(f"Q(FLAP): {q_values[1]:.2f}", True, BLACK)
        action_text = font_small.render(f"Action: {'FLAP' if action == 1 else 'WAIT'}", True, (255, 0, 0) if action == 1 else (0, 0, 255))
        
        # Multi-AI info
        multi_ai_text = font_small.render(f"Training {self.num_ais} AIs simultaneously", True, BLACK)
        
        # Controls
        controls_text = font_small.render("Q: Quit | S: Save | R: Reset | D: Debug | K: Share Knowledge", True, BLACK)
        
        # Position everything
        screen.blit(score_text, (10, 10))
        screen.blit(high_score_text, (10, 35))
        screen.blit(generation_text, (10, 60))
        screen.blit(epsilon_text, (10, 85))
        screen.blit(ai_text, (10, 110))
        
        # Q-values section (right side)
        screen.blit(q_wait_text, (SCREEN_WIDTH - 150, 10))
        screen.blit(q_flap_text, (SCREEN_WIDTH - 150, 30))
        screen.blit(action_text, (SCREEN_WIDTH - 150, 50))
        
        # Multi-AI info (bottom)
        screen.blit(multi_ai_text, (10, SCREEN_HEIGHT - 60))
        screen.blit(controls_text, (10, SCREEN_HEIGHT - 40))
        
        pygame.display.flip()
        clock.tick(60)

def multi_ai_train():
    """Train multiple AIs simultaneously"""
    pygame.init()
    
    print("Starting Multi-AI Training for Flappy Bird...")
    print("Training 4 AIs simultaneously for faster strategy discovery!")
    print("Press 'Q' to quit, 'S' to save, 'R' to reset, 'D' for debug info, 'K' for manual knowledge sharing.")
    
    trainer = MultiAITrainer(num_ais=4)
    
    try:
        while True:
            if not trainer.train_generation():
                break
            
            # Print progress every 50 generations
            if trainer.generation % 50 == 0:
                print(f"Generation {trainer.generation}, High Score: {trainer.high_score}")
            
            # Save progress every 100 generations
            if trainer.generation % 100 == 0:
                trainer.save_all_ais()
                print(f"✅ Saved progress! Generation {trainer.generation}, High score: {trainer.high_score}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        print(f"\n🏁 Training stopped! Final stats:")
        print(f"Total generations: {trainer.generation}")
        print(f"High score achieved: {trainer.high_score}")
        print(f"Best AI: {trainer.best_ai_index + 1}")
        trainer.save_all_ais()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    multi_ai_train() 