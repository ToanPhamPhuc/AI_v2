import pygame
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.ai_agent import FlappyBirdAI
from game.reward_system import RewardSystem
from game.main import Bird, Pipe, screen, clock, draw_ground
from config.config import *

def test_trained_ai():
    """Test the trained AI playing the game"""
    ai = FlappyBirdAI(epsilon=0.0)  # No exploration, only exploitation
    ai.load_q_table()
    reward_system = RewardSystem()
    
    print("Testing trained AI...")
    print("Press SPACE to start, ESC to quit")
    
    # Wait for user to start
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
    
    # Game loop with AI
    bird = Bird()
    pipes = [Pipe(SCREEN_WIDTH + 200)]
    score = 0
    running = True
    total_reward = 0
    
    while running:
        screen.fill(WHITE)
        draw_ground()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        # Get AI action
        state = ai.get_state(bird, pipes)
        action = ai.get_action(state)
        
        # Apply action
        if action == 1:  # Flap
            bird.flap()
        
        bird.move()
        bird.draw()
        
        # Update pipes
        game_over = False
        for pipe in pipes:
            pipe.move()
            pipe.draw()
            if pipe.collides_with(bird):
                print(f"AI crashed! Final score: {score}")
                game_over = True
                running = False
        
        # Remove off-screen pipes and add new ones
        if pipes[0].is_off_screen():
            pipes.pop(0)
            pipes.append(Pipe(SCREEN_WIDTH + 200))
            score += 1
        
        # Check boundaries
        if bird.y - bird.radius < 0 or bird.y + bird.radius > SCREEN_HEIGHT - GROUND_HEIGHT:
            print(f"AI went out of bounds! Final score: {score}")
            game_over = True
            running = False
        
        # Calculate reward
        reward = reward_system.calculate_reward(bird, pipes, score, game_over)
        total_reward += reward
        
        # Display info
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"AI Score: {score}", True, BLACK)
        action_text = font.render(f"Action: {'FLAP' if action else 'WAIT'}", True, BLACK)
        reward_text = font.render(f"Reward: {reward:.1f}", True, BLACK)
        total_reward_text = font.render(f"Total Reward: {total_reward:.1f}", True, BLACK)
        
        screen.blit(score_text, (10, 10))
        screen.blit(action_text, (10, 50))
        screen.blit(reward_text, (10, 90))
        screen.blit(total_reward_text, (10, 130))
        
        pygame.display.flip()
        clock.tick(60)
    
    print(f"Final score: {score}")
    print(f"Total reward: {total_reward:.1f}")
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    pygame.init()
    test_trained_ai() 