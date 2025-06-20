import pygame
import sys
from ai.dqn_agent import FlappyBirdDQN
from config.config import *
from game.main import get_adaptive_gap_center, ground_img, screen

def test_dqn_ai():
    """Test the trained DQN AI playing the game"""
    pygame.init()
    
    print("Testing Deep Q-Learning (DQN) AI...")
    print("Press SPACE to start, ESC to quit")
    
    # Initialize DQN AI
    ai = FlappyBirdDQN(epsilon=0.0)  # No exploration, only exploitation
    
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
    
    # Game loop with DQN AI
    from game.main import Bird, Pipe, clock, background_img
    from game.reward_system import RewardSystem
    
    bird = Bird()
    pipes = [Pipe(SCREEN_WIDTH + 200)]
    score = 0
    running = True
    total_reward = 0
    reward_system = RewardSystem()
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        
        # Get AI action using DQN
        state = ai.get_continuous_state(bird, pipes)
        action = ai.get_action(state)
        
        # Apply action
        if action == 1:  # Flap
            bird.flap()
        
        # Update game state
        bird.move()
        
        # Update pipes
        game_over = False
        for pipe in pipes:
            pipe.move()
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
        if bird.get_rect().bottom > SCREEN_HEIGHT - GROUND_HEIGHT:
            print(f"AI hit ground! Final score: {score}")
            game_over = True
            running = False
        
        # Calculate reward
        reward = reward_system.calculate_reward(bird, pipes, score, game_over, action)
        total_reward += reward
        
        # Render
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
        
        # Display info
        font = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 24)
        
        score_text = font.render(f"DQN AI Score: {score}", True, BLACK)
        action_text = font.render(f"Action: {'FLAP' if action else 'WAIT'}", True, BLACK)
        reward_text = font.render(f"Reward: {reward:.1f}", True, BLACK)
        total_reward_text = font.render(f"Total Reward: {total_reward:.1f}", True, BLACK)
        
        # Q-values display
        q_values = ai.get_q_values(state)
        q_wait_text = font_small.render(f"Q(WAIT): {q_values[0]:.2f}", True, BLACK)
        q_flap_text = font_small.render(f"Q(FLAP): {q_values[1]:.2f}", True, BLACK)
        
        # State info
        if pipes:
            next_pipe = pipes[0]
            gap_center_y = get_adaptive_gap_center(next_pipe.top_height)
            bird_gap_diff = bird.y - gap_center_y
            state_info = f"Bird Y: {int(bird.y)}, Vel: {bird.velocity:.1f}, Gap Diff: {bird_gap_diff:.1f}"
            state_text = font_small.render(state_info, True, BLACK)
        else:
            state_text = font_small.render("No pipes", True, BLACK)
        
        screen.blit(score_text, (10, 10))
        screen.blit(action_text, (10, 50))
        screen.blit(reward_text, (10, 90))
        screen.blit(total_reward_text, (10, 130))
        screen.blit(q_wait_text, (SCREEN_WIDTH - 150, 10))
        screen.blit(q_flap_text, (SCREEN_WIDTH - 150, 35))
        screen.blit(state_text, (10, SCREEN_HEIGHT - 40))
        
        pygame.display.flip()
        clock.tick(60)
    
    print(f"Final score: {score}")
    print(f"Total reward: {total_reward:.1f}")
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    test_dqn_ai() 