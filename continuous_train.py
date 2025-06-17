import pygame
import sys
import threading
import time
from training_loop import train_ai
from config import *

def continuous_train():
    """Train the AI continuously until user stops it"""
    pygame.init()
    
    print("Starting Continuous AI Training for Flappy Bird...")
    print("The AI will learn continuously until you stop it.")
    print("Press 'Q' to quit training.")
    print("Press 'S' to save current progress.")
    print("Press 'R' to reset training (start over).")
    
    # Global flag to control training
    training_active = True
    save_requested = False
    reset_requested = False
    
    def handle_input():
        nonlocal training_active, save_requested, reset_requested
        while training_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    training_active = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print("\nStopping training...")
                        training_active = False
                    elif event.key == pygame.K_s:
                        print("\nSaving progress...")
                        save_requested = True
                    elif event.key == pygame.K_r:
                        print("\nResetting training...")
                        reset_requested = True
            time.sleep(0.1)  # Small delay to prevent high CPU usage
    
    # Start input handling in a separate thread
    input_thread = threading.Thread(target=handle_input)
    input_thread.daemon = True
    input_thread.start()
    
    episode = 0
    best_score = 0
    
    try:
        while training_active:
            # Train for a batch of episodes
            batch_size = 100
            print(f"\nTraining batch {episode//batch_size + 1} (episodes {episode+1}-{episode+batch_size})...")
            
            # Call training function with custom parameters
            from ai_agent import FlappyBirdAI
            from reward_system import RewardSystem
            from main import Bird, Pipe, screen, clock, draw_ground
            
            ai = FlappyBirdAI()
            reward_system = RewardSystem()
            
            # Load existing Q-table if available
            ai.load_q_table()
            
            # Train for batch_size episodes
            for batch_episode in range(batch_size):
                if not training_active:
                    break
                    
                # Initialize game state
                bird = Bird()
                pipes = [Pipe(SCREEN_WIDTH + 200)]
                score = 0
                game_over = False
                
                # Get initial state
                state = ai.get_state(bird, pipes)
                
                while not game_over and training_active:
                    # Get AI action
                    action = ai.get_action(state)
                    
                    # Apply action
                    if action == 1:  # Flap
                        bird.flap()
                    
                    # Update game state
                    bird.move()
                    
                    # Update pipes
                    for pipe in pipes:
                        pipe.move()
                        if pipe.collides_with(bird):
                            game_over = True
                    
                    # Remove off-screen pipes and add new ones
                    if pipes[0].is_off_screen():
                        pipes.pop(0)
                        pipes.append(Pipe(SCREEN_WIDTH + 200))
                        score += 1
                    
                    # Check boundaries
                    if bird.y - bird.radius < 0 or bird.y + bird.radius > SCREEN_HEIGHT - GROUND_HEIGHT:
                        game_over = True
                    
                    # Get next state
                    next_state = ai.get_state(bird, pipes)
                    
                    # Calculate reward
                    reward = reward_system.calculate_reward(bird, pipes, score, game_over)
                    
                    # Update Q-table
                    ai.update_q_table(state, action, reward, next_state)
                    
                    # Update state
                    state = next_state
                    
                    # Render occasionally
                    if episode % RENDER_EVERY == 0:
                        render_frame(bird, pipes, score, episode)
                
                # Update best score and save if needed
                if score > best_score:
                    best_score = score
                    ai.save_q_table()
                    print(f"New best score: {best_score} (Episode {episode + batch_episode + 1})")
                
                # Handle save request
                if save_requested:
                    ai.save_q_table()
                    print(f"Progress saved! Current best score: {best_score}")
                    save_requested = False
                
                # Handle reset request
                if reset_requested:
                    ai.q_table = {}
                    ai.save_q_table()
                    best_score = 0
                    print("Training reset! Starting fresh...")
                    reset_requested = False
                
                episode += 1
                
                # Print progress every 50 episodes
                if (episode + batch_episode + 1) % 50 == 0:
                    print(f"Episode {episode + batch_episode + 1}, Current Score: {score}, Best: {best_score}")
                
                # Reset reward system
                reward_system.reset()
            
            # Save progress after each batch
            ai.save_q_table()
            print(f"Batch complete! Total episodes: {episode}, Best score: {best_score}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        print(f"\nTraining stopped! Final stats:")
        print(f"Total episodes: {episode}")
        print(f"Best score achieved: {best_score}")
        print("Final progress has been saved.")
        pygame.quit()
        sys.exit()

def render_frame(bird, pipes, score, episode):
    """Render a single frame for visualization"""
    from main import screen, clock, draw_ground
    
    screen.fill(WHITE)
    
    # Draw ground layer
    draw_ground()
    
    # Draw game objects
    bird.draw()
    for pipe in pipes:
        pipe.draw()
    
    # Draw info
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, BLACK)
    episode_text = font.render(f"Episode: {episode}", True, BLACK)
    screen.blit(score_text, (10, 10))
    screen.blit(episode_text, (10, 50))
    
    pygame.display.flip()
    clock.tick(60)

if __name__ == "__main__":
    continuous_train() 