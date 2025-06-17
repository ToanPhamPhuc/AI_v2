import pygame
import sys
import time
from training_loop import train_ai
from config import *

def continuous_train():
    """Train the AI continuously until user stops it"""
    pygame.init()
    
    print("Starting Continuous AI Training for Flappy Bird...")
    print("The AI will learn continuously until you stop it.")
    
    # Global flag to control training
    training_active = True
    save_requested = False
    reset_requested = False
    debug_requested = False
    
    generation = 0
    best_score = 0
    recent_scores = []  # Track recent scores for average
    
    # Initialize AI and reward system once
    from ai_agent import FlappyBirdAI
    from reward_system import RewardSystem
    from main import Bird, Pipe, screen, clock, draw_ground
    
    ai = FlappyBirdAI()
    reward_system = RewardSystem()
    
    # Load existing Q-table if available
    ai.load_q_table()
    
    try:
        while training_active:
            # Initialize game state for this generation
            bird = Bird()
            pipes = [Pipe(SCREEN_WIDTH + 200)]
            score = 0
            game_over = False
            
            # Get initial state
            state = ai.get_state(bird, pipes)
            
            while not game_over and training_active:
                # Handle input events in main thread
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        training_active = False
                        game_over = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            print("\nStopping training...")
                            training_active = False
                            game_over = True
                        elif event.key == pygame.K_s:
                            print("\nSaving progress...")
                            save_requested = True
                        elif event.key == pygame.K_r:
                            print("\nResetting training...")
                            reset_requested = True
                        elif event.key == pygame.K_d:
                            print("\nDebug info requested...")
                            debug_requested = True
                
                if not training_active:
                    break
                
                # Get AI action using smart action selection
                action = ai.get_smart_action(state, bird, pipes)
                
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
                
                # Check boundaries (only ground, no ceiling death)
                if bird.y + bird.radius > SCREEN_HEIGHT - GROUND_HEIGHT:
                    game_over = True
                
                # Get next state
                next_state = ai.get_state(bird, pipes)
                
                # Calculate reward (now includes action for flap penalty)
                reward = reward_system.calculate_reward(bird, pipes, score, game_over, action)
                
                # Update Q-table
                ai.update_q_table(state, action, reward, next_state)
                
                # Update state
                state = next_state
                
                # Render every frame with controls displayed
                render_frame(bird, pipes, score, generation, ai.epsilon)
            
            if not training_active:
                break
            
            # End generation and update learning parameters
            ai.end_episode()
            
            # Track recent scores
            recent_scores.append(score)
            if len(recent_scores) > 100:
                recent_scores.pop(0)
            
            # Update best score and save if needed
            if score > best_score:
                best_score = score
                ai.save_q_table()
                print(f"🎉 New best score: {best_score} (Generation {generation + 1})")
            
            # Handle save request
            if save_requested:
                ai.save_q_table()
                print(f"💾 Progress saved! Current best score: {best_score}")
                save_requested = False
            
            # Handle reset request
            if reset_requested:
                ai.q_table = {}
                ai.save_q_table()
                best_score = 0
                recent_scores = []
                print("🔄 Training reset! Starting fresh...")
                reset_requested = False
            
            # Handle debug request
            if debug_requested:
                avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                print(f"🔍 Debug Info:")
                print(f"   Generation: {generation + 1}")
                print(f"   Epsilon: {ai.epsilon:.4f}")
                print(f"   Q-table size: {len(ai.q_table)} states")
                print(f"   Best score: {best_score}")
                print(f"   Recent average: {avg_recent:.1f}")
                print(f"   Learning rate: {ai.learning_rate}")
                debug_requested = False
            
            # Increment generation AFTER the game is complete
            generation += 1
            
            # Print progress every 50 generations
            if generation % 50 == 0:
                avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                print(f"Generation {generation}, Current Score: {score}, Best: {best_score}, Avg: {avg_recent:.1f}, ε: {ai.epsilon:.3f}")
            
            # Save progress every 100 generations
            if generation % 100 == 0:
                ai.save_q_table()
                avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                print(f"✅ Saved progress! Generation {generation}, Best score: {best_score}, Recent avg: {avg_recent:.1f}")
            
            # Reset reward system
            reward_system.reset()
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        print(f"\n🏁 Training stopped! Final stats:")
        print(f"Total generations: {generation}")
        print(f"Best score achieved: {best_score}")
        if 'ai' in locals():
            print(f"Final epsilon: {ai.epsilon:.4f}")
            print(f"Q-table size: {len(ai.q_table)} states")
        print("Final progress has been saved.")
        pygame.quit()
        sys.exit()

def render_frame(bird, pipes, score, generation, epsilon):
    """Render a single frame for visualization with controls displayed"""
    from main import screen, clock, draw_ground
    
    screen.fill(WHITE)
    
    # Draw ground layer
    draw_ground()
    
    # Draw game objects
    bird.draw()
    for pipe in pipes:
        pipe.draw()
    
    # Draw info and controls at the top
    font = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)
    
    # Game info
    score_text = font.render(f"Score: {score}", True, BLACK)
    generation_text = font.render(f"Generation: {generation}", True, BLACK)
    epsilon_text = font.render(f"Epsilon: {epsilon:.3f}", True, BLACK)
    
    # Controls
    controls_text1 = font_small.render("Q: Quit | S: Save | R: Reset | D: Debug", True, BLACK)
    
    # Position everything at the top
    screen.blit(score_text, (10, 10))
    screen.blit(generation_text, (10, 35))
    screen.blit(epsilon_text, (10, 60))
    screen.blit(controls_text1, (10, 85))
    
    pygame.display.flip()
    clock.tick(60)

if __name__ == "__main__":
    continuous_train() 