import pygame
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.dqn_agent import FlappyBirdDQN
from config.config import *
from game.main import get_adaptive_gap_center, ground_img, screen

HIGH_SCORE_FILE = 'data/scores/DQN_high_score.txt'

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

def dqn_train():
    """Train the DQN AI continuously until user stops it"""
    pygame.init()
    
    print("Starting Deep Q-Learning (DQN) Training for Flappy Bird...")
    print("The AI will learn continuously using a neural network.")
    print("Press Q to quit, A/H/C/G for visual toggles")
    
    # Global variables for visual toggles
    global show_axes, show_hitboxes, show_collision_zones, show_gap_distances
    show_axes = False
    show_hitboxes = False
    show_collision_zones = False
    show_gap_distances = True
    
    generation = 0
    best_score = 0
    recent_scores = []  # Track recent scores for average
    high_score = load_high_score()  # Track high score
    
    # Initialize DQN AI and reward system
    from game.reward_system import RewardSystem
    from game.main import Bird, Pipe, clock, save_pipe_heatmap, load_pipe_heatmap
    
    ai = FlappyBirdDQN()
    reward_system = RewardSystem()
    
    # Load pipe heatmap
    load_pipe_heatmap()
    
    try:
        while True:
            # Initialize game state for this generation
            bird = Bird()
            pipes = [Pipe(SCREEN_WIDTH + 200)]
            score = 0
            game_over = False
            
            # Get initial state (continuous)
            state = ai.get_continuous_state(bird, pipes)
            
            while not game_over:
                # Handle input events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            return
                        elif event.key == pygame.K_a:
                            show_axes = not show_axes
                        elif event.key == pygame.K_h:
                            show_hitboxes = not show_hitboxes
                        elif event.key == pygame.K_c:
                            show_collision_zones = not show_collision_zones
                        elif event.key == pygame.K_b:
                            from game.main import GLOBAL_PIPE_HEATMAP, save_pipe_heatmap
                            for arr in (GLOBAL_PIPE_HEATMAP['top'], GLOBAL_PIPE_HEATMAP['bottom']):
                                for i in range(len(arr)):
                                    arr[i] = 0
                            save_pipe_heatmap()
                        elif event.key == pygame.K_g:
                            show_gap_distances = not show_gap_distances
                
                # Get AI action using DQN
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
                        import game.main as main
                        step = 1
                        # Clamp pink line so it's always at least 50px from top and bottom pipe
                        min_dist = 50
                        max_offset = PIPE_GAP // 2 - min_dist
                        min_offset = -PIPE_GAP // 2 + min_dist
                        if bird.y < pipe.top_height + 10:
                            main.ADAPTIVE_GAP_OFFSET = min(main.ADAPTIVE_GAP_OFFSET + step, max_offset)
                        elif bird.y > pipe.top_height + PIPE_GAP - 10:
                            main.ADAPTIVE_GAP_OFFSET = max(main.ADAPTIVE_GAP_OFFSET - step, min_offset)
                        main.save_adaptive_gap_offset()
                        game_over = True
                
                # Remove off-screen pipes and add new ones
                if pipes[0].is_off_screen():
                    pipes.pop(0)
                    pipes.append(Pipe(SCREEN_WIDTH + 200))
                    score += 1
                
                # Check boundaries (only ground, no ceiling death)
                if bird.get_rect().bottom > SCREEN_HEIGHT - GROUND_HEIGHT:
                    game_over = True
                
                # Get next state
                next_state = ai.get_continuous_state(bird, pipes)
                
                # Calculate reward
                reward = reward_system.calculate_reward(bird, pipes, score, game_over, action)
                
                # Store experience in replay buffer
                ai.memory.push(state, action, reward, next_state, game_over)
                
                # Update DQN network
                ai.update()
                
                # Update state
                state = next_state
                
                # Render every frame with controls displayed
                render_frame(bird, pipes, score, generation, ai.epsilon, high_score, ai, state, action)
            
            # End generation and update learning parameters
            ai.end_episode()
            
            # Track recent scores
            recent_scores.append(score)
            if len(recent_scores) > 100:
                recent_scores.pop(0)
            
            # Update best score and high score
            if score > best_score:
                best_score = score
                ai.save_model()
                print(f"🎉 New best score: {best_score} (Generation {generation + 1})")
            
            # Update high score
            if score > high_score:
                high_score = score
                save_high_score(high_score)
            
            # Reset reward system
            reward_system.reset()
            
            # Save heatmap after each generation
            save_pipe_heatmap()
            
            # Increment generation AFTER the game is complete
            generation += 1
            
            # Print progress every 50 generations
            if generation % 50 == 0:
                avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                stats = ai.get_learning_stats()
                print(f"Generation {generation}, Current Score: {score}, Best: {best_score}, High: {high_score}, Avg: {avg_recent:.1f}, ε: {ai.epsilon:.3f}, Loss: {stats['avg_loss']:.4f}")
            
            # Save progress every 100 generations
            if generation % 100 == 0:
                ai.save_model()
                avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                print(f"✅ Saved progress! Generation {generation}, Best score: {best_score}, High score: {high_score}, Recent avg: {avg_recent:.1f}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n🏁 Training stopped! Final stats:")
        print(f"Total generations: {generation}")
        print(f"Best score achieved: {best_score}")
        print(f"High score achieved: {high_score}")
        if 'ai' in locals():
            stats = ai.get_learning_stats()
            print(f"Final epsilon: {ai.epsilon:.4f}")
            print(f"Final loss: {stats['avg_loss']:.4f}")
            print(f"Memory size: {stats['memory_size']}")
        print("Final progress has been saved.")
        save_high_score(high_score)
        pygame.quit()
        sys.exit()

def draw_ground():
    ground_y = SCREEN_HEIGHT - GROUND_HEIGHT
    for x in range(0, SCREEN_WIDTH, ground_img.get_width()):
        screen.blit(ground_img, (x, ground_y))

def render_frame(bird, pipes, score, generation, epsilon, high_score, ai, state, action):
    """Render a single frame for visualization with controls and Q-values displayed"""
    from game.main import clock, background_img
    global show_axes, show_hitboxes, show_collision_zones, show_gap_distances
    
    # Draw background first
    screen.blit(background_img, (0, 0))
    draw_ground()
    
    # Draw axes if enabled
    if show_axes:
        pygame.draw.line(screen, (200, 200, 200), (SCREEN_WIDTH//2, 0), (SCREEN_WIDTH//2, SCREEN_HEIGHT), 1)
        pygame.draw.line(screen, (200, 200, 200), (0, SCREEN_HEIGHT//2), (SCREEN_WIDTH, SCREEN_HEIGHT//2), 1)
        if pipes:
            gap_center_y = get_adaptive_gap_center(pipes[0].top_height)
            pygame.draw.line(screen, (255, 0, 255), (0, gap_center_y), (SCREEN_WIDTH, gap_center_y), 2)
    
    # Draw pipes
    for pipe in pipes:
        pipe.draw()
        if show_hitboxes:
            pygame.draw.rect(screen, RED, (pipe.x, 0, pipe.width, pipe.top_height), 3)
            pygame.draw.rect(screen, RED, (pipe.x, SCREEN_HEIGHT - pipe.bottom_height - GROUND_HEIGHT, pipe.width, pipe.bottom_height), 3)
    
    # Draw bird sprite
    bird.draw()
    
    # Draw bird hitbox if enabled
    if show_hitboxes:
        rect = bird.get_rect()
        pygame.draw.rect(screen, RED, rect, 2)
        # Draw ground collider
        pygame.draw.rect(screen, RED, (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT), 2)
    
    # Draw collision zones if enabled
    if show_collision_zones and pipes:
        for pipe in pipes:
            # Top pipe collision zone
            pygame.draw.rect(screen, (255, 0, 0, 100), (pipe.x, 0, pipe.width, pipe.top_height))
            # Bottom pipe collision zone
            pygame.draw.rect(screen, (255, 0, 0, 100), (pipe.x, SCREEN_HEIGHT - pipe.bottom_height - GROUND_HEIGHT, pipe.width, pipe.bottom_height))
    
    # Draw gap distances and pink line if enabled
    if show_gap_distances and pipes:
        pipe = pipes[0]
        gap_center_y = get_adaptive_gap_center(pipe.top_height)
        # Draw the high-contrast pink line
        pygame.draw.line(screen, (255, 0, 255), (0, gap_center_y), (SCREEN_WIDTH, gap_center_y), 3)
        # Draw the distances with black outline for contrast
        top_dist = gap_center_y - pipe.top_height
        bottom_dist = (pipe.top_height + PIPE_GAP) - gap_center_y
        font_dist = pygame.font.Font(None, 22)
        # Draw outline for top_text
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    outline = font_dist.render(f"Top→Pink: {int(top_dist)} px", True, (0,0,0))
                    screen.blit(outline, (pipe.x + pipe.width + 5 + dx, pipe.top_height + 5 + dy))
        top_text = font_dist.render(f"Top→Pink: {int(top_dist)} px", True, (255,0,255))
        screen.blit(top_text, (pipe.x + pipe.width + 5, pipe.top_height + 5))
        # Draw outline for bottom_text
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    outline = font_dist.render(f"Pink→Bottom: {int(bottom_dist)} px", True, (0,0,0))
                    screen.blit(outline, (pipe.x + pipe.width + 5 + dx, pipe.top_height + PIPE_GAP - 25 + dy))
        bottom_text = font_dist.render(f"Pink→Bottom: {int(bottom_dist)} px", True, (255,0,255))
        screen.blit(bottom_text, (pipe.x + pipe.width + 5, pipe.top_height + PIPE_GAP - 25))
    
    # Draw info and controls at the top
    font = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)
    
    # Game info
    score_text = font.render(f"Score: {score}", True, BLACK)
    high_score_text = font.render(f"High Score: {high_score}", True, BLACK)
    generation_text = font.render(f"Generation: {generation}", True, BLACK)
    epsilon_text = font.render(f"Epsilon: {epsilon:.3f}", True, BLACK)
    
    # Q-values for current state (from DQN)
    q_values = ai.get_q_values(state)
    q_wait_text = font_small.render(f"Q(WAIT): {q_values[0]:.2f}", True, BLACK)
    q_flap_text = font_small.render(f"Q(FLAP): {q_values[1]:.2f}", True, BLACK)
    
    # Current action indicator
    action_text = font_small.render(f"Action: {'FLAP' if action == 1 else 'WAIT'}", True, (255, 0, 0) if action == 1 else (0, 0, 255))
    
    # State information with continuous values
    if pipes:
        next_pipe = pipes[0]
        gap_center_y = get_adaptive_gap_center(next_pipe.top_height)
        bird_gap_diff = bird.y - gap_center_y
        state_info = f"Bird Y: {int(bird.y)}, Vel: {bird.velocity:.1f}, Pipe X: {int(next_pipe.x)}, Gap Y: {int(gap_center_y)}, Diff: {bird_gap_diff:.1f}"
        state_text = font_small.render(state_info, True, BLACK)
    else:
        state_text = font_small.render("No pipes", True, BLACK)
    
    # DQN-specific info
    stats = ai.get_learning_stats()
    dqn_info = f"Loss: {stats['avg_loss']:.4f}, Memory: {stats['memory_size']}"
    dqn_text = font_small.render(dqn_info, True, BLACK)
    
    # Controls
    controls_text1 = font_small.render("Q: Quit | A: Toggle Axes | H: Toggle Hitboxes", True, BLACK)
    controls_text2 = font_small.render("C: Toggle Collision Zones | B: Clear Pipe Heatmap | G: Gap Dists", True, BLACK)
    
    # Position everything at the top
    screen.blit(score_text, (10, 10))
    screen.blit(high_score_text, (10, 35))
    screen.blit(generation_text, (10, 60))
    screen.blit(epsilon_text, (10, 85))
    
    # Q-values section (right side)
    screen.blit(q_wait_text, (SCREEN_WIDTH - 150, 10))
    screen.blit(q_flap_text, (SCREEN_WIDTH - 150, 30))
    screen.blit(action_text, (SCREEN_WIDTH - 150, 50))
    
    # State info (bottom)
    screen.blit(state_text, (10, SCREEN_HEIGHT - 100))
    screen.blit(dqn_text, (10, SCREEN_HEIGHT - 80))
    screen.blit(controls_text1, (10, SCREEN_HEIGHT - 60))
    screen.blit(controls_text2, (10, SCREEN_HEIGHT - 40))
    
    pygame.display.flip()
    clock.tick(60)

if __name__ == "__main__":
    dqn_train() 