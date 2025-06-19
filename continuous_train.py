import pygame
import sys
from training_loop import train_ai
from config import *
from main import get_adaptive_gap_center

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
    
    # Global variables for visual toggles
    global show_axes, show_hitboxes, show_collision_zones
    show_axes = False
    show_hitboxes = False
    show_collision_zones = False
    
    generation = 0
    best_score = 0
    recent_scores = []  # Track recent scores for average
    high_score = 0  # Track high score
    
    # Initialize AI and reward system once
    from ai_agent import FlappyBirdAI
    from reward_system import RewardSystem
    from main import Bird, Pipe, screen, clock, draw_ground, save_pipe_heatmap, load_pipe_heatmap, adjust_adaptive_gap_offset
    
    ai = FlappyBirdAI()
    reward_system = RewardSystem()
    
    # Load existing Q-table if available
    ai.load_q_table()
    
    # Load pipe heatmap
    load_pipe_heatmap()
    
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
                        elif event.key == pygame.K_a:
                            show_axes = not show_axes
                            print("Axes display: " + ("ON" if show_axes else "OFF"))
                        elif event.key == pygame.K_h:
                            show_hitboxes = not show_hitboxes
                            print("Hitboxes display: " + ("ON" if show_hitboxes else "OFF"))
                        elif event.key == pygame.K_c:
                            show_collision_zones = not show_collision_zones
                            print("Collision zones display: " + ("ON" if show_collision_zones else "OFF"))
                        elif event.key == pygame.K_b:
                            from main import GLOBAL_PIPE_HEATMAP, save_pipe_heatmap
                            for arr in (GLOBAL_PIPE_HEATMAP['top'], GLOBAL_PIPE_HEATMAP['bottom']):
                                for i in range(len(arr)):
                                    arr[i] = 0
                            save_pipe_heatmap()
                            print("Pipe heatmap cleared!")
                
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
                        # Determine if death was at top or bottom pipe
                        if bird.y < pipe.top_height:
                            adjust_adaptive_gap_offset('down')
                        elif bird.y > SCREEN_HEIGHT - pipe.bottom_height - GROUND_HEIGHT:
                            adjust_adaptive_gap_offset('up')
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
                render_frame(bird, pipes, score, generation, ai.epsilon, high_score, ai, state, action)
            
            if not training_active:
                break
            
            # End generation and update learning parameters
            ai.end_episode()
            
            # Track recent scores
            recent_scores.append(score)
            if len(recent_scores) > 100:
                recent_scores.pop(0)
            
            # Update best score and high score
            if score > best_score:
                best_score = score
                ai.save_q_table()
                print(f"🎉 New best score: {best_score} (Generation {generation + 1})")
            
            # Update high score
            if score > high_score:
                high_score = score
            
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
                high_score = 0
                recent_scores = []
                print("🔄 Training reset! Starting fresh...")
                reset_requested = False
            
            # Handle debug request
            if debug_requested:
                avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                stats = ai.get_learning_stats()
                print(f"🔍 Debug Info:")
                print(f"   Generation: {generation + 1}")
                print(f"   Epsilon: {stats['epsilon']:.4f}")
                print(f"   Q-table size: {stats['q_table_size']} states")
                print(f"   Total Q-updates: {stats['total_updates']}")
                print(f"   Avg Q-change: {stats['avg_q_change']:.4f}")
                print(f"   Q-value range: [{stats['min_q_value']:.2f}, {stats['max_q_value']:.2f}]")
                print(f"   Exploration rate: {stats['exploration_rate']:.2%}")
                print(f"   Best score: {best_score}")
                print(f"   High score: {high_score}")
                print(f"   Recent average: {avg_recent:.1f}")
                print(f"   Learning rate: {ai.learning_rate}")
                debug_requested = False
            
            # Increment generation AFTER the game is complete
            generation += 1
            
            # Print progress every 50 generations
            if generation % 50 == 0:
                avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                print(f"Generation {generation}, Current Score: {score}, Best: {best_score}, High: {high_score}, Avg: {avg_recent:.1f}, ε: {ai.epsilon:.3f}")
            
            # Save progress every 100 generations
            if generation % 100 == 0:
                ai.save_q_table()
                avg_recent = sum(recent_scores) / len(recent_scores) if recent_scores else 0
                print(f"✅ Saved progress! Generation {generation}, Best score: {best_score}, High score: {high_score}, Recent avg: {avg_recent:.1f}")
            
            # Reset reward system
            reward_system.reset()
            
            # Save heatmap after each generation
            save_pipe_heatmap()
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        print(f"\n🏁 Training stopped! Final stats:")
        print(f"Total generations: {generation}")
        print(f"Best score achieved: {best_score}")
        print(f"High score achieved: {high_score}")
        if 'ai' in locals():
            print(f"Final epsilon: {ai.epsilon:.4f}")
            print(f"Q-table size: {len(ai.q_table)} states")
        print("Final progress has been saved.")
        pygame.quit()
        sys.exit()

def render_frame(bird, pipes, score, generation, epsilon, high_score, ai, state, action):
    """Render a single frame for visualization with controls and Q-values displayed"""
    from main import screen, clock, draw_ground
    
    # Global variables for visual toggles
    global show_axes, show_hitboxes, show_collision_zones
    
    screen.fill(WHITE)
    
    # Draw ground layer
    draw_ground()
    
    # Draw axes if enabled
    if show_axes:
        # Vertical center line
        pygame.draw.line(screen, (200, 200, 200), (SCREEN_WIDTH//2, 0), (SCREEN_WIDTH//2, SCREEN_HEIGHT), 1)
        # Horizontal center line
        pygame.draw.line(screen, (200, 200, 200), (0, SCREEN_HEIGHT//2), (SCREEN_WIDTH, SCREEN_HEIGHT//2), 1)
        # Gap center line (if pipes exist)
        if pipes:
            gap_center_y = get_adaptive_gap_center(pipes[0].top_height)
            pygame.draw.line(screen, (255, 0, 255), (0, gap_center_y), (SCREEN_WIDTH, gap_center_y), 2)
    
    # Draw pipes with hitboxes if enabled
    for pipe in pipes:
        pipe.draw()
        if show_hitboxes:
            # Draw pipe hitbox
            pygame.draw.rect(screen, RED, (pipe.x, 0, pipe.width, pipe.top_height), 3)
            pygame.draw.rect(screen, RED, (pipe.x, SCREEN_HEIGHT - pipe.bottom_height - GROUND_HEIGHT, pipe.width, pipe.bottom_height), 3)
    
    # Draw bird with color based on action and hitbox if enabled
    bird_color = RED if action == 1 else BLUE  # Red for flap, blue for wait
    pygame.draw.circle(screen, bird_color, (bird.x, int(bird.y)), bird.radius)
    
    if show_hitboxes:
        # Draw bird hitbox
        pygame.draw.circle(screen, RED, (bird.x, int(bird.y)), bird.radius, 3)
    
    # Draw collision zones if enabled
    if show_collision_zones and pipes:
        next_pipe = pipes[0]
        gap_center_y = next_pipe.top_height + PIPE_GAP // 2
        
        # Draw collision zones on pipes
        for pipe in pipes:
            # Top pipe collision zone
            pygame.draw.rect(screen, (255, 0, 0, 100), (pipe.x, 0, pipe.width, pipe.top_height))
            # Bottom pipe collision zone
            pygame.draw.rect(screen, (255, 0, 0, 100), (pipe.x, SCREEN_HEIGHT - pipe.bottom_height - GROUND_HEIGHT, pipe.width, pipe.bottom_height))
    
    # Draw info and controls at the top
    font = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)
    
    # Game info
    score_text = font.render(f"Score: {score}", True, BLACK)
    high_score_text = font.render(f"High Score: {high_score}", True, BLACK)
    generation_text = font.render(f"Generation: {generation}", True, BLACK)
    epsilon_text = font.render(f"Epsilon: {epsilon:.3f}", True, BLACK)
    
    # Q-values for current state
    q_values = ai.q_table.get(state, [0, 0])
    q_wait_text = font_small.render(f"Q(WAIT): {q_values[0]:.2f}", True, BLACK)
    q_flap_text = font_small.render(f"Q(FLAP): {q_values[1]:.2f}", True, BLACK)
    
    # Current action indicator
    action_text = font_small.render(f"Action: {'FLAP' if action == 1 else 'WAIT'}", True, (255, 0, 0) if action == 1 else (0, 0, 255))
    
    # State information with bird-gap difference
    if pipes:
        next_pipe = pipes[0]
        gap_center_y = next_pipe.top_height + PIPE_GAP // 2
        bird_gap_diff = bird.y - gap_center_y
        state_info = f"Bird Y: {int(bird.y)}, Vel: {bird.velocity:.1f}, Pipe X: {int(next_pipe.x)}, Gap Y: {int(gap_center_y)}, Diff: {bird_gap_diff:.1f}"
        state_text = font_small.render(state_info, True, BLACK)
    else:
        state_text = font_small.render("No pipes", True, BLACK)
    
    # Controls
    controls_text1 = font_small.render("Q: Quit | S: Save | R: Reset | D: Debug", True, BLACK)
    controls_text2 = font_small.render("A: Toggle Axes | H: Toggle Hitboxes | C: Toggle Collision Zones | B: Clear Pipe Heatmap", True, BLACK)
    
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
    screen.blit(state_text, (10, SCREEN_HEIGHT - 80))
    screen.blit(controls_text1, (10, SCREEN_HEIGHT - 60))
    screen.blit(controls_text2, (10, SCREEN_HEIGHT - 40))
    
    pygame.display.flip()
    clock.tick(60)

if __name__ == "__main__":
    continuous_train() 