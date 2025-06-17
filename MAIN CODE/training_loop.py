from AI_v2 import *

def train_ai(episodes=1000, render_every=100):
    """Train the AI agent through multiple episodes"""
    from ai_agent import FlappyBirdAI
    from reward_system import RewardSystem
    
    ai = FlappyBirdAI()
    reward_system = RewardSystem()
    
    # Load existing Q-table if available
    ai.load_q_table()
    
    best_score = 0
    
    for episode in range(episodes):
        # Initialize game state
        bird = Bird()
        pipes = [Pipe(SCREEN_WIDTH + 200)]
        score = 0
        game_over = False
        
        # Get initial state
        state = ai.get_state(bird, pipes)
        
        while not game_over:
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
            if bird.y - bird.radius < 0 or bird.y + bird.radius > SCREEN_HEIGHT:
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
            if episode % render_every == 0:
                render_frame(bird, pipes, score, episode)
        
        # Update best score
        if score > best_score:
            best_score = score
            ai.save_q_table()  # Save when we get a new best score
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}, Score: {score}, Best: {best_score}")
        
        # Reset reward system
        reward_system.reset()
    
    print(f"Training complete! Best score: {best_score}")
    return ai

def render_frame(bird, pipes, score, episode):
    """Render a single frame for visualization"""
    screen.fill(WHITE)
    
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