import pygame
import sys
from training_loop import train_ai
from config import DEFAULT_EPISODES, RENDER_EVERY

if __name__ == "__main__":
    # Initialize pygame
    pygame.init()
    
    print("Starting AI Training for Flappy Bird...")
    print("The AI will learn to play the game through multiple episodes.")
    print("You can watch the training progress every 100 episodes.")
    print("Press Ctrl+C to stop training early.")
    
    try:
        # Start training with default episodes, render every RENDER_EVERY episode
        ai = train_ai(episodes=DEFAULT_EPISODES, render_every=RENDER_EVERY)
        print("Training completed successfully!")
        print("The AI has learned and saved its knowledge to q_table.json")
        
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
        print("Partial progress has been saved.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        pygame.quit()
        sys.exit() 