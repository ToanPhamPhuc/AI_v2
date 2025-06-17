import pygame
import sys
import random
import os

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Game variables
GRAVITY = 0.5
FLAP_STRENGTH = -10
PIPE_SPEED = 5
PIPE_GAP = 150

# High score file
HIGH_SCORE_FILE = "high_score.txt"

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock()

def load_high_score():
    """Load high score from file"""
    try:
        if os.path.exists(HIGH_SCORE_FILE):
            with open(HIGH_SCORE_FILE, 'r') as file:
                return int(file.read().strip())
    except (ValueError, IOError):
        pass
    return 0

def save_high_score(score):
    """Save high score to file"""
    try:
        with open(HIGH_SCORE_FILE, 'w') as file:
            file.write(str(score))
    except IOError:
        pass

# High score tracking
high_score = load_high_score()

# Bird class
class Bird:
    def __init__(self):
        self.x = 50
        self.y = SCREEN_HEIGHT // 2
        self.radius = 15
        self.velocity = 0

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def move(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def draw(self):
        pygame.draw.circle(screen, BLUE, (self.x, int(self.y)), self.radius)

    def reset(self):
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0

# Pipe class
class Pipe:
    def __init__(self, x):
        self.x = x
        self.top_height = random.randint(50, SCREEN_HEIGHT - PIPE_GAP - 50)
        self.bottom_height = SCREEN_HEIGHT - self.top_height - PIPE_GAP
        self.width = 50

    def move(self):
        self.x -= PIPE_SPEED

    def draw(self):
        pygame.draw.rect(screen, GREEN, (self.x, 0, self.width, self.top_height))
        pygame.draw.rect(screen, GREEN, (self.x, SCREEN_HEIGHT - self.bottom_height, self.width, self.bottom_height))

    def is_off_screen(self):
        return self.x + self.width < 0

    def collides_with(self, bird):
        if bird.x + bird.radius > self.x and bird.x - bird.radius < self.x + self.width:
            if bird.y - bird.radius < self.top_height or bird.y + bird.radius > SCREEN_HEIGHT - self.bottom_height:
                return True
        return False

def show_game_over_screen(score):
    global high_score
    if score > high_score:
        high_score = score
        save_high_score(high_score)  # Save new high score to file
    
    screen.fill(WHITE)
    
    # Game Over text
    font_large = pygame.font.Font(None, 72)
    game_over_text = font_large.render("GAME OVER", True, RED)
    game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 100))
    screen.blit(game_over_text, game_over_rect)
    
    # Score text
    font_medium = pygame.font.Font(None, 48)
    score_text = font_medium.render(f"Score: {score}", True, BLACK)
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 30))
    screen.blit(score_text, score_rect)
    
    # High Score text
    high_score_text = font_medium.render(f"High Score: {high_score}", True, BLACK)
    high_score_rect = high_score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 20))
    screen.blit(high_score_text, high_score_rect)
    
    # Restart instruction
    font_small = pygame.font.Font(None, 36)
    restart_text = font_small.render("Press SPACE to restart", True, BLACK)
    restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 80))
    screen.blit(restart_text, restart_rect)
    
    pygame.display.flip()
    
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

# Main game loop
def main():
    global high_score
    bird = Bird()
    pipes = [Pipe(SCREEN_WIDTH + 200)]
    score = 0
    running = True 

    while running:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bird.flap()

        bird.move()
        bird.draw()

        # Move and draw pipes
        for pipe in pipes:
            pipe.move()
            pipe.draw()
            if pipe.collides_with(bird):
                show_game_over_screen(score)
                # Reset game
                bird.reset()
                pipes = [Pipe(SCREEN_WIDTH + 200)]
                score = 0
                continue

        # Remove off-screen pipes and add new ones
        if pipes[0].is_off_screen():
            pipes.pop(0)
            pipes.append(Pipe(SCREEN_WIDTH + 200))
            score += 1

        # Check if bird hits the ground or flies too high
        if bird.y - bird.radius < 0 or bird.y + bird.radius > SCREEN_HEIGHT:
            show_game_over_screen(score)
            # Reset game
            bird.reset()
            pipes = [Pipe(SCREEN_WIDTH + 200)]
            score = 0
            continue

        # Display score
        font = pygame.font.Font(None, 36)
        text = font.render("Score: {}".format(score), True, BLACK)
        screen.blit(text, (10, 10))
        
        # Display high score
        high_score_text = font.render("High Score: {}".format(high_score), True, BLACK)
        screen.blit(high_score_text, (10, 50))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()