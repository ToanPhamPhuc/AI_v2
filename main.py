import pygame
import sys
import random
import os
import json
from config import *

# Initialize Pygame
pygame.init()

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock()

PIPE_HEATMAP_FILE = 'pipe_heatmap.json'

# Global persistent heatmap
if os.path.exists(PIPE_HEATMAP_FILE):
    with open(PIPE_HEATMAP_FILE, 'r') as f:
        _heatmap_data = json.load(f)
        GLOBAL_PIPE_HEATMAP = {
            'top': _heatmap_data.get('top', [0]*SCREEN_HEIGHT),
            'bottom': _heatmap_data.get('bottom', [0]*SCREEN_HEIGHT)
        }
else:
    GLOBAL_PIPE_HEATMAP = {'top': [0]*SCREEN_HEIGHT, 'bottom': [0]*SCREEN_HEIGHT}

def save_pipe_heatmap():
    with open(PIPE_HEATMAP_FILE, 'w') as f:
        json.dump(GLOBAL_PIPE_HEATMAP, f)

def load_pipe_heatmap():
    global GLOBAL_PIPE_HEATMAP
    if os.path.exists(PIPE_HEATMAP_FILE):
        with open(PIPE_HEATMAP_FILE, 'r') as f:
            _heatmap_data = json.load(f)
            GLOBAL_PIPE_HEATMAP = {
                'top': _heatmap_data.get('top', [0]*SCREEN_HEIGHT),
                'bottom': _heatmap_data.get('bottom', [0]*SCREEN_HEIGHT)
            }

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
        self.hit_ceiling = False  # Track ceiling collision

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def move(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        
        # Handle ceiling collision (treat as wall, not death)
        if self.y - self.radius < 0:
            self.y = self.radius  # Stop at ceiling
            self.velocity = 0  # Stop upward movement
            self.hit_ceiling = True  # Mark ceiling collision
        else:
            self.hit_ceiling = False  # Reset ceiling collision flag

    def draw(self):
        pygame.draw.circle(screen, BLUE, (self.x, int(self.y)), self.radius)

    def reset(self):
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.hit_ceiling = False

# Pipe class
class Pipe:
    def __init__(self, x):
        self.x = x
        self.top_height = random.randint(50, SCREEN_HEIGHT - PIPE_GAP - GROUND_HEIGHT - 50)
        self.bottom_height = SCREEN_HEIGHT - self.top_height - PIPE_GAP - GROUND_HEIGHT
        self.width = 50
        self.collision_points = []

    def move(self):
        self.x -= PIPE_SPEED

    def draw(self):
        # Draw top pipe
        pygame.draw.rect(screen, GREEN, (self.x, 0, self.width, self.top_height))
        # Draw bottom pipe
        pygame.draw.rect(screen, GREEN, (self.x, SCREEN_HEIGHT - self.bottom_height - GROUND_HEIGHT, self.width, self.bottom_height))
        # Draw heatmap for top pipe
        for y in range(self.top_height):
            hits = GLOBAL_PIPE_HEATMAP['top'][y]
            if hits > 0:
                intensity = min(255, hits * 30)
                color = (0, 0, 128 + intensity//2)
                pygame.draw.line(screen, color, (self.x, y), (self.x + self.width, y))
        # Draw heatmap for bottom pipe
        for y in range(self.bottom_height):
            y_screen = SCREEN_HEIGHT - self.bottom_height - GROUND_HEIGHT + y
            hits = GLOBAL_PIPE_HEATMAP['bottom'][y_screen]
            if hits > 0:
                intensity = min(255, hits * 30)
                color = (0, 0, 128 + intensity//2)
                pygame.draw.line(screen, color, (self.x, y_screen), (self.x + self.width, y_screen))
        # Draw collision points (optional)
        for point in self.collision_points:
            x, y = point
            pygame.draw.circle(screen, (0, 0, 255), (int(x), int(y)), 3)

    def is_off_screen(self):
        return self.x + self.width < 0

    def collides_with(self, bird):
        if bird.x + bird.radius > self.x and bird.x - bird.radius < self.x + self.width:
            # Top pipe collision
            if bird.y - bird.radius < self.top_height:
                self.collision_points.append((bird.x, bird.y))
                y_min = max(0, int(bird.y - bird.radius))
                y_max = min(self.top_height, int(bird.y + bird.radius))
                for y in range(y_min, y_max):
                    if 0 <= y < SCREEN_HEIGHT:
                        GLOBAL_PIPE_HEATMAP['top'][y] += 1
                return True
            # Bottom pipe collision
            elif bird.y + bird.radius > SCREEN_HEIGHT - self.bottom_height - GROUND_HEIGHT:
                self.collision_points.append((bird.x, bird.y))
                y_min = max(0, int(bird.y - bird.radius - (SCREEN_HEIGHT - self.bottom_height - GROUND_HEIGHT)))
                y_max = min(self.bottom_height, int(bird.y + bird.radius - (SCREEN_HEIGHT - self.bottom_height - GROUND_HEIGHT)))
                for y in range(y_min, y_max):
                    y_screen = SCREEN_HEIGHT - self.bottom_height - GROUND_HEIGHT + y
                    if 0 <= y_screen < SCREEN_HEIGHT:
                        GLOBAL_PIPE_HEATMAP['bottom'][y_screen] += 1
                return True
        return False

def draw_ground():
    """Draw the brown ground layer behind everything"""
    ground_rect = pygame.Rect(0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT)
    pygame.draw.rect(screen, BROWN, ground_rect)

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
        # Clear screen with white background
        screen.fill(WHITE)
        
        # Draw ground layer FIRST (behind everything)
        draw_ground()
        
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

        # Check if bird hits the ground (removed ceiling check)
        if bird.y + bird.radius > SCREEN_HEIGHT - GROUND_HEIGHT:
            show_game_over_screen(score)
            # Reset game
            bird.reset()
            pipes = [Pipe(SCREEN_WIDTH + 200)]
            score = 0
            continue

        # Display score (on top layer)
        font = pygame.font.Font(None, 36)
        text = font.render("Score: {}".format(score), True, BLACK)
        screen.blit(text, (10, 10))
        
        # Display high score (on top layer)
        high_score_text = font.render("High Score: {}".format(high_score), True, BLACK)
        screen.blit(high_score_text, (10, 50))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 